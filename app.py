import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)

import hashlib
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
import joblib
import numpy as np
from pydantic import BaseModel, Field, field_validator

from features import (
    extract_feature_rows,
    extract_features,
    extract_features_with_perplexity,
    extract_features_with_dual_perplexity,
)
from classifier_mlp import load_classifier
import segment_scoring
from text_normalize import normalize_segment

logger = logging.getLogger(__name__)

# Approximate max length by type (chars) for truncation when input is very long
MAX_CHARS = {
    "abstract": 2_000,
    "full_paper": 12_000,
    "review": 12_000,
}

_classifier = None
_config = None
_vectorizer = None
_feature_scaler = None
_feature_scaler_checked = False
_feature_mismatch_detail: Optional[str] = None
_model_version: Optional[str] = None

# One inference at a time per process: spaCy's shared vocab and the lazy model loads are not
# built for concurrent use, and parallel forward passes only fight over the same CPU cores.
# Scale with more processes (WEB_CONCURRENCY). A waiter gives up with a 503, which callers
# already treat as "try again later", so a backlog can't fill the threadpool.
_INFERENCE_LOCK = threading.Lock()
_INFERENCE_WAIT_SECONDS = float(os.environ.get("DETECTION_INFERENCE_WAIT_SECONDS", "30"))


@contextmanager
def _inference_slot() -> Iterator[None]:
    if not _INFERENCE_LOCK.acquire(timeout=_INFERENCE_WAIT_SECONDS):
        raise HTTPException(status_code=503, detail="Detection service is busy; try again shortly.")
    try:
        yield
    finally:
        _INFERENCE_LOCK.release()


def _get_config():
    global _config
    if _config is None:
        path = Path("model_config.pkl")
        if not path.exists():
            return {}
        _config = joblib.load(path)
    return _config


def _get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        path = Path("tfidf_vectorizer.pkl")
        if path.exists():
            _vectorizer = joblib.load(path)
        else:
            _vectorizer = False  # mark as "not used"
    return _vectorizer if _vectorizer else None


def _get_feature_scaler():
    """StandardScaler from training; must match model.pt trained with use_feature_scaler."""
    global _feature_scaler, _feature_scaler_checked
    if _feature_scaler_checked:
        return _feature_scaler
    _feature_scaler_checked = True
    if not _get_config().get("use_feature_scaler", True):
        return None
    path = Path("feature_scaler.pkl")
    if path.exists():
        _feature_scaler = joblib.load(path)
    return _feature_scaler


def _prepare_text(text: str, detection_type: str) -> str:
    """Truncate text by type so long papers/reviews don't exceed model capacity."""
    text = (text or "").strip()
    if not text:
        return text
    max_chars = MAX_CHARS.get(detection_type, MAX_CHARS["full_paper"])
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(maxsplit=1)[0] or text[:max_chars]


def _build_features(text: str, config: dict):
    """Build feature vector as in training: dense (+ optional perplexity) + optional tf-idf."""
    use_dual_ppl = config.get("use_dual_perplexity", False)
    use_ppl = config.get("use_perplexity_features", True)
    use_tfidf = config.get("use_tfidf", False)

    if use_dual_ppl:
        dense = np.array([extract_features_with_dual_perplexity(text)], dtype=np.float64)
    elif use_ppl:
        dense = np.array([extract_features_with_perplexity(text)], dtype=np.float64)
    else:
        dense = np.array([extract_features(text)], dtype=np.float64)

    if use_tfidf:
        vec = _get_vectorizer()
        if vec is not None:
            tfidf = vec.transform([text])
            feat = np.hstack([dense, tfidf.toarray()])
        else:
            feat = dense
    else:
        feat = dense

    scaler = _get_feature_scaler()
    if scaler is not None:
        feat = scaler.transform(feat)

    return feat


def _model_learned_from_perplexity(config: dict) -> bool:
    """
    Whether this model's perplexity features held real values in training. A scaler that saw no
    variance in all of them was fitted on the zeros the feature pipeline falls back to, so the
    weights never learned from them: feeding real perplexity would give meaningless scores.
    """
    if config.get("use_dual_perplexity", False):
        slots = 4
    elif config.get("use_perplexity_features", True):
        slots = 2
    else:
        return False
    scaler = _get_feature_scaler()
    variances = getattr(scaler, "var_", None)
    if variances is None:
        return True
    first = len(extract_features.feature_names)
    return bool(np.any(np.asarray(variances)[first : first + slots] > 0))


def _build_feature_batch(texts: List[str], config: dict) -> Tuple[np.ndarray, List[int]]:
    """
    _build_features for many texts: the feature matrix of the texts that could be measured,
    and each row's index into texts.
    """
    # Only for a model trained on normalized text. The deployed one saw raw abstracts, so cleaning
    # its input would change the features it was trained on.
    normalizing = bool(config.get("text_normalization"))
    if normalizing:
        texts = [normalize_segment(text) for text in texts]

    rows = extract_feature_rows(
        texts,
        dual_perplexity=config.get("use_dual_perplexity", False),
        perplexity=config.get("use_perplexity_features", True),
        measure_perplexity=_model_learned_from_perplexity(config),
    )
    kept = [
        i
        for i, row in enumerate(rows)
        if row is not None
        and np.isfinite(row).all()
        # Nothing left once cleaned: unscored, not a row of zeros the model would read as text.
        and (not normalizing or texts[i])
    ]
    if not kept:
        return np.zeros((0, 0), dtype=np.float64), []

    feat = np.array([rows[i] for i in kept], dtype=np.float64)

    if config.get("use_tfidf", False):
        vec = _get_vectorizer()
        if vec is not None:
            feat = np.hstack([feat, vec.transform([texts[i] for i in kept]).toarray()])

    scaler = _get_feature_scaler()
    if scaler is not None:
        feat = scaler.transform(feat)

    return feat, kept


def _get_model_version(config: dict) -> str:
    """model_config.pkl's model_version, or one derived from the weights so every model is told apart."""
    global _model_version
    if _model_version is None:
        declared = config.get("model_version")
        if isinstance(declared, str) and declared.strip():
            _model_version = declared.strip()[:64]
        else:
            weights = next((p for p in (Path("model.pt"), Path("model.pkl")) if p.exists()), None)
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:12] if weights else "unknown"
            _model_version = f"uncalibrated-{digest}"
    return _model_version


def _load_config_for_features() -> dict:
    path = Path("model_config.pkl")
    if not path.exists():
        return {}
    return joblib.load(path)


def _pytorch_feature_dimension_error(clf: Any) -> Optional[str]:
    """If clf is PyTorch MLP wrapper, ensure feature width matches model.input_size."""
    from classifier_mlp import PyTorchClassifierWrapper

    if not isinstance(clf, PyTorchClassifierWrapper):
        return None
    expected = int(clf.model.input_size)
    cfg = _load_config_for_features()
    dummy = "This is a short sample abstract for dimension validation."
    feat = _build_features(dummy, cfg)
    got = int(feat.shape[1])
    if got != expected:
        return (
            f"Feature dimension mismatch: model expects {expected}, inference produces {got}. "
            "Align model.pt with model_config.pkl and tfidf_vectorizer.pkl."
        )
    return None


def _sklearn_feature_dimension_error(clf: Any) -> Optional[str]:
    """If clf is a fitted sklearn estimator with n_features_in_, match inference feature width."""
    from classifier_mlp import PyTorchClassifierWrapper

    if isinstance(clf, PyTorchClassifierWrapper):
        return None
    raw = getattr(clf, "n_features_in_", None)
    if raw is None:
        return None
    # MagicMock defines __index__ → 1; only trust real integer-like sklearn values.
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        expected = raw
    elif isinstance(raw, np.integer):
        expected = int(raw)
    else:
        return None
    cfg = _load_config_for_features()
    dummy = "This is a short sample abstract for dimension validation."
    feat = _build_features(dummy, cfg)
    got = int(feat.shape[1])
    if got != expected:
        return (
            f"Feature dimension mismatch: estimator (e.g. RandomForest in model.pkl) expects {expected} "
            f"features, inference produces {got}. "
            "Set model_config.pkl to match training (use_perplexity_features, use_dual_perplexity, use_tfidf) "
            "or retrain with training.py using the current feature pipeline."
        )
    return None


def _feature_dimension_error(clf: Any) -> Optional[str]:
    """Return a user-facing error string if loaded classifier does not match _build_features width."""
    err = _pytorch_feature_dimension_error(clf)
    if err is not None:
        return err
    return _sklearn_feature_dimension_error(clf)


def _pytorch_weights_corrupted_error(clf: Any) -> Optional[str]:
    """Detect NaN/Inf in MLP weights (diverged training); otherwise outputs become NaN and look like P(AI)=0."""
    import torch

    from classifier_mlp import PyTorchClassifierWrapper

    if not isinstance(clf, PyTorchClassifierWrapper):
        return None
    for p in clf.model.parameters():
        if not torch.isfinite(p).all():
            return (
                "model.pt contains non-finite weights (NaN/Inf), usually from a diverged training run. "
                "Retrain with training.py (e.g. lower --lr, enable gradient clipping). "
                "The API was returning ai_probability 0.0 because invalid outputs were sanitized for JSON."
            )
    return None


def _classifier_load_error(clf: Any) -> Optional[str]:
    return (
        _feature_dimension_error(clf)
        or _pytorch_weights_corrupted_error(clf)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _feature_mismatch_detail
    _feature_mismatch_detail = None
    try:
        clf = load_classifier(config_path="model_config.pkl", model_dir=".")
    except FileNotFoundError:
        yield
        return
    err = _classifier_load_error(clf)
    if err:
        _feature_mismatch_detail = err
        logger.critical("%s", err)
    else:
        _classifier = clf
    yield


app = FastAPI(lifespan=lifespan)


def _safe_compare_secret(provided: str, expected: str) -> bool:
    if not provided or not expected or len(provided) != len(expected):
        return False
    return secrets.compare_digest(provided, expected)


def _verify_detection_api_key(request: Request) -> None:
    """When DETECTION_API_KEY is set, require Bearer token or X-Api-Key header."""
    expected = (os.environ.get("DETECTION_API_KEY") or "").strip()
    if not expected:
        return
    auth = request.headers.get("authorization") or ""
    bearer = ""
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    x_key = (request.headers.get("x-api-key") or "").strip()
    if _safe_compare_secret(bearer, expected) or _safe_compare_secret(x_key, expected):
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    """Liveness: process is up (does not verify model load)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: classifier loaded and consistent with feature pipeline."""
    if _feature_mismatch_detail is not None:
        raise HTTPException(status_code=503, detail=_feature_mismatch_detail)
    _get_classifier()
    return {"ready": True}


def _get_classifier():
    global _classifier
    if _feature_mismatch_detail is not None:
        raise HTTPException(status_code=503, detail=_feature_mismatch_detail)
    if _classifier is None:
        try:
            clf = load_classifier(config_path="model_config.pkl", model_dir=".")
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run training first to create model.pt or model.pkl.",
            ) from e
        err = _classifier_load_error(clf)
        if err is not None:
            raise HTTPException(status_code=503, detail=err)
        _classifier = clf
    return _classifier


async def _parse_detect_input(request: Request):
    """Parse text and text_type from either JSON body or form-data (multipart / urlencoded)."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text", "") or ""
        text_type = body.get("text_type", "abstract") or "abstract"
        return text, text_type

    form = await request.form()
    text = form.get("text")
    if text is None:
        raise HTTPException(status_code=422, detail="Missing 'text' field. Send JSON or form-data with 'text'.")
    if isinstance(text, UploadFile):
        text = (await text.read()).decode("utf-8", errors="replace")
    else:
        text = str(text)
    text_type = form.get("text_type") or "abstract"
    if isinstance(text_type, UploadFile):
        text_type = (await text_type.read()).decode("utf-8", errors="replace").strip() or "abstract"
    else:
        text_type = (str(text_type).strip() or "abstract")
    return text, text_type


@app.post("/detect")
async def detect(request: Request, _: None = Depends(_verify_detection_api_key)):
    """
    Detect likelihood that text is AI-generated (production).
    - Splits text into paragraphs (and by sentences for long single-block text).
    - Scores each segment; decision blends classifier + perplexity (low ppl → more AI-like).
    - Returns whole-text decision with AI percentage and per-paragraph breakdown.
    Accepts either:
    - JSON: {"text": "...", "text_type": "abstract"|"full_paper"|"review"}
    - Form-data (multipart or urlencoded): text=... and optional text_type=...
    """
    from detector import run_detection

    text, text_type = await _parse_detect_input(request)
    detection_type = text_type if text_type in ("abstract", "full_paper", "review") else "abstract"
    prepared = _prepare_text(text, detection_type)

    classifier = _get_classifier()
    config = _get_config()

    def build_features(paragraph: str):
        return _build_features(paragraph, config)

    def run_locked():
        with _inference_slot():
            return run_detection(
                prepared,
                classifier,
                build_features,
                min_paragraph_chars=15,
                max_paragraph_chars=700,
                target_sentences_per_chunk=3,
                decision_threshold=0.5,
                weight_by_length=True,
                preview_len=80,
                use_perplexity=True,
                perplexity_weight=0.25,
                ref_perplexity=50.0,
            )

    # Off the event loop, so /health still answers the load balancer during a long request.
    result = await run_in_threadpool(run_locked)

    return {
        "ok": True,
        "decision": result["decision"],
        "ai_percentage": result["ai_percentage"],
        "confidence": result["confidence"],
        "message": result["message"],
        "paragraph_count": result["paragraph_count"],
        "paragraphs": result["paragraphs"],
        "raw_overall_prob": result["raw_overall_prob"],
        "perplexity": result.get("perplexity"),
        "classifier_overall": result.get("classifier_overall"),
        "perplexity_ai_signal": result.get("perplexity_ai_signal"),
        "classifier_segments_failed": result.get("classifier_segments_failed", 0),
        "detection_type": detection_type,
    }


class SegmentIn(BaseModel):
    id: Union[int, str]
    text: str
    section: Optional[str] = None


class ScoreSegmentsRequest(BaseModel):
    document_id: Optional[str] = None
    text_type: Literal["abstract", "full_paper", "review"] = "full_paper"
    segments: List[SegmentIn] = Field(min_length=1)

    @field_validator("segments")
    @classmethod
    def segment_ids_are_unique(cls, segments: List[SegmentIn]) -> List[SegmentIn]:
        if len({str(segment.id) for segment in segments}) != len(segments):
            raise ValueError("Segment ids must be unique within a request.")
        return segments


@app.post("/v2/score-segments")
def score_segments(body: ScoreSegmentsRequest, _: None = Depends(_verify_detection_api_key)):
    """
    Score each of a document's paragraphs on its own. Stateless: the caller splits the document
    and aggregates the answers, so nothing is truncated and no document-level score is blended in.

    Sync on purpose: FastAPI runs it in a worker thread, so /health stays responsive.

    A segment that could not be scored comes back with scoring_ok=false and a null probability.
    `calibrated` is false until the model carries thresholds fitted on paragraph-level data; until
    then its probabilities and bands are provisional.
    """
    texts = [segment.text for segment in body.segments]
    too_big = segment_scoring.request_limit_error(texts)
    if too_big is not None:
        raise HTTPException(status_code=413, detail=too_big)

    classifier = _get_classifier()
    config = _get_config()
    thresholds, calibrated = segment_scoring.resolve_thresholds(config)

    with _inference_slot():
        results = segment_scoring.score_segments(
            [(segment.id, segment.text) for segment in body.segments],
            classifier,
            lambda batch: _build_feature_batch(batch, config),
            thresholds,
            debug=os.environ.get("DETECTION_DEBUG", "").strip().lower() in ("1", "true", "yes"),
        )

    return {
        "model_version": _get_model_version(config),
        "calibrated": calibrated,
        "thresholds": thresholds,
        "segments": results,
    }


@app.get("/v2/model-info")
def model_info(_: None = Depends(_verify_detection_api_key)):
    """Which model is serving, where its bands are drawn, and how much one request may carry."""
    _get_classifier()
    config = _get_config()
    thresholds, calibrated = segment_scoring.resolve_thresholds(config)
    return {
        "model_version": _get_model_version(config),
        "calibrated": calibrated,
        "thresholds": thresholds,
        "uses_perplexity": _model_learned_from_perplexity(config),
        "limits": {
            "max_segments": segment_scoring.MAX_SEGMENTS,
            "max_total_chars": segment_scoring.MAX_TOTAL_CHARS,
            "max_segment_chars": segment_scoring.MAX_SEGMENT_CHARS,
            "min_segment_chars": segment_scoring.MIN_SEGMENT_CHARS,
        },
    }
