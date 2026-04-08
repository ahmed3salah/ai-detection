import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
import joblib
import numpy as np

from features import extract_features, extract_features_with_perplexity, extract_features_with_dual_perplexity
from classifier_mlp import load_classifier

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
_feature_mismatch_detail: Optional[str] = None


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

    return feat


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
async def detect(request: Request):
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

    result = run_detection(
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
