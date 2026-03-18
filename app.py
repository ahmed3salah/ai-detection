import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
import joblib
import numpy as np

from features import extract_features, extract_features_with_perplexity, extract_features_with_dual_perplexity
from classifier_mlp import load_classifier

app = FastAPI()

# Approximate max length by type (chars) for truncation when input is very long
MAX_CHARS = {
    "abstract": 2_000,
    "full_paper": 12_000,
    "review": 12_000,
}

_classifier = None
_config = None
_vectorizer = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            _classifier = load_classifier(config_path="model_config.pkl", model_dir=".")
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run training first to create model.pt or model.pkl.",
            ) from e
    return _classifier


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
        "detection_type": detection_type,
    }
