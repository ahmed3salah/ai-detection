"""
Perplexity (and optional entropy) for AI detection.
Supports GPT-2 (general) and SciBERT (science-domain); long text via chunking.
Production detection: paragraph-level decisions + whole-text AI percentage.
"""
import logging
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# When set (e.g. DETECTION_DEBUG=1), failed segment entries include error_class (exception type name).
_DEBUG_SEGMENTS = os.environ.get("DETECTION_DEBUG", "").strip().lower() in ("1", "true", "yes")

import torch

# Device: use CUDA only if kernels work (e.g. Blackwell sm_120 fails on PyTorch < 2.7)
def _detector_cuda_works():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(2, device="cuda").sum().item()
        return True
    except RuntimeError:
        return False


_USE_CUDA = torch.cuda.is_available() and _detector_cuda_works()
if torch.cuda.is_available() and not _USE_CUDA:
    print("[Detector] CUDA kernels not supported on this GPU (e.g. Blackwell). Using CPU for perplexity.")
DEVICE = torch.device("cuda" if _USE_CUDA else "cpu")
_DEVICE_GPT2 = torch.device("cuda:0") if (_USE_CUDA and torch.cuda.device_count() >= 2) else DEVICE
_DEVICE_SCIBERT = torch.device("cuda:1") if (_USE_CUDA and torch.cuda.device_count() >= 2) else DEVICE

# Default: GPT-2 for backward compatibility and speed
_GPT2_MODEL = None
_GPT2_TOKENIZER = None
_SCIBERT_MODEL = None
_SCIBERT_TOKENIZER = None

# Chunk size for long texts (SciBERT max 512, GPT-2 we can use 1024)
MAX_LENGTH = 512
GPT2_MAX_LENGTH = 1024


def print_detector_device_status() -> None:
    """Print device setup for perplexity models (GPT-2, SciBERT). Call before feature extraction."""
    cuda_ok = torch.cuda.is_available()
    n = torch.cuda.device_count() if cuda_ok else 0
    print("[Detector] Perplexity (GPT-2 / SciBERT) device setup:")
    print(f"[Detector]   CUDA available: {cuda_ok}, device count: {n}")
    if n >= 2:
        print(f"[Detector]   GPT-2  -> cuda:0 ({torch.cuda.get_device_name(0)})")
        print(f"[Detector]   SciBERT -> cuda:1 ({torch.cuda.get_device_name(1)})")
    else:
        print(f"[Detector]   Both models -> {DEVICE}")


def _get_gpt2():
    global _GPT2_MODEL, _GPT2_TOKENIZER
    if _GPT2_MODEL is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
        _GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2").to(_DEVICE_GPT2)
        _GPT2_MODEL.eval()
    return _GPT2_MODEL, _GPT2_TOKENIZER


def _get_scibert():
    global _SCIBERT_MODEL, _SCIBERT_TOKENIZER
    if _SCIBERT_MODEL is None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        _SCIBERT_TOKENIZER = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        _SCIBERT_MODEL = AutoModelForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased").to(_DEVICE_SCIBERT)
        _SCIBERT_MODEL.eval()
    return _SCIBERT_MODEL, _SCIBERT_TOKENIZER


def _long_doc_token_ids(tokenizer, text: str, char_window: int = 4000) -> List[int]:
    """
    Full-document token id sequence without one huge tokenizer() call (avoids HF length warnings
    and reduces edge-case failures on very long inputs).
    """
    text = (text or "").strip()
    if not text:
        return []
    ids: List[int] = []
    for i in range(0, len(text), char_window):
        ids.extend(tokenizer.encode(text[i : i + char_window], add_special_tokens=False))
    return ids


def calculate_perplexity(text, model_type="gpt2", aggregate="mean"):
    """
    Compute perplexity of text (optionally chunked and aggregated).

    Args:
        text: Input string.
        model_type: "gpt2" (general) or "scibert" (science-domain).
        aggregate: For long text, "mean" or "min" over chunk perplexities.

    Returns:
        float: perplexity value.
    """
    text = (text or "").strip()
    if not text:
        return float("inf")

    if model_type == "scibert":
        return _perplexity_scibert(text, aggregate)
    return _perplexity_gpt2(text, aggregate)


def _perplexity_gpt2(text, aggregate):
    model, tokenizer = _get_gpt2()
    device = next(model.parameters()).device
    all_ids = _long_doc_token_ids(tokenizer, text)
    if not all_ids:
        return float("inf")
    perps = []
    with torch.no_grad():
        for start in range(0, len(all_ids), GPT2_MAX_LENGTH):
            chunk_ids = all_ids[start : start + GPT2_MAX_LENGTH]
            if not chunk_ids:
                continue
            input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            val = torch.exp(loss).item()
            if math.isfinite(val):
                perps.append(min(val, 1e6))
    if not perps:
        return float("inf")
    if aggregate == "min":
        return min(perps)
    return sum(perps) / len(perps)


def _perplexity_scibert(text, aggregate):
    """Pseudo-perplexity from MLM: forward with labels=input_ids, perplexity = exp(loss)."""
    model, tokenizer = _get_scibert()
    device = next(model.parameters()).device
    all_ids = _long_doc_token_ids(tokenizer, text)
    if not all_ids:
        return float("inf")
    perps = []
    with torch.no_grad():
        for start in range(0, len(all_ids), MAX_LENGTH):
            chunk_ids = all_ids[start : start + MAX_LENGTH]
            if not chunk_ids:
                continue
            enc = tokenizer.prepare_for_model(
                chunk_ids,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[attention_mask == 0] = -100
            stm = enc.get("special_tokens_mask")
            if stm is not None:
                labels[stm.to(device).bool()] = -100
            model_kw = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            if "token_type_ids" in enc:
                model_kw["token_type_ids"] = enc["token_type_ids"].to(device)
            outputs = model(**model_kw)
            loss = outputs.loss
            val = torch.exp(loss).item()
            if math.isfinite(val):
                perps.append(min(val, 1e6))
    if not perps:
        return float("inf")
    if aggregate == "min":
        return min(perps)
    return sum(perps) / len(perps)


def get_perplexity_for_features(text, model_type="scibert", aggregate="mean"):
    """
    Return perplexity (and optionally log perplexity) for use as classifier features.
    Uses science-domain model by default for scientific paper detection.
    """
    ppl = calculate_perplexity(text, model_type=model_type, aggregate=aggregate)
    log_ppl = float("inf") if ppl <= 0 else __import__("math").log(ppl)
    return ppl, log_ppl


def get_perplexity_for_features_dual(text, aggregate="mean"):
    """
    Return perplexity and log(perplexity) from both GPT-2 and SciBERT for dual-GPU features.
    GPT-2 runs on cuda:0, SciBERT on cuda:1 when 2+ GPUs are available.
    Returns (ppl_gpt2, log_ppl_gpt2, ppl_scibert, log_ppl_scibert) with clipping applied by caller.
    """
    import math
    ppl_gpt2 = calculate_perplexity(text, model_type="gpt2", aggregate=aggregate)
    ppl_scibert = calculate_perplexity(text, model_type="scibert", aggregate=aggregate)
    log_ppl_gpt2 = float("inf") if ppl_gpt2 <= 0 else math.log(ppl_gpt2)
    log_ppl_scibert = float("inf") if ppl_scibert <= 0 else math.log(ppl_scibert)
    return ppl_gpt2, log_ppl_gpt2, ppl_scibert, log_ppl_scibert


# --- Production: paragraph-level detection + whole-text decision ---

# Sentence boundary: period/exclamation/question followed by space and optional quote, then capital or digit
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"']?(?:[A-Z0-9]|\d))", re.MULTILINE)


def _split_into_sentences(block: str) -> List[str]:
    """Split a block of text into sentences (regex-based, no spaCy)."""
    block = (block or "").strip()
    if not block:
        return []
    parts = _SENTENCE_END.split(block)
    sentences = [s.strip() for s in parts if s.strip()]
    if not sentences:
        return [block] if block else []
    return sentences


def split_paragraphs(
    text: str,
    min_chars: int = 15,
    max_paragraph_chars: int = 700,
    target_sentences_per_chunk: int = 3,
) -> List[str]:
    """
    Split text into paragraphs for analysis.
    - First splits by double newline (\\n\\n).
    - Any block longer than max_paragraph_chars is further split by sentence boundaries
      into chunks of ~target_sentences_per_chunk sentences (so long single-block text
      gets multiple segments and a more robust overall score).
    Drops empty or very short blocks.
    """
    text = (text or "").strip()
    if not text:
        return []

    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs: List[str] = []

    for block in raw:
        if len(block) < min_chars:
            if len(block) >= max(1, min_chars // 2):
                paragraphs.append(block)
            continue
        if len(block) <= max_paragraph_chars:
            paragraphs.append(block)
            continue
        # Long block: split by sentences into chunks
        sentences = _split_into_sentences(block)
        if not sentences:
            paragraphs.append(block)
            continue
        n = target_sentences_per_chunk
        for i in range(0, len(sentences), n):
            chunk = " ".join(sentences[i : i + n]).strip()
            if len(chunk) >= min_chars:
                paragraphs.append(chunk)
        if not paragraphs and sentences:
            # Fallback: one chunk per sentence if very long sentences
            for s in sentences:
                if len(s) >= min_chars:
                    paragraphs.append(s)
    if not paragraphs and raw:
        paragraphs = [raw[0]] if len(raw[0]) >= max(1, min_chars // 2) else []
    if not paragraphs and text:
        paragraphs = [text] if len(text) >= min_chars else [text] if text else []
    return paragraphs


def _clamp_unit_prob(x: float) -> float:
    """Probability in [0, 1] for JSON/API; non-finite values become 0."""
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _perplexity_for_json(p: Optional[float]) -> Optional[float]:
    """JSON-safe perplexity: omit non-finite or non-positive values."""
    if p is None:
        return None
    if not math.isfinite(p) or p <= 0:
        return None
    return round(min(float(p), 1e9), 6)


def _perplexity_to_ai_signal(perplexity: float, ref_perplexity: float = 50.0) -> float:
    """
    Map perplexity to a 0–1 "AI-like" signal. Lower perplexity → smoother text → higher signal.
    ref_perplexity: typical boundary (e.g. 50); below = more AI-like, above = more human-like.
    """
    if perplexity <= 0 or ref_perplexity <= 0:
        return 0.5
    return 1.0 / (1.0 + perplexity / ref_perplexity)


def run_detection(
    text: str,
    classifier: Any,
    build_features_fn: Callable[[str], Any],
    *,
    min_paragraph_chars: int = 15,
    max_paragraph_chars: int = 700,
    target_sentences_per_chunk: int = 3,
    decision_threshold: float = 0.5,
    weight_by_length: bool = True,
    preview_len: int = 80,
    use_perplexity: bool = True,
    perplexity_weight: float = 0.25,
    ref_perplexity: float = 50.0,
    perplexity_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run AI detection per paragraph and aggregate to a whole-text decision.
    Uses both classifier probabilities and perplexity (low ppl → more AI-like).
    Long single-block text is split by sentences into multiple segments.

    Args:
        text: Input text (can be multi-paragraph or one long block).
        classifier: Fitted classifier with predict_proba(X)[:, 1] = P(AI).
        build_features_fn: Function that takes a string and returns a feature matrix (one row).
        min_paragraph_chars: Ignore segments shorter than this.
        max_paragraph_chars: Blocks longer than this are split by sentences.
        target_sentences_per_chunk: When splitting long blocks, aim for this many sentences per segment.
        decision_threshold: P(AI) >= this → paragraph/overall labeled "ai".
        weight_by_length: If True, overall score = length-weighted mean of paragraph probs.
        preview_len: Max chars of each paragraph to include in response.
        use_perplexity: If True, blend classifier score with perplexity-based signal.
        perplexity_weight: Weight of perplexity in final score (0–1); rest is classifier.
        ref_perplexity: Reference perplexity; below = more AI-like.
        perplexity_value: If set, use this instead of computing (e.g. from caller).

    Returns:
        dict with decision, ai_percentage, confidence, paragraphs, raw_overall_prob, perplexity,
        classifier_overall (pre-blend), perplexity_ai_signal (when used), classifier_segments_failed,
        and per-paragraph scoring_ok (False if classifier raised).
    """
    paragraphs = split_paragraphs(
        text,
        min_chars=min_paragraph_chars,
        max_paragraph_chars=max_paragraph_chars,
        target_sentences_per_chunk=target_sentences_per_chunk,
    )
    if not paragraphs:
        return {
            "decision": "human",
            "ai_percentage": 0.0,
            "confidence": "low",
            "paragraph_count": 0,
            "paragraphs": [],
            "raw_overall_prob": 0.0,
            "perplexity": None,
            "classifier_overall": 0.0,
            "perplexity_ai_signal": None,
            "classifier_segments_failed": 0,
            "message": "No sufficient text to analyze.",
        }

    para_results: List[Dict[str, Any]] = []
    probs: List[float] = []
    lengths: List[int] = []
    classifier_segments_failed = 0

    for i, para in enumerate(paragraphs):
        scoring_ok = True
        error_class: Optional[str] = None
        try:
            feats = build_features_fn(para)
            prob_ai = _clamp_unit_prob(float(classifier.predict_proba(feats)[0, 1]))
        except Exception as e:
            scoring_ok = False
            classifier_segments_failed += 1
            prob_ai = 0.0
            error_class = type(e).__name__
            preview_dbg = (para[:60] + "…") if len(para) > 60 else para
            logger.exception(
                "Classifier scoring failed for paragraph index=%d preview=%r",
                i + 1,
                preview_dbg,
            )
        probs.append(prob_ai)
        lengths.append(len(para))
        decision = "ai" if prob_ai >= decision_threshold else "human"
        preview = (para[:preview_len] + "…") if len(para) > preview_len else para
        # scoring_ok distinguishes "model predicted ~0" from "classifier did not run" (see plan).
        seg: Dict[str, Any] = {
            "index": i + 1,
            "text_preview": preview.strip(),
            "ai_probability": round(_clamp_unit_prob(prob_ai), 6),
            "decision": decision,
            "scoring_ok": scoring_ok,
        }
        if _DEBUG_SEGMENTS and not scoring_ok and error_class is not None:
            seg["error_class"] = error_class
        para_results.append(seg)

    if weight_by_length and sum(lengths) > 0:
        classifier_overall = sum(p * L for p, L in zip(probs, lengths)) / sum(lengths)
    else:
        classifier_overall = sum(probs) / len(probs) if probs else 0.0
    if not math.isfinite(classifier_overall):
        classifier_overall = 0.0
    else:
        classifier_overall = max(0.0, min(1.0, float(classifier_overall)))

    # Perplexity: compute for full text if not provided, then blend into decision
    if perplexity_value is not None:
        ppl = perplexity_value
    elif use_perplexity and text.strip():
        try:
            ppl = calculate_perplexity(text.strip(), model_type="scibert", aggregate="mean")
        except Exception:
            ppl = None
    else:
        ppl = None

    perplexity_ai_signal: Optional[float] = None
    if use_perplexity and ppl is not None and math.isfinite(ppl) and ppl > 0:
        ppl_signal = _perplexity_to_ai_signal(ppl, ref_perplexity)
        perplexity_ai_signal = round(ppl_signal, 6)
        raw_overall = (1.0 - perplexity_weight) * classifier_overall + perplexity_weight * ppl_signal
        raw_overall = max(0.0, min(1.0, raw_overall))
    else:
        raw_overall = classifier_overall

    if not math.isfinite(raw_overall):
        raw_overall = 0.5
    else:
        raw_overall = max(0.0, min(1.0, float(raw_overall)))

    ai_percentage = round(100.0 * raw_overall, 2)
    decision = "ai" if raw_overall >= decision_threshold else "human"

    # Confidence: distance from 0.5
    dist = abs(raw_overall - 0.5)
    if dist >= 0.35:
        confidence = "high"
    elif dist >= 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    result: Dict[str, Any] = {
        "decision": decision,
        "ai_percentage": ai_percentage,
        "confidence": confidence,
        "paragraph_count": len(paragraphs),
        "paragraphs": para_results,
        "raw_overall_prob": round(raw_overall, 4),
        "perplexity": _perplexity_for_json(ppl),
        "classifier_overall": round(classifier_overall, 6),
        "perplexity_ai_signal": perplexity_ai_signal,
        "classifier_segments_failed": classifier_segments_failed,
        "message": (
            f"The text is classified as **{decision.upper()}** with an estimated "
            f"**{ai_percentage}%** likelihood of being AI-generated (confidence: {confidence})."
        ),
    }
    return result
