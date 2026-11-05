"""
Paragraph-level scoring for /v2/score-segments.

The caller owns the document: it splits the manuscript, decides what to send, and aggregates
the answers. This module scores each segment on its own and says plainly when it could not:
a failed segment has no probability, rather than a 0.0 that reads as "human".
"""
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Above these the whole request is refused (413): the caller is sending the wrong thing.
MAX_SEGMENTS = int(os.environ.get("DETECTION_MAX_SEGMENTS", "50"))
MAX_TOTAL_CHARS = int(os.environ.get("DETECTION_MAX_TOTAL_CHARS", "500000"))

# A segment outside these fails on its own (scoring_ok=false), so one piece of garbled PDF
# text does not take the paragraphs around it down too.
MAX_SEGMENT_CHARS = int(os.environ.get("DETECTION_MAX_SEGMENT_CHARS", "20000"))
MIN_SEGMENT_CHARS = 15

BAND_LIKELY_AI = "likely_ai"
BAND_POSSIBLY_AI = "possibly_ai"
BAND_HUMAN = "human"

# Provisional cut-offs for a model with no calibrated ones in model_config.pkl.
DEFAULT_THRESHOLDS = {BAND_LIKELY_AI: 0.85, BAND_POSSIBLY_AI: 0.6}


def resolve_thresholds(config: Dict[str, Any]) -> Tuple[Dict[str, float], bool]:
    """The model's calibrated band thresholds, or the provisional ones. Second value: calibrated?"""
    given = config.get("segment_thresholds")
    if isinstance(given, dict):
        likely = given.get(BAND_LIKELY_AI)
        possibly = given.get(BAND_POSSIBLY_AI)
        if isinstance(likely, (int, float)) and isinstance(possibly, (int, float)) and possibly <= likely:
            return {BAND_LIKELY_AI: float(likely), BAND_POSSIBLY_AI: float(possibly)}, True
    return dict(DEFAULT_THRESHOLDS), False


def band_for(probability: float, thresholds: Dict[str, float]) -> str:
    if probability >= thresholds[BAND_LIKELY_AI]:
        return BAND_LIKELY_AI
    if probability >= thresholds[BAND_POSSIBLY_AI]:
        return BAND_POSSIBLY_AI
    return BAND_HUMAN


def request_limit_error(texts: Sequence[str]) -> Optional[str]:
    """Why a whole request is too big, or None. Segments that fail on their own don't count."""
    if len(texts) > MAX_SEGMENTS:
        return f"Too many segments: {len(texts)} sent, at most {MAX_SEGMENTS} per request."
    total = sum(len(text) for text in texts if len(text.strip()) <= MAX_SEGMENT_CHARS)
    if total > MAX_TOTAL_CHARS:
        return f"Too much text: {total} characters sent, at most {MAX_TOTAL_CHARS} per request."
    return None


def _failed(segment_id: Any, error_class: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"id": segment_id, "ai_probability": None, "band": None, "scoring_ok": False}
    if error_class is not None:
        result["error_class"] = error_class
    return result


def score_segments(
    segments: Sequence[Tuple[Any, str]],
    classifier: Any,
    build_feature_batch: Callable[[List[str]], Tuple[np.ndarray, List[int]]],
    thresholds: Dict[str, float],
    *,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Score (id, text) pairs and return one result per pair, in the order given.

    build_feature_batch takes texts and returns the feature matrix of those it could measure,
    with the indices into its input that each row belongs to.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(segments)
    scorable: List[int] = []

    for position, (segment_id, text) in enumerate(segments):
        length = len(text.strip())
        if length < MIN_SEGMENT_CHARS:
            results[position] = _failed(segment_id, "too_short")
        elif length > MAX_SEGMENT_CHARS:
            results[position] = _failed(segment_id, "too_long")
        else:
            scorable.append(position)

    probabilities: Dict[int, float] = {}
    error_class: Optional[str] = None
    if scorable:
        try:
            matrix, kept = build_feature_batch([segments[position][1] for position in scorable])
            if kept:
                row_probabilities = classifier.predict_proba(matrix)[:, 1]
                for row, kept_index in enumerate(kept):
                    probabilities[scorable[kept_index]] = float(row_probabilities[row])
        except Exception as e:
            # Ids only: the text is an unpublished manuscript.
            logger.exception("Scoring failed for a batch of %d segments", len(scorable))
            error_class = type(e).__name__

    for position in scorable:
        segment_id = segments[position][0]
        probability = probabilities.get(position)
        if probability is None or not math.isfinite(probability):
            results[position] = _failed(segment_id, error_class if debug else None)
            continue
        probability = max(0.0, min(1.0, probability))
        results[position] = {
            "id": segment_id,
            "ai_probability": round(probability, 6),
            "band": band_for(probability, thresholds),
            "scoring_ok": True,
        }

    return results  # type: ignore[return-value]
