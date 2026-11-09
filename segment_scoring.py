"""
Paragraph-level scoring for /v2/score-segments.

The caller owns the document: it splits the manuscript, decides what to send, and aggregates
the answers. This module scores each segment on its own and says plainly when it could not:
a failed segment has no probability, rather than a 0.0 that reads as "human".
"""
import logging
import math
import os
import re
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
    predict: Callable[[List[str]], Tuple[np.ndarray, List[int]]],
    thresholds: Dict[str, float],
    *,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Score (id, text) pairs and return one result per pair, in the order given.

    predict takes texts and returns P(AI) for those it could score, with the index into its input
    that each probability belongs to.
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
            row_probabilities, kept = predict([segments[position][1] for position in scorable])
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


# A sentence ends at ., ! or ? followed by space and a capital or an opening bracket or quote.
# "e.g. the" stays one sentence because a lowercase word follows; the abbreviations below are
# never an end, so "Fig. 2" and "et al. Smith" stay whole too.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z(\[\"'])")
_ABBREVIATIONS = {"al", "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "tab", "no", "vs", "cf", "e.g", "i.e", "approx", "dr", "prof", "mr", "ms", "st"}

# Evidence settings: a sentence is named when it carries at least this share of everything the
# paragraph's score drops by across its sentences, and at most this many are named.
EVIDENCE_MIN_SHARE = 0.25
EVIDENCE_MAX_SENTENCES = 2

# A paragraph's signal is "concentrated" when its top two sentences carry at least this share of
# the drop, and that share is at least SPREAD_ABOVE_EVEN more than an even spread would give
# them (or its top sentence alone stands that far, plus 0.1, above its even share).
SPREAD_CONCENTRATED_SHARE = 0.6
SPREAD_ABOVE_EVEN = 0.2

# At most this many AI-typical words or phrases are pointed out in one paragraph.
WORDING_MAX_MATCHES = 6


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) character ranges of the text's sentences, trimmed of surrounding space."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        before = text[start:match.start()].rstrip(".!?").split()
        if before and before[-1].lower().lstrip("([") in _ABBREVIATIONS:
            continue
        spans.append((start, match.start() + len(match.group(0).rstrip())))
        start = match.end()
    spans.append((start, len(text)))

    trimmed = []
    for begin, end in spans:
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if end > begin:
            trimmed.append((begin, end))
    return trimmed


def compile_phrase(phrase: str) -> "re.Pattern[str]":
    """A phrase matched as whole words, case-insensitively, with any whitespace between its words."""
    return re.compile(r"(?<![A-Za-z])" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"(?![A-Za-z])", re.IGNORECASE)


def wording_matches(text: str, lexicon: Sequence[Dict[str, Any]], limit: int = WORDING_MAX_MATCHES) -> List[Dict[str, Any]]:
    """
    Where the text uses wording far more common in AI-written text: character ranges with the
    phrase and how many times more common it is. The strongest win where matches overlap.
    """
    found: List[Dict[str, Any]] = []
    for term in sorted(lexicon, key=lambda item: -item["ratio"]):
        for match in term["pattern"].finditer(text):
            if any(match.start() < kept["end"] and kept["start"] < match.end() for kept in found):
                continue
            found.append({"start": match.start(), "end": match.end(), "phrase": term["phrase"], "ratio": term["ratio"]})
            if len(found) >= limit:
                return sorted(found, key=lambda item: item["start"])
    return sorted(found, key=lambda item: item["start"])


def signal_spread(full: float, drops: Sequence[float], without: Sequence[float], likely_threshold: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    How a paragraph's signal is spread: concentrated in a sentence or two, or through all of it,
    and whether it still reads as likely AI whichever single sentence is left out.

    `drops` are each sentence's fall in the comparison score; `without` the paragraph's
    calibrated score with that sentence left out.
    """
    positive = sorted((drop for drop in drops if drop > 0), reverse=True)
    total = sum(positive)
    if not drops:
        return None
    concentration = positive[0] / total if total > 0 else 0.0
    top_two = sum(positive[:2]) / total if total > 0 else 0.0
    # Against what an even spread over this many sentences would give: in a three-sentence
    # paragraph any two sentences carry two thirds of the drop without standing out.
    count = len(drops)
    concentrated = (top_two >= SPREAD_CONCENTRATED_SHARE and top_two - min(1.0, 2 / count) >= SPREAD_ABOVE_EVEN) or (
        concentration - 1 / count >= SPREAD_ABOVE_EVEN + 0.1
    )
    return {
        "kind": "concentrated" if concentrated else "throughout",
        "concentration": round(concentration, 3),
        "holds_without_any": bool(likely_threshold is not None and without and min(without) >= likely_threshold),
    }


def explain_segments(
    segments: Sequence[Tuple[Any, str]],
    score: Callable[[List[str]], Tuple[np.ndarray, List[int]]],
    *,
    calibrate: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    likely_threshold: Optional[float] = None,
    lexicon: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """
    The sentences that raise each paragraph's score the most, measured, not guessed: every
    sentence is left out in turn and the paragraph re-scored. A sentence's contribution is its
    share (0-1) of the total drop across the paragraph's sentences, so a paragraph whose signal
    sits in one sentence names that one, and one whose signal is spread evenly names none.
    A paragraph of one sentence has no evidence.

    `score` should not flatten near the ends (log-odds rather than a calibrated probability), or
    a strongly flagged paragraph barely moves whichever sentence is left out.

    Character offsets index the text as sent, so the caller can underline those exact ranges.
    """
    results: List[Dict[str, Any]] = []
    texts = [text for _, text in segments]
    variants: List[str] = []
    owners: List[Tuple[int, Tuple[int, int]]] = []

    for index, text in enumerate(texts):
        spans = sentence_spans(text)
        if len(spans) < 2 or len(text.strip()) > MAX_SEGMENT_CHARS:
            continue
        for begin, end in spans:
            variants.append(" ".join(part for part in (text[:begin].strip(), text[end:].strip()) if part))
            owners.append((index, (begin, end)))

    try:
        full, full_kept = score(texts) if variants else (np.zeros(0), [])
        without, without_kept = score(variants) if variants else (np.zeros(0), [])
    except Exception:
        logger.exception("Explaining failed for a batch of %d segments", len(texts))
        return [{"id": segment_id, "evidence": [], "spread": None, "wording": [], "explain_ok": False} for segment_id, _ in segments]

    full_by_index = {kept: float(full[row]) for row, kept in enumerate(full_kept)}
    without_by_variant = {kept: float(without[row]) for row, kept in enumerate(without_kept)}

    drops: Dict[int, List[Tuple[int, int, float]]] = {}
    all_drops: Dict[int, List[float]] = {}
    without_scores: Dict[int, List[float]] = {}
    for variant, (index, (begin, end)) in enumerate(owners):
        if index not in full_by_index or variant not in without_by_variant:
            continue
        drop = full_by_index[index] - without_by_variant[variant]
        if not math.isfinite(drop):
            continue
        all_drops.setdefault(index, []).append(drop)
        without_scores.setdefault(index, []).append(without_by_variant[variant])
        if drop > 0:
            drops.setdefault(index, []).append((begin, end, drop))

    for index, (segment_id, _) in enumerate(segments):
        found = drops.get(index, [])
        total = sum(drop for _, _, drop in found)
        shares = [
            {"start": begin, "end": end, "contribution": round(drop / total, 3)}
            for begin, end, drop in found
            if total > 0 and drop / total >= EVIDENCE_MIN_SHARE
        ]
        strongest = sorted(shares, key=lambda span: -span["contribution"])[:EVIDENCE_MAX_SENTENCES]
        without = without_scores.get(index, [])
        if without and calibrate is not None:
            without = [float(value) for value in calibrate(np.array(without))]
        spread = signal_spread(full_by_index.get(index, 0.0), all_drops.get(index, []), without, likely_threshold) if index in full_by_index else None
        results.append({
            "id": segment_id,
            "evidence": sorted(strongest, key=lambda span: span["start"]),
            "spread": spread,
            "wording": wording_matches(segments[index][1], lexicon) if lexicon else [],
            "explain_ok": True,
        })

    return results
