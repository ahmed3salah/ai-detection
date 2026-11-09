"""
Measure sentence scores (segment_scoring.score_sentences) on the test split before they ship.

- human: how many sentences of human paragraphs read as likely AI-written, overall and inside
  the paragraphs the model flags (where the report shows them);
- ai: how many sentences of AI paragraphs read as likely AI-written;
- agreement: how often a sentence's band differs from its paragraph's;
- mixed: paragraphs built from a human paragraph's first sentences followed by an AI paragraph's
  from the same paper, to measure whether the colours land on the right half.

    python scripts/evaluate_sentence_scores.py --artifacts artifacts/scibert-full-1108 \\
        --data dataset/segments/segments.jsonl --limit 4000
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import segment_scoring  # noqa: E402
from segment_dataset import TEST, load_segments, split_segments  # noqa: E402
from segment_model import SegmentModel  # noqa: E402

LIKELY = segment_scoring.BAND_LIKELY_AI
BATCH = 64
MIXED_SENTENCES_EACH = 3


def rate(hits: int, n: int) -> Dict[str, Any]:
    return {"n": n, "rate": round(hits / n, 4) if n else None}


def score_in_batches(model: SegmentModel, texts: Sequence[str]) -> List[Optional[float]]:
    scores: List[Optional[float]] = [None] * len(texts)
    for start in range(0, len(texts), BATCH):
        chunk = list(texts[start:start + BATCH])
        probabilities, kept = model.predict(chunk)
        for row, index in enumerate(kept):
            scores[start + index] = float(probabilities[row])
    return scores


def sentences_in_batches(model: SegmentModel, texts: Sequence[str], min_words: int) -> List[List[Dict[str, Any]]]:
    results: List[List[Dict[str, Any]]] = []
    for start in range(0, len(texts), BATCH // 4):
        chunk = list(texts[start:start + BATCH // 4])
        results.extend(segment_scoring.score_sentences([(i, text) for i, text in enumerate(chunk)], model.predict, model.thresholds, min_words))
    return results


def mixed_paragraphs(rows: Sequence[Dict[str, Any]], rng: random.Random, limit: int) -> List[Dict[str, Any]]:
    """A human paragraph's first sentences then an AI paragraph's, both from the same paper."""
    by_paper: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        by_paper[row["paper_id"]][row["label"]].append(row["text"])

    built = []
    for paper, texts in by_paper.items():
        if not texts[0] or not texts[1]:
            continue
        human = rng.choice(texts[0])
        ai = rng.choice(texts[1])
        human_spans = segment_scoring.sentence_spans(human)[:MIXED_SENTENCES_EACH]
        ai_spans = segment_scoring.sentence_spans(ai)[:MIXED_SENTENCES_EACH]
        if len(human_spans) < 2 or len(ai_spans) < 2:
            continue
        human_part = human[human_spans[0][0]:human_spans[-1][1]]
        ai_part = ai[ai_spans[0][0]:ai_spans[-1][1]]
        built.append({"text": human_part + " " + ai_part, "ai_from": len(human_part) + 1})
        if len(built) >= limit:
            break
    return built


def evaluate(artifacts: str, data: str, limit: int, seed: int, min_words: int = segment_scoring.SENTENCE_WINDOW_MIN_WORDS) -> Dict[str, Any]:
    model = SegmentModel(artifacts)
    config = model.config
    rows, _ = load_segments(data)
    test = split_segments(rows, config["split_seed"], config["heldout_generators"], config["heldout_domains"])[TEST]
    rng = random.Random(seed)
    sample = rng.sample(test, min(limit, len(test)))
    thresholds = model.thresholds

    started = time.perf_counter()
    paragraph_scores = score_in_batches(model, [row["text"] for row in sample])
    sentences = sentences_in_batches(model, [row["text"] for row in sample], min_words)
    seconds = time.perf_counter() - started

    counts = defaultdict(int)
    for row, paragraph, scored in zip(sample, paragraph_scores, sentences):
        if paragraph is None or not scored:
            continue
        label = "ai" if row["label"] == 1 else "human"
        paragraph_band = segment_scoring.band_for(paragraph, thresholds)
        flagged = paragraph_band == LIKELY
        likely = sum(sentence["band"] == LIKELY for sentence in scored)
        counts[f"{label}_paragraphs"] += 1
        counts[f"{label}_sentences"] += len(scored)
        counts[f"{label}_likely_sentences"] += likely
        counts[f"{label}_paragraphs_with_likely_sentence"] += likely > 0
        counts[f"{label}_band_disagreements"] += sum(sentence["band"] != paragraph_band for sentence in scored)
        if flagged:
            counts[f"{label}_flagged_paragraphs"] += 1
            counts[f"{label}_flagged_sentences"] += len(scored)
            counts[f"{label}_flagged_human_read_sentences"] += sum(sentence["band"] == segment_scoring.BAND_HUMAN for sentence in scored)

    mixed = mixed_paragraphs(test, rng, max(50, limit // 10))
    mixed_scored = sentences_in_batches(model, [item["text"] for item in mixed], min_words)
    for item, scored in zip(mixed, mixed_scored):
        for sentence in scored:
            origin = "ai" if sentence["start"] >= item["ai_from"] else "human"
            counts[f"mixed_{origin}_sentences"] += 1
            counts[f"mixed_{origin}_likely"] += sentence["band"] == LIKELY

    return {
        "artifacts": artifacts,
        "thresholds": thresholds,
        "window_min_words": min_words,
        "seconds": round(seconds, 1),
        "human": {
            "likely_sentences": rate(counts["human_likely_sentences"], counts["human_sentences"]),
            "paragraphs_with_a_likely_sentence": rate(counts["human_paragraphs_with_likely_sentence"], counts["human_paragraphs"]),
            "band_differs_from_paragraph": rate(counts["human_band_disagreements"], counts["human_sentences"]),
            "flagged_paragraphs": counts["human_flagged_paragraphs"],
        },
        "ai": {
            "likely_sentences": rate(counts["ai_likely_sentences"], counts["ai_sentences"]),
            "band_differs_from_paragraph": rate(counts["ai_band_disagreements"], counts["ai_sentences"]),
            "human_read_sentences_in_flagged": rate(counts["ai_flagged_human_read_sentences"], counts["ai_flagged_sentences"]),
        },
        "mixed": {
            "paragraphs": len(mixed),
            "human_half_likely": rate(counts["mixed_human_likely"], counts["mixed_human_sentences"]),
            "ai_half_likely": rate(counts["mixed_ai_likely"], counts["mixed_ai_sentences"]),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=4000, help="test paragraphs sampled")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-words", type=int, default=segment_scoring.SENTENCE_WINDOW_MIN_WORDS, help="sentence window size to measure")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    report = evaluate(args.artifacts, args.data, args.limit, args.seed, args.min_words)
    text = json.dumps(report, indent=2)
    print(text)
    out = Path(args.out or Path(args.artifacts) / f"sentence_evaluation_{args.min_words}.json")
    out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
