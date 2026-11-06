"""
Measure a train_segments.py model against the ship gates.

A gate with too little data is "not_measured", which blocks shipping just like a failure: a
1% false-positive rate means nothing on 40 paragraphs.

    python evaluate_segments.py --artifacts artifacts/<version> --data dataset/segments.jsonl --latency
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from segment_dataset import (
    HELDOUT_DOMAIN,
    HELDOUT_GENERATOR,
    TEST,
    Row,
    load_segments,
    split_segments,
    word_count,
)
from segment_model import SegmentModel

PASS = "pass"
FAIL = "fail"
NOT_MEASURED = "not_measured"

# Share of a document's words in likely_ai at which the whole document counts as flagged.
DOCUMENT_FLAG_SHARE = 0.2
LENGTH_BUCKETS = ((40, 80), (80, 120), (120, 180), (180, 300))

MIN_PARAGRAPHS = 300
MIN_GENERATOR_PARAGRAPHS = 100
MIN_DOCUMENTS = 50
MIN_BUCKET_PARAGRAPHS = 30
MAX_FEATURE_FAILURE_RATE = 0.001
MAX_SHORTCUT_DROP = 0.05

LATENCY_SEGMENTS = 25
LATENCY_THREADS = 2
LATENCY_LIMIT_SECONDS = 60.0


def wilson_upper(hits: int, n: int, z: float = 1.96) -> Optional[float]:
    """Upper end of the 95% interval: how bad a measured rate could really be."""
    if n == 0:
        return None
    p = hits / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre + margin) / denominator, 4)


def rate_at(probabilities: np.ndarray, threshold: float) -> Dict[str, Any]:
    n = int(len(probabilities))
    hits = int(np.sum(probabilities >= threshold))
    return {"n": n, "rate": round(hits / n, 4) if n else None, "upper_95": wilson_upper(hits, n)}


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> Optional[float]:
    if len(probabilities) == 0:
        return None
    index = np.minimum((probabilities * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.mean() * abs(probabilities[members].mean() - labels[members].mean())
    return round(float(total), 4)


def document_flag_rate(rows: Sequence[Row], probabilities: np.ndarray, threshold: float) -> Dict[str, Any]:
    """How many documents have at least DOCUMENT_FLAG_SHARE of their words at likely_ai."""
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for row, probability in zip(rows, probabilities):
        count = word_count(row["text"])
        words[row["paper_id"]][0] += count
        if probability >= threshold:
            words[row["paper_id"]][1] += count
    flagged = sum(1 for total, likely in words.values() if total and likely / total >= DOCUMENT_FLAG_SHARE)
    n = len(words)
    return {"n": n, "rate": round(flagged / n, 4) if n else None, "upper_95": wilson_upper(flagged, n)}


class Scored:
    """A split's rows that could be scored, with labels and calibrated probabilities."""

    def __init__(self, model: SegmentModel, rows: Sequence[Row]):
        self.total = len(rows)
        if rows:
            probabilities, kept = model.predict([row["text"] for row in rows])
        else:
            probabilities, kept = np.zeros(0), []
        self.rows = [rows[i] for i in kept]
        self.probabilities = probabilities
        self.labels = np.array([row["label"] for row in self.rows], dtype=np.int64)

    def of_label(self, label: int) -> np.ndarray:
        return self.probabilities[self.labels == label]

    def rows_of_label(self, label: int) -> List[Row]:
        return [row for row in self.rows if row["label"] == label]


def _breakdown(scored: Scored, thresholds: Dict[str, float], key) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(scored.rows):
        value = key(row)
        if value is not None:
            groups[str(value)].append(index)

    out = {}
    for name, indices in sorted(groups.items()):
        probabilities = scored.probabilities[indices]
        labels = scored.labels[indices]
        entry: Dict[str, Any] = {"n": len(indices)}
        for band, threshold in thresholds.items():
            entry[f"human_rate_{band}"] = rate_at(probabilities[labels == 0], threshold)
            entry[f"ai_rate_{band}"] = rate_at(probabilities[labels == 1], threshold)
        out[name] = entry
    return out


def length_bucket(row: Row) -> Optional[str]:
    words = word_count(row["text"])
    for low, high in LENGTH_BUCKETS:
        if low <= words < high or (high == LENGTH_BUCKETS[-1][1] and words == high):
            return f"{low}-{high}"
    return None


def _at_most(measure: Dict[str, Any], limit: float, minimum: int) -> str:
    if measure["n"] < minimum:
        return NOT_MEASURED
    return PASS if measure["rate"] <= limit else FAIL


def _at_least(measure: Dict[str, Any], limit: float, minimum: int) -> str:
    if measure["n"] < minimum:
        return NOT_MEASURED
    return PASS if measure["rate"] >= limit else FAIL


def evaluate_gates(
    *,
    human_test: Dict[str, Any],
    document: Dict[str, Any],
    ai_test: Dict[str, Any],
    ai_generator: Dict[str, Any],
    bucket_recalls: Dict[str, Dict[str, Any]],
    non_native: Dict[str, Any],
    feature_failure_rate: float,
    latency_seconds: Optional[float],
    shortcut: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Each ship gate's status, limit and measurement. Needs enough data to count as measured."""
    gates: Dict[str, Dict[str, Any]] = {
        "human_paragraph_fpr": {"limit": "<= 1%", "measured": human_test, "status": _at_most(human_test, 0.01, MIN_PARAGRAPHS)},
        "human_document_flag_rate": {"limit": "<= 1%", "measured": document, "status": _at_most(document, 0.01, MIN_DOCUMENTS)},
        "ai_recall_heldout_papers": {"limit": ">= 80%", "measured": ai_test, "status": _at_least(ai_test, 0.8, MIN_PARAGRAPHS)},
        "ai_recall_heldout_generator": {"limit": ">= 70%", "measured": ai_generator, "status": _at_least(ai_generator, 0.7, MIN_GENERATOR_PARAGRAPHS)},
    }

    enough = len(bucket_recalls) == len(LENGTH_BUCKETS) and all(b["n"] >= MIN_BUCKET_PARAGRAPHS for b in bucket_recalls.values())
    spread = None
    if enough:
        rates = [b["rate"] for b in bucket_recalls.values()]
        spread = round(max(rates) - min(rates), 4)
    gates["ai_recall_length_spread"] = {
        "limit": "<= 10 points across 40-300 words",
        "measured": {"spread": spread, "buckets": bucket_recalls},
        "status": NOT_MEASURED if spread is None else (PASS if spread <= 0.10 else FAIL),
    }

    if non_native["n"] < MIN_PARAGRAPHS or human_test["n"] < MIN_PARAGRAPHS:
        status = NOT_MEASURED
        gap = None
    else:
        gap = round(abs(non_native["rate"] - human_test["rate"]), 4)
        status = PASS if non_native["rate"] <= 0.02 and gap <= 0.01 else FAIL
    gates["non_native_fpr"] = {"limit": "<= 2%, within 1 point of the rest", "measured": {**non_native, "gap": gap}, "status": status}

    gates["feature_failures"] = {
        "limit": "<= 0.1% of rows",
        "measured": {"rate": feature_failure_rate},
        "status": PASS if feature_failure_rate <= MAX_FEATURE_FAILURE_RATE else FAIL,
    }

    drop = (shortcut or {}).get("drop")
    gates["no_parens_digits_shortcut"] = {
        "limit": f"AUROC drops <= {MAX_SHORTCUT_DROP} with parentheses and digits removed",
        "measured": shortcut or {},
        "status": NOT_MEASURED if drop is None or human_test["n"] < MIN_PARAGRAPHS else (PASS if drop <= MAX_SHORTCUT_DROP else FAIL),
    }

    gates["cpu_latency"] = {
        "limit": f"< {LATENCY_LIMIT_SECONDS:.0f} s for {LATENCY_SEGMENTS} paragraphs on {LATENCY_THREADS} threads",
        "measured": {"seconds": latency_seconds},
        "status": NOT_MEASURED if latency_seconds is None else (PASS if latency_seconds < LATENCY_LIMIT_SECONDS else FAIL),
    }
    return gates


_PARENS_AND_DIGITS = re.compile(r"[()\d]")


def strip_parens_and_digits(text: str) -> str:
    """The text without the two traces the pilot's leak check could not remove."""
    return " ".join(_PARENS_AND_DIGITS.sub(" ", text).split())


def shortcut_check(model: SegmentModel, rows: Sequence[Row], likely: float) -> Dict[str, Any]:
    """
    Score the test set again with parentheses and digits removed. If the model was leaning on them,
    as a shortcut rather than reading the writing, its AUROC drops; if not, it barely moves.
    """
    normal, stripped = Scored(model, rows), Scored(model, [{**row, "text": strip_parens_and_digits(row["text"])} for row in rows])
    if len(set(normal.labels.tolist())) < 2 or len(set(stripped.labels.tolist())) < 2:
        return {"auroc": None, "auroc_stripped": None, "drop": None}
    auroc = float(roc_auc_score(normal.labels, normal.probabilities))
    auroc_stripped = float(roc_auc_score(stripped.labels, stripped.probabilities))
    return {
        "auroc": round(auroc, 4),
        "auroc_stripped": round(auroc_stripped, 4),
        "drop": round(auroc - auroc_stripped, 4),
        "ai_recall_stripped": rate_at(stripped.of_label(1), likely)["rate"],
    }


def measure_latency(model: SegmentModel, rows: Sequence[Row]) -> Optional[float]:
    """Seconds for one full batch on two CPU threads. Run it on the serving hardware to trust it."""
    import torch

    if model.is_transformer and model.device.type != "cpu":
        # Production scores on CPU: time a CPU copy, not the GPU the evaluation ran on.
        model = SegmentModel(model.directory, device="cpu")
    texts = [row["text"] for row in rows if word_count(row["text"]) >= 120][:LATENCY_SEGMENTS]
    if len(texts) < LATENCY_SEGMENTS:
        return None
    previous = torch.get_num_threads()
    torch.set_num_threads(LATENCY_THREADS)
    try:
        started = time.perf_counter()
        model.predict(texts)
        return round(time.perf_counter() - started, 3)
    finally:
        torch.set_num_threads(previous)


def evaluate(artifacts: str, data: str, *, latency: bool = False) -> Dict[str, Any]:
    model = SegmentModel(artifacts)
    config = model.config
    rows, dataset_sha256 = load_segments(data)
    if dataset_sha256 != config["dataset_sha256"]:
        raise ValueError(
            f"{data} is not the dataset this model was trained on (sha256 {dataset_sha256[:12]}, expected "
            f"{config['dataset_sha256'][:12]}), so its held-out splits can't be trusted."
        )

    splits = split_segments(rows, config["split_seed"], config["heldout_generators"], config["heldout_domains"])
    thresholds = model.thresholds
    likely = thresholds["likely_ai"]

    scored = {name: Scored(model, splits[name]) for name in (TEST, HELDOUT_GENERATOR, HELDOUT_DOMAIN, "polished", "non_native", "e2e")}
    test = scored[TEST]

    measured_rows = sum(s.total for s in scored.values())
    failure_rate = round(1 - sum(len(s.rows) for s in scored.values()) / measured_rows, 6) if measured_rows else 0.0

    bucket_recalls: Dict[str, Dict[str, Any]] = {}
    for row, probability in zip(test.rows, test.probabilities):
        if row["label"] == 1 and length_bucket(row) is not None:
            bucket_recalls.setdefault(length_bucket(row), {"hits": 0, "n": 0})
            bucket_recalls[length_bucket(row)]["n"] += 1
            bucket_recalls[length_bucket(row)]["hits"] += int(probability >= likely)
    bucket_recalls = {
        name: {"n": b["n"], "rate": round(b["hits"] / b["n"], 4)} for name, b in sorted(bucket_recalls.items())
    }

    human_test = rate_at(test.of_label(0), likely)
    e2e = scored["e2e"]
    # Each test paper's whole human text: the paired paragraphs plus the rest of the paper.
    whole = [test, e2e, scored["non_native"]]
    document_rows = [row for part in whole for row in part.rows_of_label(0)]
    document_probabilities = np.concatenate([part.of_label(0) for part in whole]) if document_rows else np.zeros(0)
    gates = evaluate_gates(
        human_test=human_test,
        document=document_flag_rate(document_rows, document_probabilities, likely) if e2e.rows else {"n": 0, "rate": None, "upper_95": None},
        ai_test=rate_at(test.of_label(1), likely),
        ai_generator=rate_at(scored[HELDOUT_GENERATOR].of_label(1), likely),
        bucket_recalls=bucket_recalls,
        non_native=rate_at(scored["non_native"].of_label(0), likely),
        feature_failure_rate=failure_rate,
        latency_seconds=measure_latency(model, splits[TEST]) if latency else None,
        shortcut=shortcut_check(model, splits[TEST], likely),
    )

    labels = test.labels
    return {
        "model_version": config["model_version"],
        "dataset_sha256": dataset_sha256,
        "thresholds": thresholds,
        "split_counts": config["split_counts"],
        "gates": gates,
        "ready_to_ship": all(gate["status"] == PASS for gate in gates.values()),
        "metrics": {
            "auroc_test": round(float(roc_auc_score(labels, test.probabilities)), 4) if len(set(labels.tolist())) == 2 else None,
            "ece_test": expected_calibration_error(test.probabilities, labels),
            "ai_recall_heldout_domain": rate_at(scored[HELDOUT_DOMAIN].of_label(1), likely),
            "human_fpr_heldout_domain": rate_at(scored[HELDOUT_DOMAIN].of_label(0), likely),
            "e2e_ai_document_flag_rate": document_flag_rate(e2e.rows_of_label(1), e2e.of_label(1), likely),
            # No gate: how often language polish gets flagged goes in the product's caveats.
            "polished_flag_rate": {band: rate_at(scored["polished"].probabilities, t) for band, t in thresholds.items()},
        },
        "breakdowns": {
            "length": _breakdown(test, thresholds, length_bucket),
            "generator": _breakdown(test, thresholds, lambda row: row.get("generator")),
            "domain": _breakdown(test, thresholds, lambda row: row["domain"]),
            "section": _breakdown(test, thresholds, lambda row: row["section"]),
        },
    }


def summary_lines(report: Dict[str, Any]) -> List[str]:
    lines = [f"Model {report['model_version']}  thresholds {report['thresholds']}", ""]
    for name, gate in report["gates"].items():
        measured = gate["measured"]
        value = measured.get("rate", measured.get("spread", measured.get("seconds", measured.get("drop"))))
        n = measured.get("n")
        lines.append(f"  {gate['status'].upper():13s}{name:32s}{value!s:>8}  n={n!s:<7} limit {gate['limit']}")
    lines += ["", f"Ready to ship: {'YES' if report['ready_to_ship'] else 'NO'}"]
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--latency", action="store_true", help="Also time a batch. Only meaningful on the serving hardware.")
    parser.add_argument("--out", help="Default: <artifacts>/evaluation.json")
    args = parser.parse_args(argv)

    report = evaluate(args.artifacts, args.data, latency=args.latency)
    out = Path(args.out or Path(args.artifacts) / "evaluation.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print(f"\nFull report: {out}")
    return 0 if report["ready_to_ship"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
