"""
Can the classes be told apart on things that aren't the writing?

A simple classifier gets only length and PDF/formatting traces. If it separates human from AI, a
real model will happily learn the same shortcut, as the last one learned length. This runs before
money goes on the full dataset. Exit code 1 means the dataset has an artefact to fix first.

Deliberately leaves out sentence length and comma rates: those can be real differences in writing.

    python scripts/leak_check.py --data dataset/segments/segments.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_dataset import load_segments  # noqa: E402
from text_normalize import normalize_segment  # noqa: E402

MAX_AUROC = 0.65

NAMES = [
    "word_count",
    "digits_per_char",
    "non_ascii_per_char",
    "brackets_per_word",
    "parens_per_word",
    "et_al_per_word",
    "starts_lowercase",
    "ends_without_punctuation",
    "figure_or_table_mentions",
]


def artefact_features(text: str) -> List[float]:
    words = max(1, len(text.split()))
    chars = max(1, len(text))
    return [
        float(words),
        sum(c.isdigit() for c in text) / chars,
        sum(ord(c) > 127 for c in text) / chars,
        (text.count("[") + text.count("]")) / words,
        (text.count("(") + text.count(")")) / words,
        len(re.findall(r"\bet al\b", text)) / words,
        float(bool(text) and text[0].islower()),
        float(not text.rstrip().endswith((".", "!", "?"))),
        len(re.findall(r"\b(?:Fig(?:ure)?s?\.?|Tables?)\s+\d", text)) / words,
    ]


def grouped_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> float:
    """Out-of-fold AUROC, with each paper's rows kept together."""
    scores = np.zeros(len(y))
    for train, held in GroupKFold(n_splits=folds).split(X, y, groups):
        model = HistGradientBoostingClassifier(random_state=0, max_depth=3).fit(X[train], y[train])
        scores[held] = model.predict_proba(X[held])[:, 1]
    return float(roc_auc_score(y, scores))


def single_feature_auroc(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Each trace on its own, folded so a trace rarer in humans scores the same as one rarer in AI."""
    result = {}
    for column, name in enumerate(NAMES):
        if np.ptp(X[:, column]) == 0:
            result[name] = 0.5
            continue
        value = float(roc_auc_score(y, X[:, column]))
        result[name] = round(max(value, 1 - value), 3)
    return dict(sorted(result.items(), key=lambda item: -item[1]))


def check(rows: Sequence[Dict]) -> Dict:
    main_rows = [row for row in rows if not row.get("test_set")]
    y = np.array([row["label"] for row in main_rows])
    if len(set(y.tolist())) < 2:
        raise ValueError("The dataset needs both human and AI rows.")
    X = np.array([artefact_features(normalize_segment(row["text"])) for row in main_rows], dtype=np.float64)
    groups = np.array([row["paper_id"] for row in main_rows])
    auroc = grouped_auroc(X, y, groups, folds=min(5, len(set(groups.tolist()))))
    return {
        "rows": len(main_rows),
        "auroc": round(auroc, 4),
        "limit": MAX_AUROC,
        "passed": auroc <= MAX_AUROC,
        "single_feature_auroc": single_feature_auroc(X, y),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset/segments/segments.jsonl")
    args = parser.parse_args(argv)

    rows, _ = load_segments(args.data)
    report = check(rows)
    print(json.dumps(report, indent=2))
    print("PASSED: no artefact found." if report["passed"] else "FAILED: the classes separate on non-writing traces. Fix the dataset before the full run.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
