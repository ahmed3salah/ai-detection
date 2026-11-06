"""
Merge the human and AI paragraphs into the dataset train_segments.py reads.

Human rows are the paragraphs the AI rows replaced, so the classes match in paper, section and
length. Prints the comparison to read before spending on the full run.

    python scripts/merge_segments.py
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_dataset import append_jsonl, read_jsonl  # noqa: E402

PROMPT_VERSION = "v2"

KEEP = ("id", "text", "label", "paper_id", "domain", "section", "generator", "test_set", "mode")


def trimmed(row: Dict) -> Dict:
    return {key: row[key] for key in KEEP if row.get(key) is not None}


def merge(human: Sequence[Dict], ai: Sequence[Dict], prompt_version: Optional[str] = None) -> List[Dict]:
    """
    Rows from one prompt version only: a dataset mixing versions would measure neither. Polished rows
    don't depend on the generation prompt and are always kept.
    """
    by_id = {row["id"]: row for row in human}
    rows: List[Dict] = []
    used = set()
    for row in ai:
        if prompt_version and not row.get("test_set") and row.get("prompt_version", "v1") != prompt_version:
            continue
        rows.append(trimmed(row))
        source = row["source_segment_id"]
        # A polished row stands in for its source: keeping both in one test set adds nothing.
        if not row.get("test_set") and source not in used:
            used.add(source)
            rows.append(trimmed(by_id[source]))
    return rows


def describe(rows: Sequence[Dict]) -> Dict:
    def words(selected: Sequence[Dict]) -> Dict:
        counts = sorted(len(row["text"].split()) for row in selected)
        if not counts:
            return {"n": 0}
        return {
            "n": len(counts),
            "mean_words": round(statistics.mean(counts), 1),
            "p10": counts[len(counts) // 10],
            "median": counts[len(counts) // 2],
            "p90": counts[len(counts) * 9 // 10],
        }

    main = [row for row in rows if not row.get("test_set")]
    return {
        "human": words([row for row in main if row["label"] == 0]),
        "ai": words([row for row in main if row["label"] == 1]),
        "polished": words([row for row in rows if row.get("test_set") == "polished"]),
        "papers": len({row["paper_id"] for row in rows}),
        "ai_domains": Counter(row["domain"] for row in main if row["label"] == 1).most_common(),
        "human_domains": Counter(row["domain"] for row in main if row["label"] == 0).most_common(),
        "generators": Counter(row.get("generator") for row in main if row["label"] == 1).most_common(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="dataset/segments")
    parser.add_argument("--out", help="Default: <workdir>/segments.jsonl")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, help="Which generation round to use")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    rows = merge(read_jsonl(workdir / "human_segments.jsonl"), read_jsonl(workdir / "ai_segments.jsonl"), args.prompt_version)
    out = Path(args.out or workdir / "segments.jsonl")
    out.unlink(missing_ok=True)
    append_jsonl(out, rows)
    print(json.dumps(describe(rows), indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
