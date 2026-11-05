"""
Merge the human and AI paragraphs into the dataset train_segments.py reads.

Human rows are the paragraphs the AI rows replaced, so the classes match in paper, section and
length. Prints the comparison to read before spending on the full run.

    python scripts/merge_segments.py
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

KEEP = ("id", "text", "label", "paper_id", "domain", "section", "generator", "test_set", "mode")


def trimmed(row: Dict) -> Dict:
    return {key: row[key] for key in KEEP if row.get(key) is not None}


def merge(human: Sequence[Dict], ai: Sequence[Dict]) -> List[Dict]:
    by_id = {row["id"]: row for row in human}
    rows: List[Dict] = []
    used = set()
    for row in ai:
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
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    read = lambda name: [json.loads(line) for line in (workdir / name).read_text().splitlines() if line.strip()]
    rows = merge(read("human_segments.jsonl"), read("ai_segments.jsonl"))
    out = Path(args.out or workdir / "segments.jsonl")
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps(describe(rows), indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
