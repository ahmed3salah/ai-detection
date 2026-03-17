#!/usr/bin/env python3
"""
Merge human and AI JSONL files (arxiv-style: id, title, abstract, ai_written, ...) into one
training file. Shuffles with a fixed seed so train/val split is reproducible.
Output: dataset/data.jsonl (used by training.py when present).
"""
import argparse
import json
import random
import sys
from pathlib import Path


def load_jsonl(path: Path):
    """Yield one JSON object per line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    p = argparse.ArgumentParser(
        description="Merge human + AI JSONL into one training file. Pass multiple --ai for 10k+5k combined.",
    )
    p.add_argument("--human", type=Path, default=Path("dataset/arxiv-metadata-pre-llm-sample-15k.jsonl"),
                  help="Human (ai_written=0) JSONL path (default: 15k sample)")
    p.add_argument("--ai", type=Path, nargs="+", default=[Path("dataset/ai_generated_test_10k.jsonl")],
                  help="One or more AI (ai_written=1) JSONL paths (e.g. 10k from real + 5k from topics)")
    p.add_argument("--output", type=Path, default=Path("dataset/data.jsonl"),
                  help="Output merged JSONL (default: dataset/data.jsonl)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = p.parse_args()

    if not args.human.exists():
        print("Error: human file not found:", args.human, file=sys.stderr)
        return 1

    rows = []
    for obj in load_jsonl(args.human):
        if "abstract" in obj and "ai_written" in obj:
            rows.append(obj)
        else:
            obj["ai_written"] = 0
            rows.append(obj)
    n_human = len(rows)

    ai_counts = []
    for ai_path in args.ai:
        if not ai_path.exists():
            print("Error: AI file not found:", ai_path, file=sys.stderr)
            return 1
        n_before = len(rows)
        for obj in load_jsonl(ai_path):
            if "abstract" in obj and "ai_written" in obj:
                rows.append(obj)
            else:
                obj["ai_written"] = 1
                rows.append(obj)
        ai_counts.append((ai_path.name, len(rows) - n_before))
    n_ai = len(rows) - n_human

    random.seed(args.seed)
    random.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    ai_detail = " + ".join("%d from %s" % (c, n) for n, c in ai_counts)
    print("Merged %d human + %d AI (%s) = %d rows -> %s (shuffled, seed=%d)" % (
        n_human, n_ai, ai_detail, len(rows), args.output, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
