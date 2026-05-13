#!/usr/bin/env python3
"""
Create a sample of arxiv-metadata-pre-llm.jsonl with a fixed number of papers.

By default uses reservoir sampling over the full file (uniform random sample, one pass).
Use --sequential to take the first N eligible rows (old behavior).

Output has the same structure as the source plus two optional fields for AI-generated text:
  - model: name of the model that generated the text (null for human papers)
  - provider: API/provider used (e.g. openai, anthropic) (null for human papers)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def _eligible(obj: Dict[str, Any], human_only: bool) -> bool:
    if human_only and obj.get("ai_written", 0) != 0:
        return False
    abstract = (obj.get("abstract") or "").strip()
    return bool(abstract)


def _ensure_model_provider(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "model" not in obj:
        obj["model"] = None
    if "provider" not in obj:
        obj["provider"] = None
    return obj


def reservoir_sample(
    path: Path,
    k: int,
    seed: int,
    human_only: bool,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    n_seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _eligible(obj, human_only=human_only):
                continue
            n_seen += 1
            obj = dict(obj)
            _ensure_model_provider(obj)
            if n_seen <= k:
                reservoir.append(obj)
            else:
                j = rng.randrange(n_seen)
                if j < k:
                    reservoir[j] = obj
    return reservoir


def sequential_take(
    path: Path,
    k: int,
    human_only: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(out) >= k:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _eligible(obj, human_only=human_only):
                continue
            obj = dict(obj)
            _ensure_model_provider(obj)
            out.append(obj)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sample N papers from arxiv-metadata-pre-llm.jsonl with extended schema"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/arxiv-metadata-pre-llm.jsonl"),
        help="Input JSONL path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/arxiv-metadata-pre-llm-sample-10k.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "-n",
        "--count",
        type=int,
        default=10_000,
        help="Number of papers to sample (default 10000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reservoir sampling (default 42)",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Take the first N eligible rows instead of a random reservoir sample",
    )
    p.add_argument(
        "--include-ai-flagged",
        action="store_true",
        help="Include rows with ai_written != 0 (default: human-only, ai_written==0)",
    )
    args = p.parse_args()

    if not args.input.exists():
        print("Error: input file not found:", args.input, file=sys.stderr)
        return 1

    human_only = not args.include_ai_flagged
    if args.sequential:
        records = sequential_take(args.input, args.count, human_only=human_only)
    else:
        records = reservoir_sample(
            args.input, args.count, seed=args.seed, human_only=human_only
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for i, obj in enumerate(records):
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if (i + 1) % 5000 == 0:
                print("Written %d ..." % (i + 1), flush=True)

    if len(records) < args.count:
        print(
            "Warning: requested %d papers but only wrote %d (not enough eligible rows)."
            % (args.count, len(records)),
            file=sys.stderr,
        )
    print("Done. Wrote %d papers to %s" % (len(records), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
