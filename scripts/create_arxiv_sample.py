#!/usr/bin/env python3
"""
Create a sample of arxiv-metadata-pre-llm.jsonl with a fixed number of papers.

Output has the same structure as the source plus two optional fields for AI-generated text:
  - model: name of the model that generated the text (null for human papers)
  - provider: API/provider used (e.g. openai, anthropic) (null for human papers)
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Sample N papers from arxiv-metadata-pre-llm.jsonl with extended schema")
    p.add_argument("--input", type=Path, default=Path("dataset/arxiv-metadata-pre-llm.jsonl"),
                   help="Input JSONL path")
    p.add_argument("--output", type=Path, default=Path("dataset/arxiv-metadata-pre-llm-sample-10k.jsonl"),
                   help="Output JSONL path")
    p.add_argument("-n", "--count", type=int, default=10_000, help="Number of papers to sample (default 10000)")
    args = p.parse_args()

    if not args.input.exists():
        print("Error: input file not found:", args.input, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if written >= args.count:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Preserve all original keys; add model and provider (null for human papers)
            if "model" not in obj:
                obj["model"] = None
            if "provider" not in obj:
                obj["provider"] = None
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
                print("Written %d ..." % written, flush=True)

    print("Done. Wrote %d papers to %s" % (written, args.output))
    return 0

if __name__ == "__main__":
    sys.exit(main())
