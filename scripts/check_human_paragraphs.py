"""
What build_human_paragraphs.py has collected so far, for a person to read.

    python scripts/check_human_paragraphs.py
"""
import argparse
import collections
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_dataset import read_jsonl  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="dataset/segments")
    args = parser.parse_args(argv)

    # A Windows console may not have every character; show those as ? instead of crashing.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    workdir = Path(args.workdir)
    print("--- papers")
    papers = read_jsonl(workdir / "papers.jsonl")
    for entry in papers:
        print(f"{entry['paper_id']:28s} {entry['quota']:8s} {entry['status']:20s} {entry['paragraphs']:4d}  {entry['error'][:80]}")
    print(collections.Counter(entry["status"] for entry in papers).most_common())

    rows = read_jsonl(workdir / "human_segments.jsonl") if (workdir / "human_segments.jsonl").exists() else []
    print(f"--- {len(rows)} paragraphs from {len({row['paper_id'] for row in rows})} papers")
    print(collections.Counter(row["section"] for row in rows).most_common())
    print(collections.Counter(row["domain"] for row in rows).most_common())

    print("--- samples from arXiv (read through GROBID)")
    arxiv = [row for row in rows if row["paper_id"].startswith("arxiv")]
    for row in arxiv[:: max(1, len(arxiv) // 6)][:6]:
        print(f"[{row['paper_id']} | {row['heading']} | {len(row['text'].split())} words]")
        print(row["text"][:320])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
