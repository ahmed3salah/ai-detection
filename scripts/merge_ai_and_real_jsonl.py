#!/usr/bin/env python3
"""
Merge AI-written arxiv-style JSONL files with a uniform random sample
of real arxiv-metadata-pre-llm.jsonl records into a single JSONL file.

Usage (defaults are tailored to this repo):

  python -m scripts.merge_ai_and_real_jsonl \
    --output dataset/ai_plus_real_70k.jsonl

This will:
- Read AI records from:
    - dataset/ai_abstracts_20k.jsonl
    - dataset/ai_from_topics_5k.jsonl
    - dataset/ai_generated_test_10k.jsonl
- Build a normalized title set from all AI records.
- Stream over dataset/arxiv-metadata-pre-llm.jsonl and reservoir-sample
  `--real-sample-size` records (default 35000) whose titles do not
  collide with the AI title set.
- Map real arxiv records into the same schema as the AI JSONL records,
  with ai_written=0 and simple model/provider tags.
- Optionally shuffle the merged records before writing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# Ensure project root and scripts dir are importable so ai_jsonl_quality
# resolves correctly when running as a module (python -m scripts.merge_ai_and_real_jsonl).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jsonl_quality import (  # type: ignore[import]
    is_usable_abstract,
    validate_record_dict,
)


DATASET_DIR = PROJECT_ROOT / "dataset"

DEFAULT_AI_FILES = [
    DATASET_DIR / "ai_abstracts_20k.jsonl",
    DATASET_DIR / "ai_from_topics_5k.jsonl",
    DATASET_DIR / "ai_generated_test_10k.jsonl",
]

DEFAULT_ARXIV_FILE = DATASET_DIR / "arxiv-metadata-pre-llm.jsonl"


def _norm_title(title: str) -> str:
    return (title or "").strip().lower()


def _load_ai_records(ai_paths: Iterable[Path]) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load AI records from the given JSONL files and build a normalized title set."""
    all_records: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for path in ai_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Basic validation – non-fatal
                ok, _reasons = validate_record_dict(rec, require_categories=False)
                if not ok:
                    # Keep it anyway; this is the ground-truth AI set.
                    pass
                title = _norm_title(str(rec.get("title") or ""))
                if title:
                    seen_titles.add(title)
                all_records.append(rec)
    return all_records, seen_titles


def _map_arxiv_record_to_schema(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map a raw arxiv-metadata-pre-llm record into the common schema:
    id, title, abstract, categories, authors_parsed, update_date, ai_written, model, provider.
    """
    title = str(obj.get("title") or "").strip()
    abstract = str(obj.get("abstract") or "").strip().replace("\n", " ")
    if not title or not abstract:
        return None
    if not is_usable_abstract(abstract):
        return None

    # arxiv-metadata-pre-llm typically has: id, categories, authors_parsed, update_date, etc.
    record: Dict[str, Any] = {
        "id": str(obj.get("id") or "").strip(),
        "title": title,
        "abstract": abstract,
        "categories": str(obj.get("categories") or "").strip(),
        "authors_parsed": obj.get("authors_parsed") or [],
        "update_date": str(obj.get("update_date") or "").strip(),
        "ai_written": 0,
        "model": "arxiv",
        "provider": "arxiv",
    }
    # Final sanity check using shared validator; do not require categories for real data.
    ok, _reasons = validate_record_dict(record, require_categories=False)
    if not ok:
        return None
    return record


def _reservoir_sample_arxiv(
    path: Path,
    k: int,
    forbidden_titles: Set[str],
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Reservoir sample k records from a large arxiv JSONL file, skipping any whose
    (normalized) title appears in forbidden_titles. Records are mapped into the
    unified schema before being considered.
    """
    if seed is not None:
        random.seed(seed)

    reservoir: List[Dict[str, Any]] = []
    seen_valid = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            mapped = _map_arxiv_record_to_schema(obj)
            if not mapped:
                continue
            title_norm = _norm_title(mapped.get("title", ""))
            if title_norm and title_norm in forbidden_titles:
                continue

            if seen_valid < k:
                reservoir.append(mapped)
            else:
                j = random.randint(0, seen_valid)
                if j < k:
                    reservoir[j] = mapped
            seen_valid += 1

    if len(reservoir) < k:
        # Not enough valid, deduplicated samples; proceed with what we have.
        print(
            f"Warning: requested {k} real samples but only obtained {len(reservoir)} "
            f"after filtering/deduplication."
        )
    return reservoir


def merge_ai_and_real(
    ai_files: List[Path],
    arxiv_file: Path,
    real_sample_size: int,
    output: Path,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[int, int]:
    """Core merge routine. Returns (ai_count, real_count)."""
    ai_records, ai_title_set = _load_ai_records(ai_files)
    print(f"Loaded {len(ai_records)} AI records from {len(ai_files)} file(s).")

    real_records = _reservoir_sample_arxiv(
        arxiv_file,
        k=real_sample_size,
        forbidden_titles=ai_title_set,
        seed=seed,
    )
    print(f"Sampled {len(real_records)} real arxiv records.")

    combined: List[Dict[str, Any]] = []
    combined.extend(ai_records)
    combined.extend(real_records)

    if shuffle:
        random.seed(seed)
        random.shuffle(combined)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(ai_records), len(real_records)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge AI-written JSONL files with a sampled set of real arxiv metadata.",
    )
    p.add_argument(
        "--ai-files",
        nargs="*",
        type=Path,
        default=[str(p) for p in DEFAULT_AI_FILES],
        help="AI JSONL files to merge (default: three main AI files in dataset/).",
    )
    p.add_argument(
        "--arxiv-file",
        type=Path,
        default=DEFAULT_ARXIV_FILE,
        help="Path to arxiv-metadata-pre-llm.jsonl (default: dataset/arxiv-metadata-pre-llm.jsonl).",
    )
    p.add_argument(
        "--real-sample-size",
        type=int,
        default=35000,
        help="Number of real arxiv records to sample (default: 35000).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "ai_plus_real_70k.jsonl",
        help="Output JSONL path (default: dataset/ai_plus_real_70k.jsonl).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling (default: 42).",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle the merged records before writing.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    ai_files = [Path(p) for p in args.ai_files]

    missing_ai = [str(p) for p in ai_files if not p.exists()]
    if missing_ai:
        print("Warning: missing AI files (will be skipped):", ", ".join(missing_ai))

    if not args.arxiv_file.exists():
        print(f"Error: arxiv file not found: {args.arxiv_file}")
        return 1

    shuffle = not args.no_shuffle

    ai_count, real_count = merge_ai_and_real(
        ai_files=ai_files,
        arxiv_file=args.arxiv_file,
        real_sample_size=args.real_sample_size,
        output=args.output,
        seed=args.seed,
        shuffle=shuffle,
    )

    total = ai_count + real_count
    print(
        f"Wrote merged dataset to {args.output} "
        f"(ai_written=1: {ai_count}, ai_written=0: {real_count}, total: {total})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

