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
from segment_dataset import TEST, append_jsonl, assign_paper, read_jsonl  # noqa: E402

PROMPT_VERSION = "v2"

# The sentence ends generate_ai_paragraphs.rejection_reason accepts.
SENTENCE_ENDS = (".", "!", "?", '"', ")")

KEEP = ("id", "text", "label", "paper_id", "domain", "section", "generator", "test_set", "mode")


def trimmed(row: Dict) -> Dict:
    return {key: row[key] for key in KEEP if row.get(key) is not None}


def is_fragment(text: str) -> bool:
    """
    A paragraph GROBID cut at a page or column break: it starts mid-sentence or stops mid-sentence.
    Generated paragraphs never do (replies that don't end a sentence are rejected), so keeping these
    would teach a detector that a broken sentence means human.
    """
    text = text.strip()
    return not text or text[0].islower() or not text.endswith(SENTENCE_ENDS)


def merge(
    human: Sequence[Dict],
    ai: Sequence[Dict],
    prompt_version: Optional[str] = None,
    seed: int = 42,
    non_native_papers: Optional[set] = None,
) -> List[Dict]:
    """
    Rows from one prompt version only: a dataset mixing versions would measure neither. Polished rows
    don't depend on the generation prompt and are always kept, one per source paragraph. A pair whose
    human paragraph is a fragment is left out on both sides.

    A test paper's other human paragraphs, the ones no AI paragraph replaced, are added as test-only
    rows: "non_native" when its authors all work outside English-speaking countries, "e2e" otherwise.
    With the paired ones they make up the whole paper as GROBID read it from the real PDF, which is
    what the document-level gate scores.
    """
    by_id = {row["id"]: row for row in human}
    rows: List[Dict] = []
    used = set()
    polished_sources = set()
    for row in ai:
        source = by_id[row["source_segment_id"]]
        if row.get("test_set"):
            # Only from test papers: a test-only row keeps its whole paper out of training. Polished
            # rows from earlier runs were drawn from any paper.
            if row["source_segment_id"] in polished_sources or assign_paper(row["paper_id"], seed) != TEST:
                continue
            polished_sources.add(row["source_segment_id"])
        elif prompt_version and row.get("prompt_version", "v1") != prompt_version:
            continue
        elif is_fragment(source["text"]):
            continue
        rows.append(trimmed(row))
        source = row["source_segment_id"]
        # A polished row stands in for its source: keeping both in one test set adds nothing.
        if not row.get("test_set") and source not in used:
            used.add(source)
            rows.append(trimmed(by_id[source]))

    non_native_papers = non_native_papers or set()
    for row in human:
        if row["id"] in used or row["id"] in polished_sources or assign_paper(row["paper_id"], seed) != TEST:
            continue
        rows.append({**trimmed(row), "test_set": "non_native" if row["paper_id"] in non_native_papers else "e2e"})
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
        "e2e_documents": len({row["paper_id"] for row in rows if row.get("test_set") == "e2e"}),
        "non_native_paragraphs": sum(row.get("test_set") == "non_native" for row in rows),
        "non_native_documents": len({row["paper_id"] for row in rows if row.get("test_set") == "non_native"}),
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
    parser.add_argument("--seed", type=int, default=42, help="Must match train_segments.py --seed")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    countries = workdir / "author_countries.jsonl"
    non_native = {row["paper_id"] for row in read_jsonl(countries) if row["non_native"]} if countries.exists() else set()
    if not countries.exists():
        print("No author_countries.jsonl: run scripts/tag_non_native.py for the non-native set.")
    rows = merge(read_jsonl(workdir / "human_segments.jsonl"), read_jsonl(workdir / "ai_segments.jsonl"), args.prompt_version, args.seed, non_native)
    out = Path(args.out or workdir / "segments.jsonl")
    out.unlink(missing_ok=True)
    append_jsonl(out, rows)
    print(json.dumps(describe(rows), indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
