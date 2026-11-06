"""
Tag test-split papers whose authors all work outside English-speaking countries.

The non-native-English test set: a detector must not flag these authors' own writing more than
anyone else's. Countries come from OpenAlex, per author. A paper counts as non-native only when
every author with a known country is outside the English-speaking ones; mixed papers are left out
of both groups. It's a proxy (an author abroad may be a native speaker), and reported as one.

Only test-split papers are looked up: these rows are never trained on. Answers are cached, so a
rerun costs nothing.

    python scripts/tag_non_native.py
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_dataset import TEST, append_jsonl, assign_paper, read_jsonl  # noqa: E402

OPENALEX_WORK = "https://api.openalex.org/works/{key}"
ENGLISH_SPEAKING = {"US", "GB", "IE", "CA", "AU", "NZ"}
REQUEST_DELAY_SECONDS = 0.2


def openalex_key(paper_id: str) -> Optional[str]:
    source, _, identifier = paper_id.partition(":")
    if source == "arxiv":
        return f"doi:10.48550/arXiv.{identifier}"
    if source == "pmc":
        return f"pmcid:PMC{identifier}"
    return None


def author_countries(work: Dict) -> List[List[str]]:
    """Each author's countries, as OpenAlex lists them."""
    return [sorted(set(authorship.get("countries") or [])) for authorship in work.get("authorships") or []]


def classify(countries: Sequence[Sequence[str]]) -> Optional[bool]:
    """True when every author with a known country is outside the English-speaking ones, False when
    every one is inside, None when mixed or unknown."""
    known = [set(c) for c in countries if c]
    if not known:
        return None
    if all(not (c & ENGLISH_SPEAKING) for c in known):
        return True
    if all(c <= ENGLISH_SPEAKING for c in known):
        return False
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="dataset/segments")
    parser.add_argument("--seed", type=int, default=42, help="Must match train_segments.py --seed")
    parser.add_argument("--mailto", help="Optional: OpenAlex's faster pool for requests that give a contact address")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    out = workdir / "author_countries.jsonl"
    done = {row["paper_id"] for row in read_jsonl(out)} if out.exists() else set()
    papers = sorted({row["paper_id"] for row in read_jsonl(workdir / "human_segments.jsonl") if assign_paper(row["paper_id"], args.seed) == TEST})

    session = requests.Session()
    session.headers["User-Agent"] = "sparcly-ai-detection-dataset/1.0"
    for paper_id in papers:
        key = openalex_key(paper_id)
        if paper_id in done or key is None:
            continue
        time.sleep(REQUEST_DELAY_SECONDS)
        params = {"mailto": args.mailto} if args.mailto else None
        try:
            response = session.get(OPENALEX_WORK.format(key=key), params=params, timeout=60)
        except requests.RequestException as e:
            print(f"[countries] {paper_id}: {type(e).__name__}, will retry on the next run")
            continue
        if response.status_code == 404:
            entry = {"paper_id": paper_id, "countries": [], "non_native": None, "found": False}
        elif response.status_code != 200:
            print(f"[countries] {paper_id}: HTTP {response.status_code}, will retry on the next run")
            continue
        else:
            countries = author_countries(response.json())
            entry = {"paper_id": paper_id, "countries": countries, "non_native": classify(countries), "found": True}
        append_jsonl(out, [entry])
        print(f"[countries] {paper_id:28s} {entry['non_native']!s:5s} {entry['countries']}")

    tally = Counter(str(row["non_native"]) for row in read_jsonl(out)) if out.exists() else Counter()
    print(json.dumps({"test_papers": len(papers), "non_native": tally["True"], "english_speaking": tally["False"], "mixed_or_unknown": tally["None"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
