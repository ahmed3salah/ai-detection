#!/usr/bin/env python3
"""
Populate dataset/topics.txt with scientific topics from:
  - Existing topics file (kept as-is)
  - ArXiv metadata (titles from arxiv-metadata-pre-llm JSONL)
  - OpenAlex API (recent work titles; no API key required)
  - Optional: curated list of science topic phrases

Note: Google Scholar is not included (no public API; scraping is against ToS).
OpenAlex and ArXiv provide free, legal sources for diverse research topics.

Use the output with: generate_ai_data.py --topic-file dataset/topics.txt --limit 5000
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# Project root
_root = Path(__file__).resolve().parent.parent

# Curated scientific topic phrases (diverse domains)
CURATED_TOPICS = [
    "Bayesian optimization for hyperparameter tuning",
    "causal inference in observational studies",
    "differential privacy in machine learning",
    "neural symbolic reasoning",
    "continual learning and catastrophic forgetting",
    "self-supervised representation learning",
    "multimodal learning with vision and language",
    "robustness and adversarial examples",
    "transfer learning for low-resource languages",
    "protein structure prediction with deep learning",
    "cryo-EM and single-particle reconstruction",
    "CRISPR and gene editing outcomes",
    "climate modeling and extreme events",
    "dark matter detection and indirect signals",
    "quantum error correction and fault tolerance",
    "topological materials and transport",
    "battery materials and solid-state electrolytes",
    "carbon capture and utilization",
    "epidemiological modeling of infectious disease",
    "microbiome and host health",
]


def _normalize(t: str) -> str:
    """Normalize for deduplication."""
    return re.sub(r"\s+", " ", (t or "").strip()).strip()


def load_existing(path: Path) -> list:
    if not path.exists():
        return []
    return [_normalize(line) for line in path.read_text(encoding="utf-8").splitlines() if _normalize(line)]


def topics_from_arxiv(arxiv_path: Path, max_topics: int) -> list:
    """Extract paper titles as topics from arxiv JSONL."""
    topics = []
    seen = set()
    with arxiv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(topics) >= max_topics:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (obj.get("title") or "").replace("\n", " ").strip()
            title = _normalize(title)
            if not title or len(title) < 15:
                continue
            key = title.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            topics.append(title)
    return topics


def topics_from_openalex(per_page: int = 200, max_pages: int = 5) -> list:
    """Fetch recent work titles from OpenAlex (no API key required)."""
    topics = []
    seen = set()
    for page in range(max_pages):
        url = (
            "https://api.openalex.org/works?"
            "filter=publication_year:2023|2024,type:article&"
            "select=display_name&"
            "per-page=%d&page=%d&sort=cited_by_count:desc"
        ) % (per_page, page + 1)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-detection-topics/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("results") or []:
                title = _normalize(item.get("display_name") or "")
                if not title or len(title) < 15:
                    continue
                key = title.lower()[:80]
                if key in seen:
                    continue
                seen.add(key)
                topics.append(title)
        except Exception as e:
            print("  OpenAlex page %d: %s" % (page + 1, e), file=sys.stderr)
            break
    return topics


def main():
    p = argparse.ArgumentParser(
        description="Fetch topics from arxiv + OpenAlex and write dataset/topics.txt for generate_ai_data.py",
    )
    p.add_argument("--output", type=Path, default=_root / "dataset/topics.txt", help="Output topics file")
    p.add_argument("--existing", type=Path, default=None, help="Existing topics file to merge (default: output path)")
    p.add_argument("--arxiv", type=Path, default=_root / "dataset/arxiv-metadata-pre-llm-sample-15k.jsonl",
                   help="Arxiv JSONL to extract titles from")
    p.add_argument("--arxiv-max", type=int, default=2000, help="Max topics to take from arxiv (default 2000)")
    p.add_argument("--openalex", action="store_true", default=True, help="Fetch topics from OpenAlex (default on)")
    p.add_argument("--no-openalex", action="store_false", dest="openalex", help="Skip OpenAlex")
    p.add_argument("--openalex-pages", type=int, default=5, help="OpenAlex pages (200 titles per page)")
    p.add_argument("--curated", action="store_true", default=True, help="Add curated science topics (default on)")
    p.add_argument("--no-curated", action="store_false", dest="curated")
    args = p.parse_args()

    existing_path = args.existing or args.output
    combined = []
    seen_lower = set()

    # 1. Existing file
    existing = load_existing(existing_path)
    for t in existing:
        k = t.lower()
        if k not in seen_lower:
            seen_lower.add(k)
            combined.append(t)
    print("Existing topics: %d" % len(existing))

    # 2. Curated
    if args.curated:
        for t in CURATED_TOPICS:
            t = _normalize(t)
            k = t.lower()
            if k not in seen_lower:
                seen_lower.add(k)
                combined.append(t)
        print("Curated added: %d" % len(CURATED_TOPICS))

    # 3. ArXiv
    if args.arxiv.exists():
        arxiv_topics = topics_from_arxiv(args.arxiv, args.arxiv_max)
        added = 0
        for t in arxiv_topics:
            k = t.lower()
            if k not in seen_lower:
                seen_lower.add(k)
                combined.append(t)
                added += 1
        print("ArXiv titles as topics: %d new (from %s)" % (added, args.arxiv.name))
    else:
        print("ArXiv file not found: %s (skipping)" % args.arxiv)

    # 4. OpenAlex
    if args.openalex:
        openalex_topics = topics_from_openalex(per_page=200, max_pages=args.openalex_pages)
        added = 0
        for t in openalex_topics:
            k = t.lower()
            if k not in seen_lower:
                seen_lower.add(k)
                combined.append(t)
                added += 1
        print("OpenAlex titles as topics: %d new" % added)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(combined) + "\n" if combined else "", encoding="utf-8")
    print("Wrote %d total topics to %s" % (len(combined), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
