#!/usr/bin/env python3
"""
Fetch human-written scientific text for AI-detection training (Option B).

- arXiv: abstracts via public API. Optionally restrict to pre-2021 to reduce
  chance of LLM-assisted writing. When fetching arXiv, you choose category/categories
  from a list and optionally run for a fixed time (e.g. 60 minutes).
- PubMed: abstracts via NCBI eutils (biomedical).

Output: appends to dataset/human_abstracts.txt (one abstract per line) or
dataset/human_abstracts.jsonl with {"text": "...", "label": 0, "source": "arxiv", "id": "..."}.

Improvements:
- Always appends; never overwrites or deletes existing data.
- Deduplicates by content hash; safe to run multiple script instances (file lock).
- Multi-threaded fetching for higher throughput.
- arXiv: no default category; script shows full category list to choose from.
- arXiv: optional time-limited scraping (e.g. run for 60 minutes then stop).
- arXiv API compliance: 3s delay between requests, single connection (no parallel arXiv), 30k max per query.
- Can run until --target size is reached (loop over pagination and sources).
"""

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows: no file locking (avoid concurrent runs on same file)

# arXiv API usage limits (https://info.arxiv.org/help/api/user-manual.html)
ARXIV_DELAY_SEC = 3          # Required 3-second delay between consecutive requests
ARXIV_MAX_RESULTS_PER_QUERY = 30000  # API maximum; large requests often cause 500 errors
ARXIV_BATCH_SIZE = 500       # Safe default per request (use 30k only if server handles it)

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Full arXiv category list (id, short name) for interactive choice. No default.
# Source: https://arxiv.org/category_taxonomy
ARXIV_CATEGORY_LIST = [
    ("cs.AI", "Artificial Intelligence"),
    ("cs.AR", "Hardware Architecture"),
    ("cs.CC", "Computational Complexity"),
    ("cs.CE", "Computational Engineering, Finance, and Science"),
    ("cs.CG", "Computational Geometry"),
    ("cs.CL", "Computation and Language (NLP)"),
    ("cs.CR", "Cryptography and Security"),
    ("cs.CV", "Computer Vision and Pattern Recognition"),
    ("cs.CY", "Computers and Society"),
    ("cs.DB", "Databases"),
    ("cs.DC", "Distributed, Parallel, and Cluster Computing"),
    ("cs.DL", "Digital Libraries"),
    ("cs.DM", "Discrete Mathematics"),
    ("cs.DS", "Data Structures and Algorithms"),
    ("cs.ET", "Emerging Technologies"),
    ("cs.FL", "Formal Languages and Automata Theory"),
    ("cs.GL", "General Literature"),
    ("cs.GR", "Graphics"),
    ("cs.GT", "Computer Science and Game Theory"),
    ("cs.HC", "Human-Computer Interaction"),
    ("cs.IR", "Information Retrieval"),
    ("cs.IT", "Information Theory"),
    ("cs.LG", "Machine Learning"),
    ("cs.LO", "Logic in Computer Science"),
    ("cs.MA", "Multiagent Systems"),
    ("cs.MM", "Multimedia"),
    ("cs.MS", "Mathematical Software"),
    ("cs.NA", "Numerical Analysis"),
    ("cs.NE", "Neural and Evolutionary Computing"),
    ("cs.NI", "Networking and Internet Architecture"),
    ("cs.OH", "Other Computer Science"),
    ("cs.OS", "Operating Systems"),
    ("cs.PF", "Performance"),
    ("cs.PL", "Programming Languages"),
    ("cs.RO", "Robotics"),
    ("cs.SC", "Symbolic Computation"),
    ("cs.SD", "Sound"),
    ("cs.SE", "Software Engineering"),
    ("cs.SI", "Social and Information Networks"),
    ("cs.SY", "Systems and Control"),
    ("econ.EM", "Econometrics"),
    ("econ.GN", "General Economics"),
    ("econ.TH", "Theoretical Economics"),
    ("eess.AS", "Audio and Speech Processing"),
    ("eess.IV", "Image and Video Processing"),
    ("eess.SP", "Signal Processing"),
    ("eess.SY", "Systems and Control (EESS)"),
    ("math.AC", "Commutative Algebra"),
    ("math.AG", "Algebraic Geometry"),
    ("math.AP", "Analysis of PDEs"),
    ("math.AT", "Algebraic Topology"),
    ("math.CA", "Classical Analysis and ODEs"),
    ("math.CO", "Combinatorics"),
    ("math.CT", "Category Theory"),
    ("math.CV", "Complex Variables"),
    ("math.DG", "Differential Geometry"),
    ("math.DS", "Dynamical Systems"),
    ("math.FA", "Functional Analysis"),
    ("math.GM", "General Mathematics"),
    ("math.GN", "General Topology"),
    ("math.GR", "Group Theory"),
    ("math.GT", "Geometric Topology"),
    ("math.HO", "History and Overview"),
    ("math.IT", "Information Theory (math)"),
    ("math.LO", "Logic"),
    ("math.MG", "Metric Geometry"),
    ("math.NA", "Numerical Analysis"),
    ("math.NT", "Number Theory"),
    ("math.OC", "Optimization and Control"),
    ("math.PR", "Probability"),
    ("math.QA", "Quantum Algebra"),
    ("math.RA", "Rings and Algebras"),
    ("math.RT", "Representation Theory"),
    ("math.SG", "Symplectic Geometry"),
    ("math.SP", "Spectral Theory"),
    ("math.ST", "Statistics Theory"),
    ("stat.AP", "Statistics Applications"),
    ("stat.CO", "Statistics Computation"),
    ("stat.ME", "Statistics Methodology"),
    ("stat.ML", "Machine Learning (stat)"),
    ("stat.OT", "Other Statistics"),
    ("stat.TH", "Statistics Theory"),
    ("astro-ph.CO", "Cosmology and Nongalactic Astrophysics"),
    ("astro-ph.EP", "Earth and Planetary Astrophysics"),
    ("astro-ph.GA", "Astrophysics of Galaxies"),
    ("astro-ph.HE", "High Energy Astrophysical Phenomena"),
    ("astro-ph.IM", "Instrumentation and Methods for Astrophysics"),
    ("astro-ph.SR", "Solar and Stellar Astrophysics"),
    ("cond-mat.dis-nn", "Disordered Systems and Neural Networks"),
    ("cond-mat.mes-hall", "Mesoscale and Nanoscale Physics"),
    ("cond-mat.mtrl-sci", "Materials Science"),
    ("cond-mat.other", "Other Condensed Matter"),
    ("cond-mat.quant-gas", "Quantum Gases"),
    ("cond-mat.soft", "Soft Condensed Matter"),
    ("cond-mat.stat-mech", "Statistical Mechanics"),
    ("cond-mat.str-el", "Strongly Correlated Electrons"),
    ("cond-mat.supr-con", "Superconductivity"),
    ("gr-qc", "General Relativity and Quantum Cosmology"),
    ("hep-ex", "High Energy Physics - Experiment"),
    ("hep-lat", "High Energy Physics - Lattice"),
    ("hep-ph", "High Energy Physics - Phenomenology"),
    ("hep-th", "High Energy Physics - Theory"),
    ("math-ph", "Mathematical Physics"),
    ("nlin.AO", "Adaptation and Self-Organizing Systems"),
    ("nlin.CD", "Chaotic Dynamics"),
    ("nlin.CG", "Cellular Automata and Lattice Gases"),
    ("nlin.PS", "Pattern Formation and Solitons"),
    ("nlin.SI", "Exactly Solvable and Integrable Systems"),
    ("nucl-ex", "Nuclear Experiment"),
    ("nucl-th", "Nuclear Theory"),
    ("physics.acc-ph", "Accelerator Physics"),
    ("physics.ao-ph", "Atmospheric and Oceanic Physics"),
    ("physics.app-ph", "Applied Physics"),
    ("physics.atom-ph", "Atomic Physics"),
    ("physics.bio-ph", "Biological Physics"),
    ("physics.chem-ph", "Chemical Physics"),
    ("physics.class-ph", "Classical Physics"),
    ("physics.comp-ph", "Computational Physics"),
    ("physics.data-an", "Data Analysis, Statistics and Probability"),
    ("physics.ed-ph", "Physics Education"),
    ("physics.flu-dyn", "Fluid Dynamics"),
    ("physics.geo-ph", "Geophysics"),
    ("physics.ins-det", "Instrumentation and Detectors"),
    ("physics.med-ph", "Medical Physics"),
    ("physics.optics", "Optics"),
    ("physics.plasm-ph", "Plasma Physics"),
    ("physics.space-ph", "Space Physics"),
    ("quant-ph", "Quantum Physics"),
    ("q-bio.BM", "Biomolecules"),
    ("q-bio.CB", "Cell Behavior"),
    ("q-bio.GN", "Genomics"),
    ("q-bio.MN", "Molecular Networks"),
    ("q-bio.NC", "Neurons and Cognition"),
    ("q-bio.PE", "Populations and Evolution"),
    ("q-bio.SC", "Subcellular Processes"),
    ("q-bio.TO", "Tissues and Organs"),
    ("q-fin.CP", "Computational Finance"),
    ("q-fin.GN", "General Finance"),
    ("q-fin.MF", "Mathematical Finance"),
    ("q-fin.PM", "Portfolio Management"),
    ("q-fin.PR", "Pricing of Securities"),
    ("q-fin.RM", "Risk Management"),
    ("q-fin.ST", "Statistical Finance"),
    ("q-fin.TR", "Trading and Market Microstructure"),
]

PUBMED_QUERIES = [
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "clinical trial",
    "randomized controlled trial",
]


def normalize_text(text: str) -> str:
    """Normalize abstract for deduplication: strip, collapse whitespace, lower case."""
    if not text or not isinstance(text, str):
        return ""
    t = " ".join(re.split(r"\s+", text.strip()))
    return t.lower()


def content_hash(text: str) -> str:
    """Stable hash of normalized content for deduplication."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _lock_path(out_path: Path) -> Path:
    """Path for the lock file (allows multiple processes to coordinate on same output file)."""
    return out_path.parent / (out_path.name + ".lock")


class _OutputLock:
    """File lock so multiple script runs can safely read/append the same output file."""

    def __init__(self, out_path: Path):
        self._out_path = out_path
        self._lock_path = _lock_path(out_path)
        self._lock_file = None

    def __enter__(self):
        self._lock_path.touch(exist_ok=True)
        self._lock_file = self._lock_path.open("a")
        if fcntl is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if fcntl is not None and self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def load_hashes(self, is_jsonl: bool) -> set[str]:
        """Load content hashes from output file (call while holding lock)."""
        seen = set()
        if not self._out_path.exists():
            return seen
        with self._out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if is_jsonl:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text") or ""
                    except json.JSONDecodeError:
                        continue
                else:
                    text = line
                seen.add(content_hash(text))
        return seen

    def append_new(self, is_jsonl: bool, seen_hashes: set[str], items: list[tuple[str, str, str]]) -> int:
        """
        Append only items whose content hash is not in seen_hashes.
        Updates seen_hashes in place. Returns number of new lines written.
        Call while holding lock; re-reads file to get latest hashes (for concurrent runs).
        """
        current = self.load_hashes(is_jsonl)
        written = 0
        with self._out_path.open("a", encoding="utf-8") as f:
            for text, source, sid in items:
                h = content_hash(text)
                if h in current:
                    continue
                current.add(h)
                seen_hashes.add(h)
                if is_jsonl:
                    f.write(json.dumps({"text": text, "label": 0, "source": source, "id": sid}, ensure_ascii=False) + "\n")
                else:
                    f.write(text.replace("\n", " ") + "\n")
                written += 1
        return written


def load_existing_hashes(out_path: Path, is_jsonl: bool) -> set[str]:
    """Load existing output file and return set of content hashes (no lock; use _OutputLock for concurrent safe load)."""
    seen = set()
    if not out_path.exists():
        return seen
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or ""
                except json.JSONDecodeError:
                    continue
            else:
                text = line
            seen.add(content_hash(text))
    return seen


def prompt_arxiv_categories() -> list[str]:
    """Show full arXiv category list and prompt user to choose. Returns list of category IDs."""
    valid_ids = {t[0] for t in ARXIV_CATEGORY_LIST}
    print("\n--- arXiv categories (choose one or more) ---")
    # Print in columns: id - name
    for i, (cid, name) in enumerate(ARXIV_CATEGORY_LIST):
        print("  %-22s %s" % (cid, name))
    print()
    while True:
        try:
            raw = input("Enter category ID(s), comma-separated (or 'all' for all): ").strip()
        except EOFError:
            print("No input (non-interactive). Use --arxiv-category=cs.AI (or another id) on the command line.")
            sys.exit(1)
        if not raw:
            print("Please enter at least one category ID or 'all'.")
            continue
        if raw.lower() == "all":
            return [t[0] for t in ARXIV_CATEGORY_LIST]
        chosen = [s.strip() for s in raw.split(",") if s.strip()]
        invalid = [c for c in chosen if c not in valid_ids]
        if invalid:
            print("Unknown category ID(s): %s. Use IDs from the list above." % invalid)
            continue
        return chosen


def prompt_arxiv_duration_minutes() -> float:
    """Ask how many minutes to scrape arXiv. 0 or empty = no time limit."""
    print()
    while True:
        try:
            raw = input("Scrape arXiv for how many minutes? (0 or empty = no limit): ").strip()
        except EOFError:
            return 0.0
        if not raw:
            return 0.0
        try:
            v = float(raw)
            if v < 0:
                print("Enter a non-negative number.")
                continue
            return v
        except ValueError:
            print("Enter a number (e.g. 60 for one hour).")


def fetch_arxiv(max_results: int = 500, start: int = 0, before_year: int = 2021, category: str = "cs.AI"):
    """Fetch arXiv abstracts via public API. Returns list of (abstract_text, arxiv_id).
    Retries on HTTP 429 (Too Many Requests) with exponential backoff."""
    try:
        import urllib.request
        import urllib.parse
        import urllib.error
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("Use Python standard library only (urllib, xml.etree).")

    url = "http://export.arxiv.org/api/query?"
    query = "cat:%s" % category
    sort_order = "ascending" if before_year else "descending"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": sort_order,
    }
    req = urllib.request.Request(url + urllib.parse.urlencode(params))
    req.add_header("User-Agent", "ai-detection-training/1.0")

    last_error = None
    for attempt in range(5):  # max 5 attempts
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                root = ET.fromstring(resp.read())
            break
        except urllib.error.HTTPError as e:
            last_error = e
            # 429 = rate limit; 5xx = server error (often caused by very large requests)
            if e.code == 429:
                wait = min(60, (2 ** attempt) * 5)
                if e.headers.get("Retry-After"):
                    try:
                        wait = min(120, int(e.headers["Retry-After"]))
                    except (ValueError, TypeError):
                        pass
                if attempt < 4:
                    print("  arXiv rate limited (429), retrying in %ds..." % wait, flush=True)
                    time.sleep(wait)
                    continue
            elif e.code in (500, 502, 503):  # Internal Server Error, Bad Gateway, Service Unavailable
                wait = min(60, (2 ** attempt) * 5)
                if attempt < 4:
                    print("  arXiv server error (%d), retrying in %ds..." % (e.code, wait), flush=True)
                    time.sleep(wait)
                    continue
            raise
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected state in fetch_arxiv retry loop")

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("atom:entry", ns):
        abstract_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)
        if abstract_el is not None and abstract_el.text:
            text = abstract_el.text.strip().replace("\n", " ")
            arxiv_id = (id_el.text or "").strip().split("/")[-1]
            year = None
            if published_el is not None and published_el.text:
                try:
                    year = int(published_el.text[:4])
                except ValueError:
                    pass
            if before_year and year is not None and year >= before_year:
                continue
            out.append((text, arxiv_id))
    return out


def fetch_pubmed(max_results: int = 500, start: int = 0, before_year: int = 2021, query: str = "machine learning"):
    """Fetch PubMed abstracts via NCBI eutils. Returns list of (abstract_text, pmid)."""
    try:
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("Use Python standard library only.")

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # Search (esearch supports JSON)
    search_params = {
        "db": "pubmed",
        "term": query,
        "retstart": start,
        "retmax": max_results,
        "sort": "date",
        "retmode": "json",
    }
    req = urllib.request.Request(
        base + "/esearch.fcgi?" + urllib.parse.urlencode(search_params),
        headers={"User-Agent": "ai-detection-training/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    # Fetch full records: efetch does NOT support retmode=json for PubMed, use XML
    fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
    req = urllib.request.Request(
        base + "/efetch.fcgi?" + urllib.parse.urlencode(fetch_params),
        headers={"User-Agent": "ai-detection-training/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        root = ET.fromstring(resp.read())

    # PubMed XML: PubmedArticleSet > PubmedArticle > MedlineCitation > Article > Abstract > AbstractText
    out = []
    for article in root.findall(".//PubmedArticle"):
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            pmid_el = medline.find("PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
            article_el = medline.find("Article")
            if article_el is None:
                continue
            abstract_el = article_el.find("Abstract")
            if abstract_el is None:
                continue
            parts = []
            for ab in abstract_el.findall("AbstractText"):
                if ab.text:
                    parts.append(ab.text.strip())
                elif list(ab):
                    parts.append("".join(ab.itertext()).strip())
            abstract = " ".join(parts).replace("\n", " ")
            if not abstract:
                continue
            year = None
            pub_date = article_el.find("ArticleDate") or article_el.find("Journal/JournalIssue/PubDate")
            if pub_date is not None and pub_date.find("Year") is not None:
                try:
                    year = int(pub_date.find("Year").text)
                except (ValueError, TypeError, AttributeError):
                    pass
            if before_year and year is not None and year >= before_year:
                continue
            out.append((abstract, str(pmid or "")))
        except (AttributeError, TypeError):
            continue
    return out


def _fetch_arxiv_task(category: str, start: int, before_year: int, batch_size: int) -> tuple[str, int, list[tuple[str, str]]]:
    """One arXiv fetch (used only for sequential, single-connection fetching). Returns (category, start, [(text, arxiv_id), ...])."""
    batch = fetch_arxiv(max_results=batch_size, start=start, before_year=before_year, category=category)
    return (category, start, [(t, aid) for t, aid in batch])


def _fetch_pubmed_task(query: str, start: int, before_year: int, batch_size: int) -> tuple[str, int, list[tuple[str, str]]]:
    """One PubMed fetch task for the thread pool. Returns (query, start, [(text, pmid), ...])."""
    batch = fetch_pubmed(max_results=batch_size, start=start, before_year=before_year, query=query)
    return (query, start, [(t, pid) for t, pid in batch])


def main():
    p = argparse.ArgumentParser(description="Fetch human scientific abstracts (arXiv and/or PubMed)")
    p.add_argument("--source", choices=("arxiv", "pubmed", "both"), default="both",
                   help="Source: arxiv, pubmed, or both (default: both)")
    p.add_argument("--max", type=int, default=1000,
                   help="Max abstracts per source/category per run (default 1000)")
    p.add_argument("--workers", type=int, default=6,
                   help="Number of parallel fetch threads (default 6)")
    p.add_argument("--target", type=int, default=0,
                   help="Target total lines in output file. If > 0, keep fetching until reached or exhausted (default 0 = single run)")
    p.add_argument("--before-year", type=int, default=0,
                   help="Only include papers before this year (0 = no filter, use 2021 for pre-LLM)")
    p.add_argument("--arxiv-category", default=None,
                   help="arXiv category ID (e.g. cs.AI). If not set, script shows full category list to choose from.")
    p.add_argument("--arxiv-minutes", type=float, default=None,
                   help="Scrape arXiv for this many minutes then stop (0 = no limit). If not set, script prompts interactively.")
    p.add_argument("--arxiv-batch-size", type=int, default=None,
                   help="Max results per arXiv request (default 500). Use 30000 for API max but may cause 500 errors.")
    p.add_argument("--pubmed-query", default=None,
                   help="Single PubMed query (default: use all when targeting, else 'machine learning')")
    p.add_argument("--output", default="dataset/human_abstracts.txt",
                   help="Output file (default: dataset/human_abstracts.txt). Use .jsonl for JSONL.")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_jsonl = out_path.suffix.lower() == ".jsonl"

    # Always append; never overwrite. Use file lock so multiple runs can share the same file.
    with _OutputLock(out_path) as lock:
        seen_hashes = lock.load_hashes(is_jsonl)
    existing_lines = len(seen_hashes)
    if existing_lines:
        print("Loaded %d existing abstracts from %s (will skip duplicates)." % (existing_lines, out_path))
    if fcntl is None:
        print("Note: file locking not available on this platform; avoid running multiple instances writing to the same file.")

    # Resolve arXiv categories: required when source is arxiv/both; no default, prompt if missing
    if args.source in ("arxiv", "both"):
        if args.arxiv_category:
            arxiv_cats = [s.strip() for s in args.arxiv_category.split(",") if s.strip()]
        else:
            arxiv_cats = prompt_arxiv_categories()
        if args.arxiv_minutes is not None:
            scrape_minutes = max(0.0, float(args.arxiv_minutes))
        else:
            scrape_minutes = prompt_arxiv_duration_minutes()
    else:
        arxiv_cats = []
        scrape_minutes = 0.0

    pubmed_queries = [args.pubmed_query] if args.pubmed_query else PUBMED_QUERIES

    arxiv_batch_size = args.arxiv_batch_size if args.arxiv_batch_size is not None else ARXIV_BATCH_SIZE
    arxiv_batch_size = min(arxiv_batch_size, ARXIV_MAX_RESULTS_PER_QUERY)
    batch_size_pubmed = 100
    max_per_source = 10000 if args.target > 0 else args.max
    # arXiv: use API max 30k per query; allow pagination beyond general max_per_source
    arxiv_max_per_source = max(max_per_source, ARXIV_MAX_RESULTS_PER_QUERY)
    total_new = 0
    pass_count = 0
    arxiv_start_time = time.monotonic()

    next_start_arxiv = {c: 0 for c in arxiv_cats} if args.source in ("arxiv", "both") else {}
    next_start_pubmed = {q: 0 for q in pubmed_queries} if args.source in ("pubmed", "both") else {}
    active_arxiv = list(next_start_arxiv.keys())
    active_pubmed = list(next_start_pubmed.keys())

    while True:
        pass_count += 1
        if args.target > 0:
            current_total = existing_lines + total_new
            if current_total >= args.target:
                print("Target reached: %d >= %d" % (current_total, args.target))
                break
            print("\n--- Pass %d (current total: %d, target: %d) ---" % (pass_count, current_total, args.target))

        added_this_pass = 0

        # arXiv: single connection, 3-second delay between requests, up to 30k results per query (API compliance)
        if scrape_minutes > 0 and (time.monotonic() - arxiv_start_time) / 60.0 >= scrape_minutes:
            if active_arxiv:
                print("  arXiv: time limit reached (%.1f min). Stopping arXiv scrape." % scrape_minutes)
            active_arxiv = []

        for cat in list(active_arxiv):
            start = next_start_arxiv[cat]
            if start >= arxiv_max_per_source:
                continue
            req_size = min(arxiv_batch_size, arxiv_max_per_source - start)
            try:
                _, _, batch = _fetch_arxiv_task(cat, start, args.before_year, req_size)
            except Exception as e:
                print("  Fetch error (arxiv %s): %s" % (cat, e))
                continue
            items = [(t, "arxiv", aid) for t, aid in batch]
            with _OutputLock(out_path) as lock:
                written = lock.append_new(is_jsonl, seen_hashes, items)
            added_this_pass += written
            total_new += written
            next_start_arxiv[cat] = start + len(batch)
            if len(batch) < req_size:
                active_arxiv = [c for c in active_arxiv if c != cat]
            if written > 0:
                print("  arxiv %s start=%d: +%d new (total new this run: %d)" % (cat, start, written, total_new))
            time.sleep(ARXIV_DELAY_SEC)  # Required 3-second delay between consecutive arXiv requests

        active_arxiv = [c for c in active_arxiv if next_start_arxiv.get(c, 0) < arxiv_max_per_source]

        # PubMed: can run in parallel (arXiv limits do not apply)
        if active_pubmed:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for q in active_pubmed:
                    start = next_start_pubmed[q]
                    if start >= max_per_source:
                        continue
                    fut = executor.submit(_fetch_pubmed_task, q, start, args.before_year, batch_size_pubmed)
                    futures[fut] = q

                if futures:
                    for future in as_completed(futures):
                        q = futures[future]
                        try:
                            key_or_q, start, batch = future.result()
                        except Exception as e:
                            print("  Fetch error (pubmed %s): %s" % (q[:30], e))
                            continue
                        items = [(t, "pubmed", pid) for t, pid in batch]
                        with _OutputLock(out_path) as lock:
                            written = lock.append_new(is_jsonl, seen_hashes, items)
                        added_this_pass += written
                        total_new += written
                        next_start_pubmed[q] = start + len(batch)
                        if len(batch) < batch_size_pubmed:
                            active_pubmed = [x for x in active_pubmed if x != q]
                        if written > 0:
                            label = key_or_q[:30] + "..." if len(key_or_q) > 30 else key_or_q
                            print("  pubmed %s start=%d: +%d new (total new this run: %d)" % (label, start, written, total_new))

        active_pubmed = [q for q in active_pubmed if next_start_pubmed.get(q, 0) < max_per_source]

        if args.target > 0 and existing_lines + total_new >= args.target:
            break
        if not active_arxiv and not active_pubmed:
            print("No more pages to fetch; stopping.")
            break
        time.sleep(0.3)

    total_in_file = existing_lines + total_new
    print("\nDone. Wrote %d new abstracts to %s (total in file: %d)." % (total_new, out_path, total_in_file))
    if total_new == 0 and not seen_hashes:
        print("No abstracts fetched. Try different --before-year or categories/queries.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
