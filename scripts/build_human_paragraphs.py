"""
Build the human side of the paragraph dataset from open-licensed papers written before LLMs.

Sources: arXiv papers licensed CC BY or CC0, read through GROBID as production reads PDFs, and the
PMC Open Access Subset (JATS XML) under the same licences. Licence and first-submission date come
from each source's OAI-PMH metadata. Papers first submitted after 2021 are left out, and so is any
licence other than CC BY or CC0.

Safe to stop and rerun: finished papers are skipped, and PDFs and TEI are cached.

    python scripts/build_human_paragraphs.py --papers 200 --grobid-url http://localhost:8070
"""
import argparse
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import segmenter  # noqa: E402

ARXIV_OAI = "https://oaipmh.arxiv.org/oai"
ARXIV_PDF = "https://export.arxiv.org/pdf/{id}"
PMC_OAI = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"

# arXiv asks for no more than one request every 3 seconds.
POLITE_DELAY_SECONDS = 3.1
USER_AGENT = "sparcly-ai-detection-dataset/1.0 (research; open-licensed papers only)"

LAST_ALLOWED_YEAR = 2021
HARVEST_START = date(2016, 1, 1)
HARVEST_END = date(LAST_ALLOWED_YEAR, 12, 31)

# Weighted toward the CS and engineering conferences Sparcly serves.
DOMAIN_WEIGHTS = {"cs": 0.45, "eess": 0.15, "pmc": 0.20, "physics": 0.10, "stat": 0.05, "math": 0.05}

OPEN_LICENCE = re.compile(r"creativecommons\.org/(?:licenses/by/|publicdomain/zero/)", re.I)
MIN_SEGMENTS_PER_PAPER = 5
MIN_ALPHA_RATIO = 0.6
MIN_ENGLISH_STOPWORD_RATIO = 0.15
ENGLISH_STOPWORDS = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "we", "with", "on", "this", "are", "as", "by"}

SECTION_CLASSES = (
    ("introduction", r"introduc|background|motivation|related work|literature review|overview"),
    ("methods", r"method|material|approach|experiment(al)? (setup|design)|model|implementation|design|procedure|data|proposed|system|algorithm|framework"),
    ("results", r"result|finding|evaluation|performance|analysis|experiment"),
    ("discussion", r"discussion|limitation|implication"),
    ("conclusion", r"conclu|summary|future work|outlook"),
)

_JATS_NON_PROSE = {"fig", "fig-group", "table-wrap", "table-wrap-group", "disp-formula", "inline-formula", "supplementary-material", "ref-list", "fn", "fn-group", "boxed-text", "graphic"}


@dataclass(frozen=True)
class Candidate:
    paper_id: str
    domain: str
    title: str
    licence: str
    first_date: str
    source_url: str
    jats: Optional[ET.Element] = None
    arxiv_id: Optional[str] = None


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1] if isinstance(element.tag, str) else ""


def find_local(element: ET.Element, name: str) -> Optional[ET.Element]:
    return next((child for child in element.iter() if local_name(child) == name), None)


def text_of(element: Optional[ET.Element]) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def section_class(heading: Optional[str], kind: str = segmenter.KIND_PARAGRAPH) -> str:
    """The rough kind of section a heading names, for the report's per-section numbers."""
    if kind == segmenter.KIND_ABSTRACT:
        return "abstract"
    heading = (heading or "").lower()
    for name, pattern in SECTION_CLASSES:
        if re.search(pattern, heading):
            return name
    return "other"


def is_english(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and sum(word in ENGLISH_STOPWORDS for word in words) / len(words) >= MIN_ENGLISH_STOPWORD_RATIO


def is_prose(text: str) -> bool:
    """Not a table or formula that survived as a paragraph."""
    letters = sum(character.isalpha() or character.isspace() for character in text)
    return bool(text) and letters / len(text) >= MIN_ALPHA_RATIO and is_english(text)


def arxiv_first_year(identifier: str) -> Optional[int]:
    """
    The year of first submission, from the id arXiv assigned then: 2008.03408 or cond-mat/0504132.
    The OAI <created> field can't be used: it gives 2021-05-31 for 1902.10899.
    """
    match = re.search(r"(?:^|/)(\d{2})\d{2}", identifier)
    if match is None:
        return None
    year = int(match.group(1))
    return 1900 + year if year >= 91 else 2000 + year


def parse_arxiv_records(xml: str) -> Tuple[List[Candidate], Optional[str], int]:
    """Eligible papers on a page of arXiv OAI-PMH, its resumption token, and how many records it had."""
    root = ET.fromstring(xml)
    candidates = []
    records = [element for element in root.iter() if local_name(element) == "record"]
    for record in records:
        arxiv = find_local(record, "arXiv")
        if arxiv is None:
            continue
        fields = {local_name(child): child for child in arxiv}
        licence = text_of(fields.get("license"))
        categories = text_of(fields.get("categories")).split()
        identifier = text_of(fields.get("id"))
        first_year = arxiv_first_year(identifier)
        if not OPEN_LICENCE.search(licence) or first_year is None or first_year > LAST_ALLOWED_YEAR or not categories:
            continue
        candidates.append(Candidate(
            paper_id=f"arxiv:{identifier}",
            domain=categories[0].split(".")[0],
            title=text_of(fields.get("title")),
            licence=licence,
            first_date=str(first_year),
            source_url=f"https://arxiv.org/abs/{identifier}",
            arxiv_id=identifier,
        ))
    return candidates, resumption_token(root), len(records)


def resumption_token(root: ET.Element) -> Optional[str]:
    token = find_local(root, "resumptionToken")
    return (token.text or "").strip() or None if token is not None else None


def jats_licence(article: ET.Element) -> str:
    """The licence URL an article declares, if any. A bare license-type="open-access" isn't enough."""
    XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
    for element in article.iter():
        if local_name(element) == "license" and element.get(XLINK_HREF):
            return element.get(XLINK_HREF)
    for element in article.iter():
        if local_name(element) == "license_ref" and element.text:
            return element.text.strip()
    for element in article.iter():
        if local_name(element) == "license":
            for link in element.iter():
                if link.get(XLINK_HREF) and "creativecommons" in link.get(XLINK_HREF):
                    return link.get(XLINK_HREF)
    return ""


def parse_pmc_records(xml: str) -> Tuple[List[Candidate], Optional[str], int]:
    root = ET.fromstring(xml)
    candidates = []
    records = [element for element in root.iter() if local_name(element) == "record"]
    for record in records:
        article = find_local(record, "article")
        identifier = find_local(record, "identifier")
        if article is None or identifier is None or not identifier.text or find_local(article, "body") is None:
            continue
        pmc_id = identifier.text.rsplit(":", 1)[-1]
        licence = jats_licence(article)
        years = [int(text_of(y)) for pub in article.iter() if local_name(pub) == "pub-date" for y in pub if local_name(y) == "year" and text_of(y).isdigit()]
        if not OPEN_LICENCE.search(licence) or not years or min(years) > LAST_ALLOWED_YEAR:
            continue
        candidates.append(Candidate(
            paper_id=f"pmc:{pmc_id}",
            domain="pmc",
            title=text_of(find_local(article, "article-title")),
            licence=licence,
            first_date=str(min(years)),
            source_url=f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/",
            jats=article,
        ))
    return candidates, resumption_token(root), len(records)


def _jats_prose(element: ET.Element) -> str:
    text = element.text or ""
    for child in element:
        text += " " if local_name(child) in _JATS_NON_PROSE else _jats_prose(child)
        text += child.tail or ""
    return text


def _walk_jats(element: ET.Element, section: Optional[str], kind: str, out: List[segmenter.Segment]) -> None:
    for child in element:
        name = local_name(child)
        if name in _JATS_NON_PROSE:
            continue
        if name == "p":
            out.append(segmenter.Segment(kind, _jats_prose(child), section))
        elif name == "sec":
            title = next((c for c in child if local_name(c) == "title"), None)
            _walk_jats(child, text_of(title) or section, kind, out)
        else:
            _walk_jats(child, section, kind, out)


def segments_from_jats(article: ET.Element) -> List[segmenter.Segment]:
    """The same shape from_tei gives: abstract and body paragraphs, evened out by the same rules."""
    segments: List[segmenter.Segment] = []
    abstract = find_local(article, "abstract")
    if abstract is not None:
        _walk_jats(abstract, "Abstract", segmenter.KIND_ABSTRACT, segments)
    body = find_local(article, "body")
    if body is not None:
        _walk_jats(body, None, segmenter.KIND_PARAGRAPH, segments)
    return segmenter.normalize_segments(segments)


def rows_for(candidate: Candidate, segments: Sequence[segmenter.Segment]) -> List[Dict]:
    """Dataset rows for a paper's usable paragraphs. No paragraphs if the paper isn't usable."""
    rows = []
    for position, segment in enumerate(segments):
        if segmenter.count_words(segment.text) < segmenter.MIN_SCORABLE_WORDS or not is_prose(segment.text):
            continue
        rows.append({
            "id": f"{candidate.paper_id}#{position}",
            "text": segment.text,
            "label": 0,
            "paper_id": candidate.paper_id,
            "domain": candidate.domain,
            "section": section_class(segment.section, segment.kind),
            "heading": segment.section,
            "position": position,
            "title": candidate.title,
            "licence": candidate.licence,
            "source_url": candidate.source_url,
            "first_date": candidate.first_date,
        })
    return rows if len(rows) >= MIN_SEGMENTS_PER_PAPER else []


class Fetcher:
    """Requests with a polite pace and retries when a source says it is busy."""

    def __init__(self, session: Optional[requests.Session] = None, delay: float = POLITE_DELAY_SECONDS):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.last_request = 0.0

    def get(self, url: str, params: Optional[Dict] = None, timeout: int = 120, attempts: int = 5) -> requests.Response:
        for attempt in range(attempts):
            wait = self.last_request + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
                time.sleep(10 * (attempt + 1))
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                retry_after = response.headers.get("Retry-After", "")
                time.sleep(int(retry_after) if retry_after.isdigit() else 15 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise RuntimeError(f"Gave up on {url}")


def harvest(fetcher: Fetcher, domain: str, rng: random.Random) -> Iterator[Candidate]:
    """
    Eligible papers of a domain, from random weeks of 2016-2021 so no one period dominates.

    The date window also keeps out later revisions: OAI selects on a record's last change, so a paper
    revised after 2021 isn't returned, and the PDF fetched is the version written before LLMs.
    """
    span = (HARVEST_END - HARVEST_START).days
    while True:
        start = HARVEST_START + timedelta(days=rng.randrange(span))
        end = min(start + timedelta(days=6), HARVEST_END)
        if domain == "pmc":
            url, parse = PMC_OAI, parse_pmc_records
            params = {"verb": "ListRecords", "metadataPrefix": "pmc", "set": "pmc-open", "from": start.isoformat(), "until": end.isoformat()}
        else:
            url, parse = ARXIV_OAI, parse_arxiv_records
            params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": domain, "from": start.isoformat(), "until": end.isoformat()}

        for _ in range(3):  # Up to three pages of a week, then on to another week.
            xml = fetcher.get(url, params).text
            candidates, token, _ = parse(xml)
            rng.shuffle(candidates)
            yield from candidates
            if not token:
                break
            params = {"verb": "ListRecords", "resumptionToken": token}


def grobid_tei(session: requests.Session, grobid_url: str, pdf: bytes, attempts: int = 4) -> str:
    """Full-text TEI, with the parameters GrobidExtractionService::extractFulltextXml uses."""
    for attempt in range(attempts):
        response = session.post(
            f"{grobid_url.rstrip('/')}/api/processFulltextDocument",
            files=[("input", ("paper.pdf", pdf, "application/pdf"))],
            data=[("consolidateHeader", "0"), ("consolidateCitations", "0"), ("teiCoordinates", "p"), ("teiCoordinates", "head")],
            headers={"Accept": "application/xml"},
            timeout=300,
        )
        if response.status_code == 503 and attempt < attempts - 1:  # Busy: all its threads are taken.
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        if "<TEI" not in response.text[:2000]:
            raise RuntimeError("GROBID did not answer with TEI.")
        return response.text
    raise RuntimeError("GROBID stayed busy.")


def check_grobid(grobid_url: str) -> None:
    try:
        response = requests.get(f"{grobid_url.rstrip('/')}/api/isalive", timeout=15)
    except requests.RequestException as e:
        raise SystemExit(f"GROBID isn't reachable at {grobid_url}: {e}\nStart it first (see the commands you were given).")
    if response.status_code != 200 or "true" not in response.text.lower():
        raise SystemExit(f"GROBID at {grobid_url} isn't ready (status {response.status_code}).")


def quotas(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Whole papers per domain, adding up to exactly total."""
    counts = {domain: int(total * weight) for domain, weight in weights.items()}
    for domain in sorted(weights, key=lambda d: -weights[d])[: total - sum(counts.values())]:
        counts[domain] += 1
    return counts


def process_arxiv(candidate: Candidate, pdf_path: Path, tei_path: Path, grobid_url: str, session: requests.Session) -> List[segmenter.Segment]:
    if not tei_path.exists():
        tei_path.write_text(grobid_tei(session, grobid_url, pdf_path.read_bytes()))
    return segmenter.from_tei(tei_path.read_text())


def build(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    workdir = Path(args.workdir)
    (workdir / "cache" / "pdf").mkdir(parents=True, exist_ok=True)
    (workdir / "cache" / "tei").mkdir(parents=True, exist_ok=True)
    output = workdir / "human_segments.jsonl"
    log_path = workdir / "papers.jsonl"

    done: Dict[str, str] = {}
    accepted: Dict[str, int] = {domain: 0 for domain in DOMAIN_WEIGHTS}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            entry = json.loads(line)
            done[entry["paper_id"]] = entry["status"]
            if entry["status"] == "ok":
                accepted[entry["quota"]] = accepted.get(entry["quota"], 0) + 1

    targets = quotas(args.papers, DOMAIN_WEIGHTS)
    if not args.no_arxiv:
        check_grobid(args.grobid_url)

    fetcher = Fetcher()
    session = requests.Session()
    rng = random.Random(args.seed)
    stats: Dict[str, Dict[str, int]] = {domain: {"accepted": accepted.get(domain, 0), "failed": 0} for domain in targets}

    def record(candidate: Candidate, quota: str, segments: Optional[List[segmenter.Segment]], error: str = "") -> None:
        rows = rows_for(candidate, segments or [])
        status = "ok" if rows else ("failed" if error else "too_few_paragraphs")
        with output.open("a") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with log_path.open("a") as log:
            log.write(json.dumps({"paper_id": candidate.paper_id, "quota": quota, "status": status, "paragraphs": len(rows), "error": error[:200]}) + "\n")
        done[candidate.paper_id] = status
        stats[quota]["accepted" if rows else "failed"] += 1
        print(f"[human] {candidate.paper_id:28s} {status:20s} {len(rows):4d} paragraphs   {quota}: {stats[quota]['accepted']}/{targets[quota]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for domain, target in targets.items():
            if domain != "pmc" and args.no_arxiv:
                continue
            pending: List[Tuple[Candidate, Future]] = []

            def drain(limit: int) -> None:
                while len(pending) > limit:
                    candidate, future = pending.pop(0)
                    try:
                        record(candidate, domain, future.result())
                    except Exception as e:  # One bad PDF costs that paper, not the run.
                        record(candidate, domain, None, f"{type(e).__name__}: {e}")

            for candidate in harvest(fetcher, domain, rng):
                if stats[domain]["accepted"] + len(pending) >= target:
                    drain(0)
                    if stats[domain]["accepted"] >= target:
                        break
                if candidate.paper_id in done:
                    continue
                done[candidate.paper_id] = "pending"
                if candidate.jats is not None:
                    record(candidate, domain, segments_from_jats(candidate.jats))
                    continue
                pdf_path = workdir / "cache" / "pdf" / f"{candidate.arxiv_id.replace('/', '_')}.pdf"
                tei_path = workdir / "cache" / "tei" / f"{candidate.arxiv_id.replace('/', '_')}.tei.xml"
                try:
                    if not pdf_path.exists() and not tei_path.exists():
                        pdf_path.write_bytes(fetcher.get(ARXIV_PDF.format(id=candidate.arxiv_id), timeout=180).content)
                except Exception as e:
                    record(candidate, domain, None, f"{type(e).__name__}: {e}")
                    continue
                pending.append((candidate, pool.submit(process_arxiv, candidate, pdf_path, tei_path, args.grobid_url, session)))
                drain(args.workers * 2)
            drain(0)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers", type=int, required=True, help="Total papers to collect across domains")
    parser.add_argument("--workdir", default="dataset/segments")
    parser.add_argument("--grobid-url", default=os.environ.get("GROBID_URL", "http://localhost:8070"))
    parser.add_argument("--workers", type=int, default=4, help="PDFs GROBID reads at once")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-arxiv", action="store_true", help="PMC only: needs no GROBID")
    args = parser.parse_args(argv)

    stats = build(args)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
