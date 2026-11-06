"""
Does the model judge the writing, or the PDF it came out of?

Human training paragraphs carry GROBID's noise (stray spaces, run-together words); generated
ones were never in a PDF. A model could learn "clean text means AI", which would miss real
AI-written papers (they go through GROBID too) and flag human text pasted in clean.

For each test paper this builds two documents with identical layout, one from its human paragraphs
and one from the AI paragraphs that replaced them, prints both to PDF with Edge, reads them back
through GROBID like production, and scores what comes out. If the model reads the writing, the
results barely move from scoring the same text directly.

    python scripts/render_check.py --artifacts artifacts/<version>
"""
import argparse
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import segmenter  # noqa: E402
from segment_dataset import HELDOUT_GENERATOR, TEST, read_jsonl, split_segments  # noqa: E402

EDGE_PATHS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

# Two columns of justified serif text, like most conference papers.
PAGE_CSS = """
@page { size: A4; margin: 18mm 15mm; }
body { font-family: "Times New Roman", serif; font-size: 10pt; line-height: 1.25; }
h1 { font-size: 15pt; text-align: center; margin: 0 0 10pt; }
main { column-count: 2; column-gap: 7mm; text-align: justify; hyphens: auto; }
h2 { font-size: 11pt; margin: 8pt 0 4pt; }
p { margin: 0 0 6pt; text-indent: 1em; }
"""


def document_html(title: str, sections: Sequence[Dict]) -> str:
    """A paper-like page: title, then each section's heading and paragraphs in order."""
    body = []
    for section in sections:
        if section.get("heading"):
            body.append(f"<h2>{html.escape(section['heading'])}</h2>")
        body.extend(f"<p>{html.escape(text)}</p>" for text in section["paragraphs"])
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{PAGE_CSS}</style></head><body><h1>{html.escape(title)}</h1><main>{''.join(body)}</main></body></html>"
    )


def paired_documents(
    merged: Sequence[Dict],
    generated: Sequence[Dict],
    human: Sequence[Dict],
    seed: int,
    heldout_generators: Sequence[str],
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    For each test paper: its AI rows and the human rows they replaced, in the paper's order. The
    merged dataset picks the rows; the generation output says which paragraph each one replaced.
    """
    splits = split_segments(merged, seed, heldout_generators)
    by_id = {row["id"]: row for row in human}
    generated_by_id = {row["id"]: row for row in generated}
    ai_rows = [generated_by_id[row["id"]] for name in (TEST, HELDOUT_GENERATOR) for row in splits[name] if row["label"] == 1]
    papers: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"human": [], "ai": []})
    for row in ai_rows:
        source = by_id[row["source_segment_id"]]
        papers[row["paper_id"]]["ai"].append({**row, "heading": source.get("heading"), "position": source.get("position", 0), "title": source.get("title", "")})
        papers[row["paper_id"]]["human"].append(source)
    for paper in papers.values():
        for side in paper.values():
            side.sort(key=lambda row: row.get("position", 0))
    return dict(papers)


def sections_of(rows: Sequence[Dict]) -> List[Dict]:
    sections: List[Dict] = []
    for row in rows:
        heading = row.get("heading")
        if not sections or sections[-1]["heading"] != heading:
            sections.append({"heading": heading, "paragraphs": []})
        sections[-1]["paragraphs"].append(row["text"])
    return sections


def print_pdf(edge: str, html_path: Path, pdf_path: Path) -> None:
    subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
        check=True,
        timeout=120,
        capture_output=True,
    )
    if not pdf_path.exists() or not pdf_path.read_bytes().startswith(b"%PDF"):
        raise RuntimeError(f"Edge did not write {pdf_path.name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--workdir", default="dataset/segments")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--papers", type=int, default=0, help="Limit, for a quick look")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_human_paragraphs import grobid_tei  # noqa: E402
    from segment_model import SegmentModel  # noqa: E402

    edge = next((path for path in EDGE_PATHS if Path(path).exists()), None)
    if edge is None:
        print("Microsoft Edge not found; it prints the documents to PDF.")
        return 1

    model = SegmentModel(args.artifacts)
    config = model.config
    likely = model.thresholds["likely_ai"]
    workdir = Path(args.workdir)
    out_dir = workdir / "render_check"
    out_dir.mkdir(exist_ok=True)
    papers = paired_documents(
        read_jsonl(workdir / "segments.jsonl"),
        read_jsonl(workdir / "ai_segments.jsonl"),
        read_jsonl(workdir / "human_segments.jsonl"),
        config["split_seed"],
        config["heldout_generators"],
    )
    if args.papers:
        papers = dict(list(sorted(papers.items()))[: args.papers])

    session = requests.Session()
    results = {side: {"direct": [], "rendered": [], "documents": 0, "failed": 0} for side in ("human", "ai")}
    for paper_id, sides in sorted(papers.items()):
        title = sides["human"][0].get("title") or "Untitled"
        for side, rows in sides.items():
            # Not Path.with_suffix: the dot in an arXiv id would read as the extension.
            name = f"{paper_id.replace(':', '_').replace('/', '_')}-{side}"
            html_path, pdf_path = out_dir / f"{name}.html", out_dir / f"{name}.pdf"
            try:
                html_path.write_text(document_html(title, sections_of(rows)), encoding="utf-8")
                print_pdf(edge, html_path, pdf_path)
                segments = segmenter.from_tei(grobid_tei(session, args.grobid_url, pdf_path.read_bytes()))
            except Exception as e:
                results[side]["failed"] += 1
                print(f"[render] {paper_id} {side}: {type(e).__name__}: {str(e)[:120]}")
                continue
            texts = [s.text for s in segments if s.kind == segmenter.KIND_PARAGRAPH and segmenter.count_words(s.text) >= segmenter.MIN_SCORABLE_WORDS]
            rendered, _ = model.predict(texts) if texts else (np.zeros(0), [])
            direct, _ = model.predict([row["text"] for row in rows])
            results[side]["direct"].extend(direct.tolist())
            results[side]["rendered"].extend(rendered.tolist())
            results[side]["documents"] += 1
            print(f"[render] {paper_id:28s} {side:5s} {len(rows):3d} paragraphs in, {len(texts):3d} back, flagged {np.mean(np.array(rendered) >= likely) if len(rendered) else 0:.0%} (direct {np.mean(direct >= likely):.0%})")

    summary = {}
    for side, r in results.items():
        direct, rendered = np.array(r["direct"]), np.array(r["rendered"])
        summary[side] = {
            "documents": r["documents"],
            "failed": r["failed"],
            "paragraphs_direct": len(direct),
            "paragraphs_rendered": len(rendered),
            "flagged_likely_direct": round(float(np.mean(direct >= likely)), 4) if len(direct) else None,
            "flagged_likely_rendered": round(float(np.mean(rendered >= likely)), 4) if len(rendered) else None,
            "mean_probability_direct": round(float(direct.mean()), 4) if len(direct) else None,
            "mean_probability_rendered": round(float(rendered.mean()), 4) if len(rendered) else None,
        }
    report = {"model_version": config["model_version"], "likely_ai_threshold": likely, **summary}
    (Path(args.artifacts) / "render_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
