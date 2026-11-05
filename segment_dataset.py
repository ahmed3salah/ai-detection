"""
Paragraph-level dataset for train_segments.py and evaluate_segments.py.

One JSON object per line:
    id        unique string
    text      the paragraph
    label     0 human, 1 AI
    paper_id  the source paper; an AI paragraph uses the paper it was generated from
    domain    e.g. the arXiv archive or "pmc"
    section   section heading class, e.g. "introduction"
    generator the model that wrote it; required when label is 1
    test_set  optional: "polished", "non_native" or "e2e". Never trained on.

Loading is strict. A bad line stops the run rather than being skipped or defaulted, because
silently dropped or mislabelled rows are how the last dataset went wrong.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

REQUIRED_FIELDS = ("id", "text", "label", "paper_id", "domain", "section")
TEST_ONLY_SETS = ("polished", "non_native", "e2e")

TRAIN = "train"
CALIBRATION = "calibration"
TEST = "test"
HELDOUT_GENERATOR = "heldout_generator"
HELDOUT_DOMAIN = "heldout_domain"

CALIBRATION_FRACTION = 0.1
TEST_FRACTION = 0.2

Row = Dict[str, Any]


class DatasetError(ValueError):
    """The dataset cannot be trained on as it stands."""


def read_jsonl(path: Union[str, Path]) -> List[Row]:
    """
    Rows of a JSONL file. Always UTF-8, never the system's encoding (cp1252 on Windows), and split
    only on real line ends: str.splitlines would also break a row at a raw U+2028 in its text.
    """
    with Path(path).open(encoding="utf-8", newline="\n") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Union[str, Path], rows: Iterable[Row]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_segments(path: Union[str, Path]) -> Tuple[List[Row], str]:
    """The dataset's rows and the sha256 of the file, which every artifact records."""
    path = Path(path)
    digest = hashlib.sha256()
    rows: List[Row] = []
    seen = set()

    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{number}: not valid JSON: {e.msg}") from e
            _validate(row, f"{path}:{number}", seen)
            rows.append(row)

    if not rows:
        raise DatasetError(f"{path}: no rows.")
    return rows, digest.hexdigest()


def _validate(row: Any, where: str, seen: set) -> None:
    if not isinstance(row, dict):
        raise DatasetError(f"{where}: expected a JSON object.")
    missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise DatasetError(f"{where}: missing {', '.join(missing)}.")
    if row["label"] not in (0, 1) or isinstance(row["label"], bool):
        raise DatasetError(f"{where}: label must be 0 or 1, got {row['label']!r}.")
    if row["label"] == 1 and not row.get("generator"):
        raise DatasetError(f"{where}: an AI row must name its generator.")
    test_set = row.get("test_set")
    if test_set is not None and test_set not in TEST_ONLY_SETS:
        raise DatasetError(f"{where}: unknown test_set {test_set!r}.")
    if row["id"] in seen:
        raise DatasetError(f"{where}: duplicate id {row['id']!r}.")
    seen.add(row["id"])


def assign_paper(paper_id: str, seed: int) -> str:
    """
    Which split a paper belongs to, from its id alone: it doesn't depend on row order or on
    what else is in the file. The generation script uses this too, to give held-out generators
    only test papers.
    """
    value = int(hashlib.sha256(f"{seed}:{paper_id}".encode()).hexdigest()[:12], 16) / 16**12
    if value < TEST_FRACTION:
        return TEST
    if value < TEST_FRACTION + CALIBRATION_FRACTION:
        return CALIBRATION
    return TRAIN


def split_segments(
    rows: Sequence[Row],
    seed: int,
    heldout_generators: Iterable[str] = (),
    heldout_domains: Iterable[str] = (),
) -> Dict[str, List[Row]]:
    """
    Split by source paper, so a paper and the AI paragraphs generated from it stay together.

    Test-only rows go to their own set, and their papers never reach training. A held-out
    domain goes to its own set whole. A held-out generator's rows are kept only for test papers:
    on a training paper, the model has already seen that paper's text.
    """
    heldout_generators = set(heldout_generators)
    heldout_domains = set(heldout_domains)
    reserved_papers = {row["paper_id"] for row in rows if row.get("test_set")}

    splits: Dict[str, List[Row]] = {
        name: [] for name in (TRAIN, CALIBRATION, TEST, HELDOUT_GENERATOR, HELDOUT_DOMAIN, *TEST_ONLY_SETS)
    }

    for row in rows:
        if row.get("test_set"):
            splits[row["test_set"]].append(row)
            continue
        if row["domain"] in heldout_domains:
            splits[HELDOUT_DOMAIN].append(row)
            continue

        split = TEST if row["paper_id"] in reserved_papers else assign_paper(row["paper_id"], seed)
        if row.get("generator") in heldout_generators:
            if split == TEST:
                splits[HELDOUT_GENERATOR].append(row)
            continue
        splits[split].append(row)

    assert_no_paper_overlap(splits)
    return splits


def assert_no_paper_overlap(splits: Dict[str, List[Row]]) -> None:
    """No paper used for fitting (train, calibration) may appear in anything used to measure."""
    fitting = {name: {row["paper_id"] for row in splits[name]} for name in (TRAIN, CALIBRATION)}
    if fitting[TRAIN] & fitting[CALIBRATION]:
        raise DatasetError("A paper is in both train and calibration.")
    for name, rows in splits.items():
        if name in fitting:
            continue
        leaked = (fitting[TRAIN] | fitting[CALIBRATION]) & {row["paper_id"] for row in rows}
        if leaked:
            raise DatasetError(f"{len(leaked)} paper(s) are in both a fitting split and {name}, e.g. {sorted(leaked)[0]}.")


def split_counts(splits: Dict[str, List[Row]]) -> Dict[str, Dict[str, int]]:
    return {
        name: {
            "rows": len(rows),
            "human": sum(row["label"] == 0 for row in rows),
            "ai": sum(row["label"] == 1 for row in rows),
            "papers": len({row["paper_id"] for row in rows}),
        }
        for name, rows in splits.items()
    }


def word_count(text: str) -> int:
    return len(text.split())

