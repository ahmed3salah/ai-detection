"""
Data layer: load IDMGSP benchmark and/or local JSONL/separator-based files.
Unifies to (texts, labels) with optional train/val split.
Supports arxiv-metadata-pre-llm.jsonl (streaming sample for large files).
"""
from pathlib import Path
import json
import random
from typing import Optional, Union

from sklearn.model_selection import train_test_split


# Document separator for plain-text multi-document files
DOC_SEPARATOR = "---DOC---"


def load_idmgsp(
    subset: str = "classifier_input",
    text_mode: str = "abstract",
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load tum-nlp/IDMGSP from Hugging Face.

    Args:
        subset: Dataset split name: "classifier_input", "train+gpt3", "train-cg", etc.
        text_mode: "abstract" | "full" (abstract + introduction + conclusion)
        val_size: Fraction for validation (0..1). 0 = no split, return single train.
        random_state: For reproducible train/val split.

    Returns:
        If val_size > 0: (X_train, y_train, X_val, y_val) as lists of (text, label).
        Else: (X, y) as lists.
    """
    try:
        import datasets
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' to use IDMGSP: pip install datasets")

    # datasets >= 3.0 no longer supports loading scripts; skip call to avoid noisy deprecation message
    version = getattr(datasets, "__version__", "0.0.0")
    if version >= "3.0":
        raise ImportError(
            "IDMGSP requires dataset loading scripts; 'datasets' %s no longer supports them. "
            "Use --local and data in dataset/, or: pip install 'datasets<3.0'" % version
        )

    try:
        ds = load_dataset("tum-nlp/IDMGSP", subset)
    except Exception as e:
        raise ImportError(
            "IDMGSP could not be loaded. Use --local and data in dataset/, or: pip install 'datasets<3.0'"
        ) from e
    # Handle DatasetDict (e.g. train/test) or single Dataset
    if hasattr(ds, "keys"):
        # Prefer 'train' for main data; we'll split for val
        data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    else:
        data = ds

    texts, labels = _idmgsp_rows_to_texts_labels(data, text_mode)
    if val_size <= 0 or val_size >= 1:
        return (texts, labels)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=val_size, stratify=labels, random_state=random_state
    )
    return (X_train, y_train, X_val, y_val)


def _idmgsp_rows_to_texts_labels(data, text_mode):
    """Convert IDMGSP rows to (texts, labels)."""
    texts = []
    labels = []
    for row in data:
        if text_mode == "abstract":
            t = (row.get("abstract") or "").strip()
        else:
            parts = [
                (row.get("abstract") or "").strip(),
                (row.get("introduction") or "").strip(),
                (row.get("conclusion") or "").strip(),
            ]
            t = "\n\n".join(p for p in parts if p)
        if not t:
            continue
        label = int(row.get("label", 0))
        texts.append(t)
        labels.append(label)
    return texts, labels


def load_idmgsp_test(
    subset: str = "classifier_input",
    text_mode: str = "abstract",
):
    """
    Load IDMGSP test split if available; otherwise return ([], []).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return ([], [])
    ds = load_dataset("tum-nlp/IDMGSP", subset, trust_remote_code=True)
    if not hasattr(ds, "keys") or "test" not in ds:
        return ([], [])
    return _idmgsp_rows_to_texts_labels(ds["test"], text_mode)


def load_jsonl(path: Union[str, Path], text_mode: str = "abstract"):
    """
    Load JSONL for training. Supports two formats:

    1. Simple: {"text": "...", "label": 0|1}. Newlines in "text" as \\n.
    2. Arxiv-style: id, title, abstract, categories, ai_written, model, provider, ...
       Uses abstract as text and ai_written as label (same schema as generate_ai_data output).
       text_mode: "abstract" | "title_abstract" (title + " " + abstract).

    Auto-detects format from first line.
    """
    path = Path(path)
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Arxiv-style: abstract + ai_written (and optional title, model, provider)
            if "abstract" in obj and "ai_written" in obj:
                abstract = (obj.get("abstract") or "").strip().replace("\\n", "\n")
                if not abstract:
                    continue
                if text_mode == "title_abstract":
                    title = (obj.get("title") or "").strip().replace("\n", " ")
                    text = (title + " " + abstract).strip() or abstract
                else:
                    text = abstract
                label = int(obj.get("ai_written", 0))
                texts.append(text)
                labels.append(label)
                continue
            # Simple format: text + label
            text = (obj.get("text") or "").replace("\\n", "\n")
            label = int(obj.get("label", 0))
            if text:
                texts.append(text)
                labels.append(label)
    return (texts, labels)


# ArXiv metadata JSONL: id, title, abstract, categories, authors_parsed, update_date, ai_written
ARXIV_METADATA_JSONL = "arxiv-metadata-pre-llm.jsonl"


def load_arxiv_metadata_jsonl(
    path: Union[str, Path],
    max_samples: Optional[int] = 100_000,
    random_state: Optional[int] = 42,
    human_only: bool = True,
    text_mode: str = "abstract",
) -> tuple[list[str], list[int]]:
    """
    Load human abstracts from arxiv-metadata-pre-llm.jsonl (or similar).
    Uses reservoir sampling when max_samples is set so the file is not fully loaded into memory.

    Args:
        path: Path to the JSONL file.
        max_samples: Max number of abstracts to load (None = load all; can be huge).
        random_state: For reproducible sampling.
        human_only: If True, only include rows with ai_written == 0.
        text_mode: "abstract" | "title_abstract" (title + " " + abstract).

    Returns:
        (texts, labels) with label=0 (human) for each row.
    """
    path = Path(path)
    if not path.exists():
        return ([], [])

    rng = random.Random(random_state)
    texts: list[str] = []
    labels: list[int] = []
    n_seen = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if human_only and obj.get("ai_written", 0) != 0:
                continue
            abstract = (obj.get("abstract") or "").strip()
            if not abstract:
                continue
            if text_mode == "title_abstract":
                title = (obj.get("title") or "").strip()
                text = (title + " " + abstract).strip() or abstract
            else:
                text = abstract

            n_seen += 1
            if max_samples is None:
                texts.append(text)
                labels.append(0)
            elif n_seen <= max_samples:
                texts.append(text)
                labels.append(0)
            else:
                # Reservoir sampling: replace with probability max_samples / n_seen
                j = rng.randrange(n_seen)
                if j < max_samples:
                    texts[j] = text
                    # labels[j] already 0

    return (texts, labels)


def load_separator_based(
    path: Union[str, Path],
    separator: str = DOC_SEPARATOR,
    default_label: int = 0,
):
    """
    Load a file where documents are separated by separator (e.g. ---DOC---).
    If separator is not in the file, treat each non-empty line as one document (backward compat).
    Optional companion file with same base name and .labels suffix (one label per line);
    otherwise default_label is used for all.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if separator in raw:
        blocks = [b.strip() for b in raw.split(separator) if b.strip()]
    else:
        blocks = [line.strip() for line in raw.splitlines() if line.strip()]
    labels_path = path.with_suffix(path.suffix + ".labels")
    if labels_path.exists():
        labels = [int(l.strip()) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(labels) != len(blocks):
            raise ValueError(f"Label count ({len(labels)}) != document count ({len(blocks)})")
    else:
        labels = [default_label] * len(blocks)
    return (blocks, labels)


# Separate files for organization: abstracts vs reviews (human vs AI)
LOCAL_HUMAN_FILES = [
    "human_abstracts.txt",
    "human_reviews.txt",
    "human_text.txt",   # backward compat
]
LOCAL_AI_FILES = [
    "ai_abstracts.txt",
    "ai_reviews.txt",
    "ai_text.txt",      # backward compat
]


def load_local_dataset(
    dataset_dir: Union[str, Path] = "dataset",
    jsonl_name: Optional[str] = "data.jsonl",
    val_size: float = 0.2,
    random_state: int = 42,
    arxiv_metadata_max_samples: Optional[int] = 100_000,
):
    """
    Load local data: prefer data.jsonl if present, else separate text files.
    Human: human_abstracts.txt, human_reviews.txt (and human_text.txt for compat).
    AI: ai_abstracts.txt, ai_reviews.txt (and ai_text.txt for compat).
    If arxiv-metadata-pre-llm.jsonl exists in dataset_dir, human abstracts are
    loaded from it (sampled to arxiv_metadata_max_samples to avoid OOM). Set
    arxiv_metadata_max_samples=0 to skip; None to load all (can be slow/huge).
    Returns unified (texts, labels) or train/val split.
    """
    dataset_dir = Path(dataset_dir)
    texts = []
    labels = []

    # Allow passing a path to a JSONL file directly (e.g. dataset/data.jsonl)
    if dataset_dir.is_file():
        jsonl_path = dataset_dir
    else:
        jsonl_path = dataset_dir / (jsonl_name or "data.jsonl")
    if jsonl_path.exists():
        texts, labels = load_jsonl(jsonl_path)
    else:
        for name in LOCAL_HUMAN_FILES:
            path = dataset_dir / name
            if path.exists():
                t, lab = load_separator_based(path, default_label=0)
                texts.extend(t)
                labels.extend(lab)
        for name in LOCAL_AI_FILES:
            path = dataset_dir / name
            if path.exists():
                t, lab = load_separator_based(path, default_label=1)
                texts.extend(t)
                labels.extend(lab)

        # ArXiv pre-LLM metadata (large JSONL): add human abstracts with optional sampling
        if arxiv_metadata_max_samples != 0:
            arxiv_path = dataset_dir / ARXIV_METADATA_JSONL
            if arxiv_path.exists():
                cap = None if arxiv_metadata_max_samples is None else max(0, arxiv_metadata_max_samples)
                t, lab = load_arxiv_metadata_jsonl(
                    arxiv_path,
                    max_samples=cap if cap else None,
                    random_state=random_state,
                )
                if t:
                    texts.extend(t)
                    labels.extend(lab)

    if not texts:
        return (texts, labels) if val_size <= 0 or val_size >= 1 else (texts, labels, [], [])

    if val_size <= 0 or val_size >= 1:
        return (texts, labels)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=val_size, stratify=labels, random_state=random_state
    )
    return (X_train, y_train, X_val, y_val)
