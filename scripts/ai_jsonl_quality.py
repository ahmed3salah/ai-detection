"""
Shared validation for AI-generated arxiv-style JSONL records (title, categories, abstract).
Used by generate_ai_data.py and audit_ai_jsonl.py.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Minimum length for an abstract to be considered usable.
MIN_ABSTRACT_LEN = 50
# Max title length (chars) after normalization.
MAX_TITLE_LEN = 250

_GARBAGE_SUBSTRINGS = (
    "**Categories**",
    "titlerc",
    "categories\\",
    "  *   *",
    "Pert-urbOfUrbativeive",
    "Order Pert-urb",
    "OfUrbativeive",
    "quququ",
    "pertpert",
)

# Model narrates JSON construction instead of outputting JSON.
_META_JSON_PHRASES = (
    "i've assembled",
    "json output",
    "my title is now confirmed",
    "necessary components",
    "here is the json",
    "here's the json",
    "as requested",
    "below is the json",
    "putting together the json",
    "i'm now putting",
    "now putting together",
    "final json",
    "final json construction",
    "json object",
    "combine the title",
    "combine the title, categories",
    "now constructing the json",
    "here is the json object",
    "below is the json object",
    "single line of valid json",
    "output a single json",
)

# Prompt / instruction echo in abstract field.
_PROMPT_ECHO_SUBSTRINGS = (
    "(abstract):",
    "2-4 sentences, under 350",
    "2–4 sentences, under 350",
    "keys: title, categories, abstract",
    "arxiv-style codes",
)

# Homework / exam style (Chinese).
_HOMEWORK_CJK_MARKERS = (
    "反函数",
    "求它的反函数",
    "定义域",
    "值域",
)


def abstract_failure_reasons(text: str) -> List[str]:
    """Return human-readable reasons why abstract is not usable; empty list if OK."""
    reasons: List[str] = []
    if not text or not isinstance(text, str):
        return ["abstract_missing"]
    s = text.strip().replace("\n", " ")
    if len(s) < MIN_ABSTRACT_LEN:
        reasons.append("abstract_too_short")
    s_lower = s.lower()

    for bad in _GARBAGE_SUBSTRINGS:
        if bad.lower() in s_lower:
            reasons.append("abstract_garbage_substring")
            break

    for phrase in _META_JSON_PHRASES:
        if phrase in s_lower:
            reasons.append("abstract_meta_json")
            break

    for echo in _PROMPT_ECHO_SUBSTRINGS:
        if echo.lower() in s_lower:
            reasons.append("abstract_prompt_echo")
            break

    for marker in _HOMEWORK_CJK_MARKERS:
        if marker in s:
            reasons.append("abstract_homework_cjk")
            break

    cjk_count = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    if cjk_count > 25:
        reasons.append("abstract_heavy_cjk")

    if "## " in s or "### " in s:
        reasons.append("abstract_markdown_headings")

    if "**final json" in s_lower or "**categories**" in s_lower:
        reasons.append("abstract_markdown_meta")

    if "\\[" in s or "\\(" in s or "\\\\frac" in s_lower or "\\\\]" in s:
        reasons.append("abstract_latex_heavy")

    if s.startswith("-") and len(s) < 200:
        reasons.append("abstract_truncated_dash")

    if s.endswith(": A") or s.endswith(": A "):
        reasons.append("abstract_title_fragment")
    elif len(s) < 80 and s.count(" ") < 4:
        reasons.append("abstract_too_few_words")

    s_norm = s.replace('\\"', '"')
    if s_norm.startswith("{"):
        if '"title"' in s_norm and s_norm.count('"') >= 4:
            reasons.append("abstract_json_blob")
        elif len(s_norm) < 200 and "title" in s_norm and (":" in s_norm or '"' in s_norm):
            reasons.append("abstract_truncated_json")

    return reasons


def is_usable_abstract(text: str) -> bool:
    return len(abstract_failure_reasons(text)) == 0


def title_failure_reasons(title: str) -> List[str]:
    reasons: List[str] = []
    t = (title or "").strip().replace("\n", " ")
    if not t:
        reasons.append("empty_title")
        return reasons
    if len(t) > MAX_TITLE_LEN:
        reasons.append("title_too_long")
    tl = t.lower()
    if tl.startswith("title:") or tl.startswith('"title"'):
        reasons.append("title_echo_key")
    for phrase in ("json construction", "final json", "json object", "here is the"):
        if phrase in tl:
            reasons.append("title_meta_json")
            break
    return reasons


def validate_metadata_fields(
    title: str,
    categories: str,
    abstract: str,
    *,
    require_categories: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Full check for topic-generated JSONL rows (non-empty title, optional categories, usable abstract).
    Returns (ok, list of reason codes).
    """
    reasons: List[str] = []
    reasons.extend(title_failure_reasons(title))
    if require_categories and not (categories or "").strip():
        reasons.append("empty_categories")
    reasons.extend(abstract_failure_reasons(abstract))
    return (len(reasons) == 0, reasons)


def validate_record_dict(
    record: Dict[str, Any],
    *,
    require_categories: bool = True,
) -> Tuple[bool, List[str]]:
    if record.get("generation_failed"):
        return (False, ["generation_failed"])
    title = str(record.get("title") or "")
    categories = str(record.get("categories") or "")
    abstract = str(record.get("abstract") or "")
    return validate_metadata_fields(title, categories, abstract, require_categories=require_categories)


def parse_ai_index_from_id(record_id: Any) -> int | None:
    """Parse 'ai-123' -> 123; return None if not matching."""
    if record_id is None:
        return None
    s = str(record_id).strip()
    m = re.match(r"^ai-(\d+)$", s)
    return int(m.group(1)) if m else None
