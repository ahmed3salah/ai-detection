"""
Text clean-up shared by dataset building and inference.

Human paragraphs come out of PDFs through GROBID with citation markers, formula remnants and
ligatures. Generated paragraphs have none of that, so a model trained on the raw text could
learn "[12] means human" instead of anything about the writing. Both sides go through this.

A model opts in through text_normalization in its model_config.pkl. Bump NORMALIZATION_VERSION
when the rules change: a model has to be served with the rules it was trained on.
"""
import re
import unicodedata

NORMALIZATION_VERSION = "v1"

_HYPHEN_LINE_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=[a-z])")

_LATEX_MATH = re.compile(r"\$[^$\n]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?")

# [12], [1, 2], [1-3], [1]-[3]
_NUMERIC_CITATION = re.compile(r"\[\s*\d+(?:\s*[-\u2010-\u2015,;]\s*\d+)*\s*\](?:\s*[-\u2010-\u2015,;]\s*\[\s*\d+\s*\])*")

_NAME = r"[A-Z][\w'\u2019-]+(?:\s+[A-Z][\w'\u2019-]+)*"
_AUTHORS = rf"{_NAME}(?:\s+et\s+al\.?|\s*(?:,|&|and)\s*{_NAME})*"
_YEAR = r"(?:19|20)\d{2}[a-z]?"
# (Smith, 2019), (Smith et al. 2019; Jones and Lee, 2020a)
_AUTHOR_YEAR_CITATION = re.compile(
    rf"\(\s*(?:see\s+|e\.g\.,?\s+)?{_AUTHORS},?\s+{_YEAR}(?:\s*[;,]\s*(?:{_AUTHORS},?\s+)?{_YEAR})*\s*\)"
)
# Smith et al. (2019): the name stays, it reads as prose; the year goes.
_YEAR_IN_PARENS = re.compile(rf"\(\s*{_YEAR}(?:\s*[;,]\s*{_YEAR})*\s*\)")

_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?)\]])")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]")
_WHITESPACE = re.compile(r"\s+")


def _is_math_symbol_token(token: str) -> bool:
    """A token that is only math symbols, as GROBID leaves behind where a formula was."""
    return bool(token) and all(unicodedata.category(c) in ("Sm", "So") for c in token)


def normalize_segment(text: str) -> str:
    """The text with citation markers, math remnants and PDF artefacts taken out."""
    # NFKC folds PDF ligatures (fi, fl) and compatibility forms into plain letters.
    text = unicodedata.normalize("NFKC", text or "")
    text = _HYPHEN_LINE_BREAK.sub("", text)
    text = _LATEX_MATH.sub(" ", text)
    text = _LATEX_COMMAND.sub(" ", text)
    text = _NUMERIC_CITATION.sub(" ", text)
    text = _AUTHOR_YEAR_CITATION.sub(" ", text)
    text = _YEAR_IN_PARENS.sub(" ", text)
    text = _EMPTY_BRACKETS.sub(" ", text)
    text = " ".join(token for token in text.split() if not _is_math_symbol_token(token))
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()
