"""
Feature extraction for AI vs human scientific text.
Includes generic style + scientific-domain (passive, hedging, diversity, structure).
"""
import re
import subprocess
import sys

import numpy as np
import spacy

# Hedging cues common in scientific writing (AI may over/under-use)
HEDGING_WORDS = {
    "may", "might", "could", "would", "suggest", "suggests", "suggested",
    "likely", "perhaps", "possibly", "generally", "often", "sometimes",
    "usually", "typically", "appear", "appears", "seem", "seems",
    "indicate", "indicates", "indicated", "potential", "probably",
    "unclear", "presumably", "relatively", "somewhat", "rather",
}

nlp = None


def _get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed; download it (e.g. first run or new env)
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
    return nlp


def _safe_mean(arr, default=0.0):
    if arr is None or len(arr) == 0:
        return default
    return float(np.mean(arr))


def _safe_var(arr, default=0.0):
    if arr is None or len(arr) == 0:
        return default
    return float(np.var(arr))


def _is_passive_sentence(sent):
    """Heuristic: sentence has a passive dependency (aux + past participle)."""
    for token in sent:
        if token.dep_ == "nsubjpass" or token.dep_ == "csubjpass":
            return True
        if token.tag_ == "VBN" and token.dep_ == "ROOT":
            # Check for auxiliary
            for child in token.children:
                if child.dep_ == "auxpass":
                    return True
    return False


def extract_features(text):
    """
    Extract a fixed-size feature vector for classification.
    Order must match feature_names for model compatibility.
    """
    text = (text or "").strip()
    if not text:
        return _empty_features()

    doc = _get_nlp()(text)
    sentences = list(doc.sents)
    words = [t.text for t in doc if t.is_alpha]
    word_count = len(words)
    sentence_count = len(sentences)

    # --- Original features ---
    sentence_lengths = [len(s) for s in sentences]
    avg_sentence_length = _safe_mean(sentence_lengths)
    sentence_variance = _safe_var(sentence_lengths)
    avg_word_length = _safe_mean([len(w) for w in words]) if words else 0.0
    punctuation_count = len([t for t in doc if t.is_punct])

    # --- Length / structure ---
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)
    avg_words_per_paragraph = word_count / paragraph_count if paragraph_count else 0.0

    # --- Lexical diversity ---
    unique_words = len(set(w.lower() for w in words))
    type_token_ratio = unique_words / word_count if word_count else 0.0

    # --- Scientific style: passive voice ratio ---
    passive_count = sum(1 for s in sentences if _is_passive_sentence(s))
    passive_ratio = passive_count / sentence_count if sentence_count else 0.0

    # --- Hedging ---
    hedging_count = sum(1 for t in doc if t.text.lower() in HEDGING_WORDS)
    hedging_ratio = hedging_count / sentence_count if sentence_count else 0.0

    # --- Punctuation and numbers ---
    comma_count = sum(1 for t in doc if t.text == ",")
    semicolon_count = sum(1 for t in doc if t.text == ";")
    commas_per_sentence = comma_count / sentence_count if sentence_count else 0.0
    semicolons_per_sentence = semicolon_count / sentence_count if sentence_count else 0.0
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / len(text) if text else 0.0

    # --- Citation-like ---
    et_al_count = len(re.findall(r"\bet\s+al\.?", text, re.IGNORECASE))
    paren_open = text.count("(")
    citation_like = et_al_count + min(paren_open, 50)  # cap parens

    # --- AI-artifact / formulaic cues (strong discriminative signal) ---
    text_lower = text.lower()
    first_80 = text_lower[:80]
    first_120 = text_lower[:120]
    starts_with_this_paper = 1.0 if ("this paper" in first_80 or "in this paper" in first_80) else 0.0
    contains_alternative_abstract = 1.0 if ("alternative abstract" in text_lower or "here's an alternative" in text_lower) else 0.0
    contains_markdown_abstract = 1.0 if re.search(r"\*\*abstract\*\*", text_lower) else 0.0
    formulaic_opening = 1.0 if (
        "this paper presents" in first_120
        or "this paper introduces" in first_120
        or "we introduce" in first_120
        or "we present" in first_120
        or "in this work we" in first_120
        or "in this article we" in first_120
    ) else 0.0

    return [
        avg_sentence_length,
        sentence_variance,
        avg_word_length,
        punctuation_count,
        word_count,
        sentence_count,
        paragraph_count,
        avg_words_per_paragraph,
        type_token_ratio,
        passive_ratio,
        hedging_ratio,
        commas_per_sentence,
        semicolons_per_sentence,
        digit_ratio,
        citation_like,
        starts_with_this_paper,
        contains_alternative_abstract,
        contains_markdown_abstract,
        formulaic_opening,
    ]


def _empty_features(include_perplexity=False):
    n = 19 + (2 if include_perplexity else 0)
    return [0.0] * n


def extract_features_with_perplexity(text, model_type="scibert", aggregate="mean"):
    """
    Extract feature vector including perplexity and log(perplexity) for classifier.
    Use this when training or when the deployed model was trained with perplexity.
    """
    base = extract_features(text)
    try:
        from detector import get_perplexity_for_features
        ppl, log_ppl = get_perplexity_for_features(text, model_type=model_type, aggregate=aggregate)
        # Clip extreme values for stability
        ppl = min(ppl, 1e6) if ppl != float("inf") else 1e6
        log_ppl = min(log_ppl, 20.0) if log_ppl != float("inf") else 20.0
        return base + [ppl, log_ppl]
    except Exception:
        return base + [0.0, 0.0]


# For config and debugging: ordered feature names (base only; + perplexity, log_perplexity when used)
extract_features.feature_names = [
    "avg_sentence_length",
    "sentence_variance",
    "avg_word_length",
    "punctuation_count",
    "word_count",
    "sentence_count",
    "paragraph_count",
    "avg_words_per_paragraph",
    "type_token_ratio",
    "passive_ratio",
    "hedging_ratio",
    "commas_per_sentence",
    "semicolons_per_sentence",
    "digit_ratio",
    "citation_like",
    "starts_with_this_paper",
    "contains_alternative_abstract",
    "contains_markdown_abstract",
    "formulaic_opening",
]
FEATURE_NAMES_WITH_PPL = extract_features.feature_names + ["perplexity", "log_perplexity"]
