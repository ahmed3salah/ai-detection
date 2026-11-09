"""
Build the list of words and phrases AI-written paragraphs use far more often than human ones.

The report shows these as "AI-typical wording" beside a flagged paragraph's score, with how
much more common each is in AI-written text. They are measured on our own data, not read
out of the model, and the report says so.

Only training-split papers are counted, so the list never sees the papers the model was
tested on, and every AI paragraph sits next to the human paragraphs of the same paper, which
cancels most topic words. A term is kept when it is clearly and reliably AI-typical:

- weighted log-odds z-score (Monroe, Colaresi & Quinn 2008, informative Dirichlet prior) of
  at least MIN_Z,
- at least MIN_RATIO times more frequent per word in AI paragraphs than in human ones,
- in at least MIN_AI_PARAGRAPHS AI paragraphs, from at least MIN_GENERATORS different models,
  so one model's habit doesn't speak for all of them.

The test split then measures how often the list fires on human writing, the number the
report quotes as the cue's false-alarm rate.

    python scripts/build_wording_lexicon.py --data dataset/segments/segments.jsonl \\
        --out artifacts/scibert-full-1108/wording.json
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_dataset import TEST, TRAIN, assign_paper, read_jsonl  # noqa: E402

MIN_Z = 4.0
# Measured on the test split when the list was first built: at 15x a term turns up in 2.9% of
# human paragraphs (and 40% of AI ones); at 3x it was 70.7%, too often to mean anything.
MIN_RATIO = 15.0
MIN_AI_PARAGRAPHS = 50
MIN_GENERATORS = 5
MAX_TERMS = 300

# A word is shown as the phrase it sits in when the phrase accounts for this share of its uses.
COVERED_SHARE = 0.8

# Words joined by hyphens or apostrophes stay one token ("state-of-the-art", "model's").
_TOKEN = re.compile(r"[a-z]+(?:['’-][a-z]+)*")

# Maths and LaTeX commands ("$\\mathcal{L}$", "\\phi") say how a model writes formulas, not prose.
_MATH = re.compile(r"\$[^$]*\$|\\[A-Za-z]+")


def stopwords() -> Set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return set(ENGLISH_STOP_WORDS)


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(_MATH.sub(" ", text).lower())


def terms(text: str, stop: Set[str]) -> List[str]:
    """Unigrams that aren't stopwords, and bigrams with at least one word that isn't."""
    words = tokens(text)
    out = [word for word in words if word not in stop and len(word) > 2]
    out += [f"{a} {b}" for a, b in zip(words, words[1:]) if not (a in stop and b in stop) and len(a) > 1 and len(b) > 1]
    return out


def is_usable_row(row: Dict) -> bool:
    return not row.get("test_set") and row.get("label") in (0, 1) and bool(row.get("text"))


def count(rows: Iterable[Dict], stop: Set[str]) -> Tuple[Counter, Counter, Counter, Dict[str, Set[str]], Dict[int, int]]:
    """Term counts per class, AI paragraph frequency, generators per term and words per class."""
    by_class = {0: Counter(), 1: Counter()}
    ai_paragraphs: Counter = Counter()
    generators: Dict[str, Set[str]] = defaultdict(set)
    words = {0: 0, 1: 0}

    for row in rows:
        label = row["label"]
        found = terms(row["text"], stop)
        by_class[label].update(found)
        words[label] += len(tokens(row["text"]))
        if label == 1:
            distinct = set(found)
            ai_paragraphs.update(distinct)
            for term in distinct:
                generators[term].add(row.get("generator") or "?")

    return by_class[1], by_class[0], ai_paragraphs, generators, words


def log_odds_z(ai: Counter, human: Counter) -> Dict[str, float]:
    """
    Weighted log-odds of a term being AI rather than human, with the pooled counts as an
    informative Dirichlet prior, divided by its standard error.
    """
    pooled = ai + human
    alpha_total = sum(pooled.values())
    n_ai = sum(ai.values())
    n_human = sum(human.values())
    scores = {}

    for term, alpha in pooled.items():
        y_ai, y_human = ai[term], human[term]
        delta = math.log((y_ai + alpha) / (n_ai + alpha_total - y_ai - alpha)) - math.log(
            (y_human + alpha) / (n_human + alpha_total - y_human - alpha)
        )
        scores[term] = delta / math.sqrt(1.0 / (y_ai + alpha) + 1.0 / (y_human + alpha))

    return scores


def build(rows: Sequence[Dict], seed: int = 42, stop: Optional[Set[str]] = None) -> Dict:
    stop = stop if stop is not None else stopwords()
    train = [row for row in rows if is_usable_row(row) and assign_paper(row["paper_id"], seed) == TRAIN]
    ai, human, ai_paragraphs, generators, words = count(train, stop)
    z = log_odds_z(ai, human)

    lexicon = []
    for term, score in z.items():
        ai_rate = ai[term] * 10_000 / max(1, words[1])
        human_rate = (human[term] + 0.5) * 10_000 / max(1, words[0])
        ratio = ai_rate / human_rate
        if score < MIN_Z or ratio < MIN_RATIO or ai_paragraphs[term] < MIN_AI_PARAGRAPHS or len(generators[term]) < MIN_GENERATORS:
            continue
        lexicon.append({
            "phrase": term,
            "ratio": round(ratio, 1),
            "z": round(score, 1),
            "ai_per_10k": round(ai_rate, 2),
            "human_per_10k": round(human[term] * 10_000 / max(1, words[0]), 2),
            "generators": len(generators[term]),
        })

    lexicon.sort(key=lambda entry: (-entry["z"], entry["phrase"]))
    lexicon = drop_covered_unigrams(lexicon[:MAX_TERMS])
    return {"terms": lexicon, "measured": measure(rows, lexicon, seed), "seed": seed}


def drop_covered_unigrams(lexicon: List[Dict]) -> List[Dict]:
    """
    A word that mostly appears inside a kept phrase ("delves" in "delves into") is shown as the
    phrase only: it is dropped when the phrase accounts for at least COVERED_SHARE of its uses.
    """
    phrases = [entry for entry in lexicon if " " in entry["phrase"]]

    def covered(entry: Dict) -> bool:
        return any(
            entry["phrase"] in phrase["phrase"].split() and phrase["ai_per_10k"] >= COVERED_SHARE * entry["ai_per_10k"]
            for phrase in phrases
        )

    return [entry for entry in lexicon if " " in entry["phrase"] or not covered(entry)]


def measure(rows: Sequence[Dict], lexicon: Sequence[Dict], seed: int) -> Dict:
    """How often at least one term fires on test-split human and AI paragraphs."""
    patterns = [compile_phrase(entry["phrase"]) for entry in lexicon]
    test = [row for row in rows if is_usable_row(row) and assign_paper(row["paper_id"], seed) == TEST]
    hits = {0: 0, 1: 0}
    totals = {0: 0, 1: 0}
    for row in test:
        totals[row["label"]] += 1
        if any(pattern.search(row["text"]) for pattern in patterns):
            hits[row["label"]] += 1
    share = lambda label: round(100 * hits[label] / totals[label], 1) if totals[label] else None
    return {"human_paragraphs_with_a_term_percent": share(0), "ai_paragraphs_with_a_term_percent": share(1), "test_paragraphs": totals}


def compile_phrase(phrase: str) -> "re.Pattern[str]":
    """A phrase matched as whole words, case-insensitively, with any whitespace between its words."""
    return re.compile(r"(?<![A-Za-z])" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"(?![A-Za-z])", re.IGNORECASE)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset/segments/segments.jsonl")
    parser.add_argument("--out", required=True, help="Usually the model's artifacts folder: <dir>/wording.json")
    parser.add_argument("--seed", type=int, default=42, help="Must match the model's split seed")
    parser.add_argument("--show", type=int, default=40, help="Print this many of the strongest terms")
    args = parser.parse_args(argv)

    result = build(read_jsonl(args.data), seed=args.seed)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{len(result['terms'])} terms; measured on the test split: {json.dumps(result['measured'])}")
    for entry in result["terms"][: args.show]:
        print(f"  {entry['phrase']:28s} {entry['ratio']:6.1f}x  z={entry['z']:5.1f}  generators={entry['generators']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
