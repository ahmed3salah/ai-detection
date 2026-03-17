#!/usr/bin/env python3
"""
Diagnose training data: perplexity and AI-artifact features by label.
Run after merging human + AI data. Use --sample to limit work (perplexity is slow).
Expected: human abstracts often have higher perplexity (jargon, formulas); AI lower.
If the opposite holds, the classifier may learn the wrong direction.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Project root
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from data_loader import load_jsonl


def main():
    p = argparse.ArgumentParser(description="Diagnose dataset: perplexity and AI-artifact stats by label")
    p.add_argument("--data", type=Path, default=_root / "dataset/data.jsonl", help="JSONL with abstract + ai_written")
    p.add_argument("--sample", type=int, default=200, help="Max samples per class for perplexity (0 = all, slow)")
    p.add_argument("--no-perplexity", action="store_true", help="Skip perplexity (only print artifact stats)")
    args = p.parse_args()

    if not args.data.exists():
        print("Error: %s not found" % args.data)
        return 1

    texts, labels = load_jsonl(args.data)
    labels = np.array(labels)
    n_human = int((labels == 0).sum())
    n_ai = int((labels == 1).sum())
    print("Loaded %d samples: %d human (0), %d AI (1)" % (len(texts), n_human, n_ai))

    # --- AI-artifact features (fast) ---
    from features import extract_features
    X = np.array([extract_features(t) for t in texts], dtype=np.float64)
    names = getattr(extract_features, "feature_names", [])
    human_idx = labels == 0
    ai_idx = labels == 1

    print("\n--- AI-artifact features (mean by label) ---")
    for i, name in enumerate(names):
        if "this_paper" in name or "alternative" in name or "markdown" in name or "formulaic" in name:
            h_mean = X[human_idx, i].mean()
            a_mean = X[ai_idx, i].mean()
            print("  %s:  human=%.3f  ai=%.3f  (higher in AI is good)" % (name, h_mean, a_mean))

    # --- Perplexity (slow) ---
    if args.no_perplexity:
        print("\nSkipping perplexity (--no-perplexity). Done.")
        return 0

    from detector import calculate_perplexity

    def sample_per_class(texts, labels, n_per_class):
        human_texts = [t for t, l in zip(texts, labels) if l == 0]
        ai_texts = [t for t, l in zip(texts, labels) if l == 1]
        np.random.seed(42)
        if n_per_class > 0 and len(human_texts) > n_per_class:
            human_texts = list(np.random.choice(human_texts, n_per_class, replace=False))
        if n_per_class > 0 and len(ai_texts) > n_per_class:
            ai_texts = list(np.random.choice(ai_texts, n_per_class, replace=False))
        return human_texts, ai_texts

    n_sample = args.sample
    human_texts, ai_texts = sample_per_class(texts, labels, n_sample)
    print("\n--- Perplexity (SciBERT, sampled %d per class) ---" % len(human_texts))

    human_ppl = []
    for i, t in enumerate(human_texts):
        if (i + 1) % 50 == 0:
            print("  Human %d/%d..." % (i + 1, len(human_texts)))
        try:
            ppl = calculate_perplexity(t, model_type="scibert", aggregate="mean")
            human_ppl.append(ppl)
        except Exception as e:
            print("  Warning: %s" % e)
    ai_ppl = []
    for i, t in enumerate(ai_texts):
        if (i + 1) % 50 == 0:
            print("  AI %d/%d..." % (i + 1, len(ai_texts)))
        try:
            ppl = calculate_perplexity(t, model_type="scibert", aggregate="mean")
            ai_ppl.append(ppl)
        except Exception as e:
            print("  Warning: %s" % e)

    human_ppl = np.array(human_ppl)
    ai_ppl = np.array(ai_ppl)
    print("\n  Human:  mean ppl=%.2f  std=%.2f  min=%.2f  max=%.2f" % (
        human_ppl.mean(), human_ppl.std(), human_ppl.min(), human_ppl.max()))
    print("  AI:     mean ppl=%.2f  std=%.2f  min=%.2f  max=%.2f" % (
        ai_ppl.mean(), ai_ppl.std(), ai_ppl.min(), ai_ppl.max()))

    if human_ppl.mean() > ai_ppl.mean():
        print("\n  -> Human has HIGHER perplexity (expected). Model should learn: high ppl -> human.")
    else:
        print("\n  -> AI has HIGHER perplexity. If the model learns 'high ppl -> AI', predictions may be inverted.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
