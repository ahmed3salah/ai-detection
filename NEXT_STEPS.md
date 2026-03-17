# Next Steps After Improving the AI Detector

This document lists what was changed in code and what you should do next to get better detection results.

## What Was Changed in Code

### 1. New AI-artifact features (`features.py`)

Four new binary (0/1) features were added so the model can use strong cues that often appear in AI-generated abstracts:

- **starts_with_this_paper** – First 80 chars contain "this paper" or "in this paper"
- **contains_alternative_abstract** – Text contains "alternative abstract" or "here's an alternative"
- **contains_markdown_abstract** – Text contains `**abstract**` (markdown)
- **formulaic_opening** – First 120 chars contain phrases like "this paper presents", "we introduce", "in this work we"

The feature vector is now **19 base + 2 perplexity = 21** when using perplexity. You **must retrain** so the saved model uses this size.

### 2. Diagnostic script (`scripts/diagnose_dataset.py`)

A script that, on your merged `dataset/data.jsonl`:

- Reports **mean of the new AI-artifact features by label** (human vs AI). You want higher means for AI on these.
- Optionally computes **perplexity on a sample per class** and prints mean/std. Expected: human mean perplexity **higher** than AI. If the script says the opposite, the classifier may be learning the wrong direction.

**Run it:**

```bash
# Quick: only AI-artifact stats (no perplexity)
python3 scripts/diagnose_dataset.py --no-perplexity

# Full: include perplexity on 200 samples per class (slow)
python3 scripts/diagnose_dataset.py --sample 200
```

---

## What You Should Do Next

### Step 1: Retrain with the new features

The model on disk was trained with 17 features. The code now uses 21. Retrain so the classifier and config match:

```bash
python3 training.py --dataset-dir dataset
# or
python3 training.py --dataset-dir dataset/data.jsonl
```

After this, `model.pkl` and `model_config.pkl` will use the new 21-dimensional feature vector.

### Step 2: Run the diagnostic

```bash
python3 scripts/diagnose_dataset.py --no-perplexity
```

Check that the new AI-artifact features are **higher on average for AI** than for human. If they are, the new features are informative.

Optionally run with perplexity (takes a while):

```bash
python3 scripts/diagnose_dataset.py --sample 200
```

If it reports that **human has higher mean perplexity than AI**, that’s the expected direction. If it reports the opposite, consider collecting more or different data so the perplexity–label relationship is clearer.

### Step 3: Add unpaired AI data (recommended)

Right now all AI abstracts are **paraphrases** of your human abstracts (same topic). The model has to separate classes using only style and artifacts, which is hard.

- Generate some AI abstracts **from topics only** (no human abstract as input), e.g. with `--topics` or `--topic-file`, so you have AI text that is **not** paired with a specific human abstract.
- Merge those into your training JSONL (same schema: `abstract`, `ai_written=1`, `model`, `provider`) and retrain. That gives the model more diverse AI examples and can improve generalization.

### Step 4: Optional: try TF-IDF

If accuracy is still low after retraining and (optionally) adding unpaired AI:

```bash
python3 training.py --dataset-dir dataset --tfidf
```

TF-IDF adds n-gram text features; it can help when formulaic phrasing differs between human and AI.

### Step 5: Evaluate

After retraining, run evaluation on your validation split:

```bash
python3 evaluate.py --dataset-dir dataset
```

Check accuracy and F1. If AI is still classified as “more human” than human, run the diagnostic again and consider adding more unpaired AI data or revisiting the data balance.

---

## Summary

| Action | Command / note |
|--------|-----------------|
| Retrain (required) | `python3 training.py --dataset-dir dataset` |
| Diagnose artifacts only | `python3 scripts/diagnose_dataset.py --no-perplexity` |
| Diagnose with perplexity | `python3 scripts/diagnose_dataset.py --sample 200` |
| Add unpaired AI | Generate with topics only, merge into data.jsonl, retrain |
| Optional TF-IDF | `python3 training.py --dataset-dir dataset --tfidf` |
| Evaluate | `python3 evaluate.py --dataset-dir dataset` |

---

## 15k human + 15k AI (10k from real + 5k from topics)

A **15k human** sample has been created from the full arxiv file:

- **`dataset/arxiv-metadata-pre-llm-sample-15k.jsonl`** – 15,000 human abstracts (same schema as before).

**Your steps:**

1. **Refresh topics (optional)** – Topics are in `dataset/topics.txt` (filled from your existing list + arxiv titles + OpenAlex). To refresh from arxiv + OpenAlex again:
   ```bash
   python3 scripts/fetch_topics.py --arxiv-max 1500 --openalex-pages 5
   ```

2. **Generate 5k AI from topics** (unpaired AI data):
   ```bash
   python3 scripts/generate_ai_data.py --topic-file dataset/topics.txt --output dataset/ai_from_topics_5k.jsonl --limit 5000
   ```
   Uses the first 5000 lines of `dataset/topics.txt` (one topic per line).

3. **Merge 15k human + 10k AI (from real) + 5k AI (from topics)** into one training file:
   ```bash
   python3 scripts/merge_dataset.py \
     --human dataset/arxiv-metadata-pre-llm-sample-15k.jsonl \
     --ai dataset/ai_generated_test_10k.jsonl dataset/ai_from_topics_5k.jsonl \
     --output dataset/data.jsonl
   ```
   This produces **30k rows** (15k human + 15k AI), shuffled.

4. **Retrain** on the new `dataset/data.jsonl`:
   ```bash
   python3 training.py --dataset-dir dataset
   ```
