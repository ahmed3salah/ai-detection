# Dual-GPU training (2x RTX A6000)

When you have **two or more GPUs**, you can use both by training with **dual perplexity**: GPT-2 runs on `cuda:0` and SciBERT on `cuda:1`. Each text gets four perplexity-derived features instead of two (`perplexity_gpt2`, `log_perplexity_gpt2`, `perplexity_scibert`, `log_perplexity_scibert`).

## How to use

Train with the `--dual-perplexity` flag:

```bash
python Training.py --dual-perplexity
```

You can combine it with other options (e.g. `--dataset-dir dataset`, `--tfidf`). Dual perplexity **implies** perplexity features; do not use `--no-perplexity` with `--dual-perplexity` (training will raise an error).

## Behaviour

- **With 2+ GPUs:** GPT-2 is loaded on `cuda:0`, SciBERT on `cuda:1`. Both are used during feature extraction (training and inference).
- **With 1 GPU:** Both models are loaded on the same device; the feature vector is still the 4-value dual-perplexity vector, so checkpoints remain compatible.

The saved config stores `use_dual_perplexity: True` and the longer `feature_names` list. Evaluation (`evaluate.py`) and the app (`app.py`) read this flag and call the dual-perplexity extractor when loading a model trained with `--dual-perplexity`.

## Backward compatibility

Models trained **without** `--dual-perplexity` are unchanged. Configs that do not contain `use_dual_perplexity` (or have it `False`) continue to use the single-perplexity extractor. Existing `model.pkl` / `model_config.pkl` checkpoints remain valid.
