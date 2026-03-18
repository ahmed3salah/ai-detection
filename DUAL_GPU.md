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

## PyTorch and GPU support (Blackwell / RTX 50-series)

GPUs with **Blackwell** (sm_120), e.g. NVIDIA RTX PRO 6000 Blackwell, need PyTorch built with **CUDA 12.8**. The default `pip install torch` (or `torch` from `requirements.txt`) is often CPU-only or an older CUDA build that does not support sm_120.

On the machine with the GPUs, install the CUDA 12.8 wheel after installing the rest of the project:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then run training as usual; the detector and classifier will use the GPUs. Use **PyTorch 2.7 or 2.8** with the `cu128` index so that Blackwell is supported.
