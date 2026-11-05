#!/usr/bin/env python3
"""Populate Hugging Face hub cache for offline inference (used during Docker image build)."""
from __future__ import annotations

import os

# Honor HF_HOME set in Dockerfile before importing transformers.
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")


def main() -> None:
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
    )

    GPT2TokenizerFast.from_pretrained("gpt2")
    GPT2LMHeadModel.from_pretrained("gpt2")
    AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    AutoModelForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased")
    print("prefetch_hf_models: cached gpt2 and allenai/scibert_scivocab_uncased")


if __name__ == "__main__":
    main()
