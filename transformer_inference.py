"""
Running a fine-tuned paragraph transformer: shared by the trainer, the evaluator and the API, so the
server image doesn't carry the training code.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from text_normalize import normalize_segment


def pick_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(tokenizer, texts: Sequence[str], max_length: int) -> Dict[str, torch.Tensor]:
    return tokenizer(
        [normalize_segment(text) for text in texts],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def _logits(model, tokenizer, texts: Sequence[str], max_length: int, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    out: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = {k: v.to(device) for k, v in encode(tokenizer, texts[start : start + batch_size], max_length).items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out.append(model(**batch).logits.float().cpu())
    return torch.cat(out) if out else torch.zeros((0, 2))


def predict_raw(model, tokenizer, texts: Sequence[str], max_length: int, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """P(AI) before calibration, in the order given."""
    return torch.softmax(_logits(model, tokenizer, texts, max_length, device, batch_size), dim=-1)[:, 1].numpy().astype(np.float64)


def predict_log_odds(model, tokenizer, texts: Sequence[str], max_length: int, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    The model's log-odds of AI, in the order given. Unlike a probability it doesn't flatten near
    0 and 1, so it still moves when one sentence of a strongly flagged paragraph is left out.
    """
    logits = _logits(model, tokenizer, texts, max_length, device, batch_size)
    return (logits[:, 1] - logits[:, 0]).numpy().astype(np.float64)
