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


def predict_raw(model, tokenizer, texts: Sequence[str], max_length: int, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """P(AI) before calibration, in the order given."""
    model.eval()
    out: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = {k: v.to(device) for k, v in encode(tokenizer, texts[start : start + batch_size], max_length).items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            out.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return np.array(out, dtype=np.float64)
