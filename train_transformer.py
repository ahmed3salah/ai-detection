"""
Fine-tune a transformer to classify paragraphs as human or AI-written.

The second candidate next to train_segments.py's feature models: it reads the text itself, so it
can learn what 11 style features miss (the pilot's feature model caught only 6-7% of Claude's
paragraphs). Same dataset, splits, calibration and artifact rules as train_segments.py, so
evaluate_segments.py measures both the same way.

    python train_transformer.py --data dataset/segments/segments.jsonl --heldout-generators moonshotai/kimi-k3.1-pro,z-ai/glm-5.3
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from segment_dataset import CALIBRATION, TRAIN, DatasetError, Row, load_segments, split_counts, split_segments
from text_normalize import NORMALIZATION_VERSION, normalize_segment
from train_segments import LIKELY_AI_FPR, POSSIBLY_AI_FPR, TrainingAborted, threshold_for_fpr

DEFAULT_BASE_MODEL = "distilroberta-base"
# Paragraphs run to about 300 words, which is roughly 400 tokens: 384 covers nearly all of them.
DEFAULT_MAX_LENGTH = 384


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
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = {k: v.to(device) for k, v in encode(tokenizer, texts[start : start + batch_size], max_length).items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            out.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return np.array(out, dtype=np.float64)


def usable(rows: Sequence[Row], split: str) -> List[Row]:
    """Rows with text left after normalising. More than a few without is a dataset problem."""
    kept = [row for row in rows if normalize_segment(row["text"])]
    if len(rows) - len(kept) > max(1, len(rows) // 1000):
        raise TrainingAborted(f"{split}: {len(rows) - len(kept)} of {len(rows)} rows have no text after normalising.")
    return kept


def train(
    data: str,
    *,
    out_root: str = "artifacts",
    version: Optional[str] = None,
    base_model: str = DEFAULT_BASE_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
    heldout_generators: Sequence[str] = (),
    heldout_domains: Sequence[str] = (),
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    warmup_fraction: float = 0.1,
    device: Optional[str] = None,
) -> Path:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    rows, dataset_sha256 = load_segments(data)
    short_name = Path(base_model).name.replace("/", "-")
    version = version or f"transformer-{short_name}-{dataset_sha256[:8]}"
    target = Path(out_root) / version
    if target.exists():
        raise TrainingAborted(f"{target} already exists. Artifacts are never overwritten; pick another --version.")

    splits = split_segments(rows, seed, heldout_generators, heldout_domains)
    counts = split_counts(splits)
    print(f"[transformer] {version}: {json.dumps(counts)}")
    for split in (TRAIN, CALIBRATION):
        if not counts[split]["human"] or not counts[split]["ai"]:
            raise TrainingAborted(f"{split} needs both human and AI rows: {counts[split]}.")
    train_rows, calibration_rows = usable(splits[TRAIN], TRAIN), usable(splits[CALIBRATION], CALIBRATION)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2).to(run_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(train_rows) / batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * warmup_fraction), steps)
    calibration_texts = [row["text"] for row in calibration_rows]
    y_calibration = np.array([row["label"] for row in calibration_rows], dtype=np.float64)

    Path(out_root).mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=out_root, prefix=f".{version}."))
    try:
        best_loss, best_epoch = math.inf, -1
        for epoch in range(epochs):
            model.train()
            order = list(range(len(train_rows)))
            random.Random(seed + epoch).shuffle(order)
            for start in range(0, len(order), batch_size):
                batch_rows = [train_rows[i] for i in order[start : start + batch_size]]
                batch = {k: v.to(run_device) for k, v in encode(tokenizer, [r["text"] for r in batch_rows], max_length).items()}
                labels = torch.tensor([r["label"] for r in batch_rows], device=run_device)
                with torch.autocast(device_type=run_device.type, dtype=torch.bfloat16, enabled=run_device.type == "cuda"):
                    loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), labels)
                if not torch.isfinite(loss):
                    raise TrainingAborted("Non-finite training loss. Lower the learning rate.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            raw = np.clip(predict_raw(model, tokenizer, calibration_texts, max_length, run_device), 1e-7, 1 - 1e-7)
            held_loss = float(-np.mean(y_calibration * np.log(raw) + (1 - y_calibration) * np.log(1 - raw)))
            print(f"[transformer] epoch {epoch + 1}/{epochs}  calibration loss {held_loss:.4f}  device {run_device}")
            # The best epoch so far is kept on disk, not the last one.
            if held_loss < best_loss:
                best_loss, best_epoch = held_loss, epoch
                model.save_pretrained(out / "model")
                tokenizer.save_pretrained(out / "model")

        model = AutoModelForSequenceClassification.from_pretrained(out / "model").to(run_device)
        raw = predict_raw(model, tokenizer, calibration_texts, max_length, run_device)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y_calibration)
        joblib.dump(calibrator, out / "calibrator.pkl")

        human = calibrator.predict(raw[y_calibration == 0])
        likely = threshold_for_fpr(human, LIKELY_AI_FPR)
        thresholds = {"likely_ai": likely, "possibly_ai": min(threshold_for_fpr(human, POSSIBLY_AI_FPR), likely)}

        joblib.dump(
            {
                "model_version": version,
                "classifier_type": "transformer",
                "base_model": base_model,
                "max_length": max_length,
                "best_epoch": best_epoch + 1,
                "text_normalization": NORMALIZATION_VERSION,
                "segment_thresholds": thresholds,
                "dataset_sha256": dataset_sha256,
                "split_seed": seed,
                "heldout_generators": sorted(heldout_generators),
                "heldout_domains": sorted(heldout_domains),
                "split_counts": counts,
            },
            out / "model_config.pkl",
        )
        out.chmod(0o755)
        out.rename(target)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise

    print(f"[transformer] wrote {target}  best epoch {best_epoch + 1}  thresholds {thresholds}")
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--version")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="A Hugging Face model id or a local directory")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-generators", default="")
    parser.add_argument("--heldout-domains", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="cuda, cuda:1, cpu... Default: the first GPU if there is one")
    args = parser.parse_args(argv)
    try:
        train(
            args.data,
            out_root=args.out,
            version=args.version,
            base_model=args.base_model,
            max_length=args.max_length,
            seed=args.seed,
            heldout_generators=[g for g in args.heldout_generators.split(",") if g],
            heldout_domains=[d for d in args.heldout_domains.split(",") if d],
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (TrainingAborted, DatasetError) as e:
        print(f"[transformer] ABORTED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
