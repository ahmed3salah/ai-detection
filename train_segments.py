"""
Train a paragraph-level AI detector from a segment_dataset.py JSONL file.

Replaces training.py for paragraph models, and refuses what training.py let through. That
pipeline swallowed a crash in the perplexity code, trained on four columns of zeros, and
overwrote the deployed model in the repo root. Here:
  - a feature that fails, is non-finite or never varies stops the run;
  - papers never cross splits;
  - the epoch kept is the best on calibration data, not the last;
  - everything lands in artifacts/<version>/, written all at once, and never overwrites.

    python train_segments.py --data dataset/segments.jsonl --model mlp
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from classifier_mlp import MLPClassifier, PyTorchClassifierWrapper, assert_mlp_weights_finite, get_device, save_mlp
from segment_dataset import (
    CALIBRATION,
    TRAIN,
    DatasetError,
    Row,
    load_segments,
    split_counts,
    split_segments,
)
from segment_model import feature_names, feature_rows
from text_normalize import NORMALIZATION_VERSION, normalize_segment

# Of rows whose features could not be measured, before the whole run is called off.
MAX_FEATURE_FAILURE_RATE = 0.001

# Human false-positive rates the bands are drawn at, on calibration data.
LIKELY_AI_FPR = 0.01
POSSIBLY_AI_FPR = 0.05


class TrainingAborted(RuntimeError):
    """The data can't produce a trustworthy model. Stop, rather than train one anyway."""


def checked_matrix(rows: Sequence[Optional[List[float]]], names: Sequence[str], split: str) -> Tuple[np.ndarray, List[int]]:
    """
    The feature matrix and the indices of the rows in it. Aborts on too many failed rows, on any
    non-finite value, or on a feature that never varies (a column of fallback zeros, say).
    """
    failed = sum(row is None for row in rows)
    if not rows or failed / len(rows) > MAX_FEATURE_FAILURE_RATE:
        raise TrainingAborted(f"{split}: features failed for {failed} of {len(rows)} rows.")

    kept = [i for i, row in enumerate(rows) if row is not None]
    matrix = np.array([rows[i] for i in kept], dtype=np.float64)

    bad = [names[c] for c in range(matrix.shape[1]) if not np.isfinite(matrix[:, c]).all()]
    if bad:
        raise TrainingAborted(f"{split}: non-finite values in {', '.join(bad)}.")

    constant = [names[c] for c in range(matrix.shape[1]) if np.ptp(matrix[:, c]) == 0]
    if constant:
        raise TrainingAborted(
            f"{split}: these features never vary: {', '.join(constant)}. A failing feature falling back to a "
            "constant looks exactly like this. Fix it, or drop it from the feature set."
        )
    return matrix, kept


def threshold_for_fpr(human_probabilities: np.ndarray, target_fpr: float) -> float:
    """The lowest threshold at which at most target_fpr of human text scores at or above it."""
    descending = np.sort(np.asarray(human_probabilities, dtype=np.float64))[::-1]
    allowed = int(math.floor(target_fpr * len(descending)))
    if allowed >= len(descending):
        return 0.0
    return float(np.nextafter(descending[allowed], np.inf))


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
) -> MLPClassifier:
    """The epoch with the lowest calibration loss, not simply the last one."""
    device = get_device()
    torch.manual_seed(seed)
    model = MLPClassifier(input_size=X_train.shape[1], hidden_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    X = torch.from_numpy(X_train.astype(np.float32))
    y = torch.from_numpy(y_train.astype(np.float32))
    X_held = torch.from_numpy(X_calibration.astype(np.float32)).to(device)
    y_held = torch.from_numpy(y_calibration.astype(np.float32)).to(device)

    best_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(X), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X[batch].to(device)), y[batch].to(device))
            if not torch.isfinite(loss):
                raise TrainingAborted("Non-finite training loss. Check the features, or lower the learning rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            held_loss = float(criterion(model(X_held), y_held))
        if held_loss < best_loss:
            best_loss = held_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[train] epoch {epoch + 1}/{epochs}  calibration loss {held_loss:.4f}  device {device}")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    assert_mlp_weights_finite(model, "After training")
    return model


def _measured(rows: List[Row], feature_set: str, names: Sequence[str], split: str) -> Tuple[np.ndarray, np.ndarray]:
    measured = feature_rows([normalize_segment(row["text"]) for row in rows], feature_set)
    matrix, kept = checked_matrix(measured, names, split)
    return matrix, np.array([rows[i]["label"] for i in kept], dtype=np.float64)


def train(
    data: str,
    *,
    out_root: str = "artifacts",
    version: Optional[str] = None,
    model_kind: str = "mlp",
    feature_set: str = "style_v2",
    seed: int = 42,
    heldout_generators: Sequence[str] = (),
    heldout_domains: Sequence[str] = (),
    epochs: int = 40,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> Path:
    """Train, calibrate and write one artifacts directory. Returns its path."""
    rows, dataset_sha256 = load_segments(data)
    version = version or f"{feature_set}-{model_kind}-{dataset_sha256[:8]}"
    target = Path(out_root) / version
    if target.exists():
        raise TrainingAborted(f"{target} already exists. Artifacts are never overwritten; pick another --version.")

    splits = split_segments(rows, seed, heldout_generators, heldout_domains)
    counts = split_counts(splits)
    print(f"[train] {version}: {json.dumps(counts)}")
    for split in (TRAIN, CALIBRATION):
        if not counts[split]["human"] or not counts[split]["ai"]:
            raise TrainingAborted(f"{split} needs both human and AI rows: {counts[split]}.")

    names = feature_names(feature_set)
    X_train, y_train = _measured(splits[TRAIN], feature_set, names, TRAIN)
    X_calibration, y_calibration = _measured(splits[CALIBRATION], feature_set, names, CALIBRATION)

    Path(out_root).mkdir(parents=True, exist_ok=True)
    # Written beside the target and renamed at the end, so a failed run leaves nothing behind.
    out = Path(tempfile.mkdtemp(dir=out_root, prefix=f".{version}."))
    try:
        if model_kind == "mlp":
            scaler = StandardScaler().fit(X_train)
            joblib.dump(scaler, out / "feature_scaler.pkl")
            model = _train_mlp(
                scaler.transform(X_train), y_train, scaler.transform(X_calibration), y_calibration,
                seed=seed, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size,
            )
            save_mlp(model, out / "model.pt")
            raw = PyTorchClassifierWrapper(model, next(model.parameters()).device).predict_proba(scaler.transform(X_calibration))[:, 1]
            classifier_type = "pytorch"
        elif model_kind == "gbm":
            classifier = HistGradientBoostingClassifier(random_state=seed).fit(X_train, y_train)
            joblib.dump(classifier, out / "model.pkl")
            raw = classifier.predict_proba(X_calibration)[:, 1]
            classifier_type = "sklearn"
        else:
            raise TrainingAborted(f"Unknown model kind {model_kind!r}.")

        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y_calibration)
        joblib.dump(calibrator, out / "calibrator.pkl")

        human = calibrator.predict(raw[y_calibration == 0])
        likely = threshold_for_fpr(human, LIKELY_AI_FPR)
        thresholds = {"likely_ai": likely, "possibly_ai": min(threshold_for_fpr(human, POSSIBLY_AI_FPR), likely)}

        joblib.dump(
            {
                "model_version": version,
                "classifier_type": classifier_type,
                "feature_set": feature_set,
                "feature_names": names,
                "text_normalization": NORMALIZATION_VERSION,
                "segment_thresholds": thresholds,
                "use_perplexity_features": False,
                "use_dual_perplexity": False,
                "use_tfidf": False,
                "use_feature_scaler": model_kind == "mlp",
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

    print(f"[train] wrote {target}  thresholds {thresholds}")
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Segment dataset JSONL")
    parser.add_argument("--out", default="artifacts", help="Directory the version's folder is created in")
    parser.add_argument("--version", help="Default: <feature set>-<model>-<dataset hash>")
    parser.add_argument("--model", choices=["mlp", "gbm"], default="mlp")
    parser.add_argument("--feature-set", default="style_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-generators", default="", help="Comma-separated")
    parser.add_argument("--heldout-domains", default="", help="Comma-separated")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    try:
        train(
            args.data,
            out_root=args.out,
            version=args.version,
            model_kind=args.model,
            feature_set=args.feature_set,
            seed=args.seed,
            heldout_generators=[g for g in args.heldout_generators.split(",") if g],
            heldout_domains=[d for d in args.heldout_domains.split(",") if d],
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
        )
    except (TrainingAborted, DatasetError) as e:
        print(f"[train] ABORTED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
