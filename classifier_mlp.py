"""
PyTorch MLP classifier for AI detection. Supports CUDA, MPS (Apple Silicon), and CPU.
Provides sklearn-like predict_proba(X) and predict(X) for compatibility with detector/app/evaluate.
"""
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Device selection (CUDA > MPS > CPU)
# -----------------------------------------------------------------------------


def get_device() -> torch.device:
    """Return best available device: cuda, then mps (Apple Silicon), then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# MLP model
# -----------------------------------------------------------------------------


class MLPClassifier(nn.Module):
    """Binary classifier MLP: input -> hidden -> single logit (BCE)."""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Wrapper for sklearn-like API (predict_proba, predict)
# -----------------------------------------------------------------------------


class PyTorchClassifierWrapper:
    """Wraps a trained MLPClassifier to expose predict_proba(X) and predict(X) with numpy in/out."""

    def __init__(self, model: MLPClassifier, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) with [:, 1] = P(AI). X: (n, n_features) float64."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
            logits = self.model(t)
            p = torch.sigmoid(logits).cpu().numpy()
        # sklearn convention: [P(human), P(AI)]
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n,) of 0 or 1. X: (n, n_features) float64."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(np.int64)


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------


def save_mlp(model: MLPClassifier, path: Union[str, Path]) -> None:
    """Save state_dict and input_size so the model can be reconstructed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": model.input_size, "hidden_size": model.hidden_size},
        path,
    )


def load_mlp(path: Union[str, Path], device: Optional[torch.device] = None) -> MLPClassifier:
    """Load MLP from model.pt; place on device (default: get_device())."""
    path = Path(path)
    data = torch.load(path, map_location="cpu")
    input_dim = data["input_dim"]
    hidden_size = data.get("hidden_size", 128)
    model = MLPClassifier(input_size=input_dim, hidden_size=hidden_size)
    model.load_state_dict(data["state_dict"])
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Unified classifier loader (PyTorch or legacy joblib)
# -----------------------------------------------------------------------------


def load_classifier(
    config_path: Union[str, Path] = "model_config.pkl",
    model_dir: Union[str, Path] = ".",
) -> Any:
    """
    Load the trained classifier for inference.
    - If config has classifier_type == "pytorch" and model.pt exists, load PyTorch MLP and return wrapper.
    - Otherwise load model.pkl (legacy RandomForest) with joblib.
    Returns an object with predict_proba(X) and predict(X) (numpy in/out).
    """
    import joblib

    config_path = Path(config_path)
    model_dir = Path(model_dir)
    if not config_path.exists():
        pkl_path = model_dir / "model.pkl"
        if pkl_path.exists():
            return joblib.load(pkl_path)
        raise FileNotFoundError("No model_config.pkl and no model.pkl. Run training first.")

    config = joblib.load(config_path)
    if config.get("classifier_type") == "pytorch":
        pt_path = model_dir / "model.pt"
        if not pt_path.exists():
            raise FileNotFoundError("config says classifier_type=pytorch but model.pt not found. Run training.")
        device = get_device()
        model = load_mlp(pt_path, device=device)
        return PyTorchClassifierWrapper(model, device)

    pkl_path = model_dir / "model.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError("No model.pkl. Run training first.")
    return joblib.load(pkl_path)
