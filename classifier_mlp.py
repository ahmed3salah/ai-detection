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


def _cuda_kernels_work() -> bool:
    """Try a minimal CUDA op. Returns False if GPU is not supported (e.g. Blackwell sm_120 on older PyTorch)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(2, device="cuda").sum().item()
        return True
    except RuntimeError:
        return False


def get_device() -> torch.device:
    """Return best available device: cuda (if kernels work), then mps (Apple Silicon), then cpu."""
    if torch.cuda.is_available() and _cuda_kernels_work():
        return torch.device("cuda")
    if torch.cuda.is_available() and not _cuda_kernels_work():
        print("[Device] CUDA is available but kernels failed (e.g. Blackwell sm_120 on PyTorch < 2.7). Using CPU.")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_status(prefix: str = "Classifier") -> None:
    """Print PyTorch device availability and selected device for training/inference."""
    cuda_ok = torch.cuda.is_available()
    cuda_works = _cuda_kernels_work() if cuda_ok else False
    cuda_count = torch.cuda.device_count() if cuda_ok else 0
    mps = getattr(torch.backends, "mps", None)
    mps_ok = mps is not None and mps.is_available()
    device = get_device()
    print(f"[{prefix}] PyTorch CUDA available: {cuda_ok}  (kernels work: {cuda_works})")
    if cuda_ok:
        print(f"[{prefix}] CUDA device count: {cuda_count}")
        for i in range(cuda_count):
            print(f"[{prefix}]   cuda:{i} -> {torch.cuda.get_device_name(i)}")
    print(f"[{prefix}] PyTorch MPS (Apple Silicon) available: {mps_ok}")
    print(f"[{prefix}] Selected device: {device}")


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
    - Prefer model.pt when classifier_type is pytorch, or when only model.pt exists (current training).
    - Otherwise load model.pkl (legacy RandomForest) with joblib.
    Returns an object with predict_proba(X) and predict(X) (numpy in/out).
    """
    import joblib

    config_path = Path(config_path)
    model_dir = Path(model_dir)
    pt_path = model_dir / "model.pt"
    pkl_path = model_dir / "model.pkl"

    def _load_pt() -> PyTorchClassifierWrapper:
        device = get_device()
        model = load_mlp(pt_path, device=device)
        return PyTorchClassifierWrapper(model, device)

    if not config_path.exists():
        if pt_path.exists():
            return _load_pt()
        if pkl_path.exists():
            return joblib.load(pkl_path)
        raise FileNotFoundError(
            "No model_config.pkl and no model artifacts. Place model.pt (+ model_config.pkl) or model.pkl in %s."
            % model_dir.resolve()
        )

    config = joblib.load(config_path)
    use_pytorch = config.get("classifier_type") == "pytorch"
    # Current training only saves model.pt; old configs may omit classifier_type
    if not use_pytorch and pt_path.exists() and not pkl_path.exists():
        use_pytorch = True

    if use_pytorch:
        if not pt_path.exists():
            raise FileNotFoundError(
                "model_config.pkl expects PyTorch but model.pt not found in %s. Run training or copy model.pt."
                % model_dir.resolve()
            )
        return _load_pt()

    if pkl_path.exists():
        return joblib.load(pkl_path)
    if pt_path.exists():
        return _load_pt()
    raise FileNotFoundError("No model.pkl or model.pt in %s. Run training first." % model_dir.resolve())
