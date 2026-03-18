"""
Evaluate trained model on IDMGSP val or test split.
Reports accuracy and F1 for comparison with published baselines.
"""
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

from features import (
    extract_features,
    extract_features_with_perplexity,
    extract_features_with_dual_perplexity,
)
from data_loader import load_idmgsp, load_idmgsp_test, load_local_dataset
from classifier_mlp import load_classifier


def _get_config():
    path = Path("model_config.pkl")
    if not path.exists():
        return {}
    return joblib.load(path)


def _get_vectorizer():
    path = Path("tfidf_vectorizer.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


def build_features(texts, config):
    """Same feature pipeline as app/training."""
    use_dual_ppl = config.get("use_dual_perplexity", False)
    use_ppl = config.get("use_perplexity_features", True)
    use_tfidf = config.get("use_tfidf", False)
    if use_dual_ppl:
        extract_fn = extract_features_with_dual_perplexity
    elif use_ppl:
        extract_fn = extract_features_with_perplexity
    else:
        extract_fn = extract_features

    dense = np.array([extract_fn(t) for t in texts], dtype=np.float64)
    if use_tfidf:
        vec = _get_vectorizer()
        if vec is not None:
            tfidf = vec.transform(texts)
            return np.hstack([dense, tfidf.toarray()])
    return dense


def evaluate(
    idmgsp_subset: str = "classifier_input",
    text_mode: str = "abstract",
    use_test_split: bool = True,
    val_fraction: float = 0.2,
    random_state: int = 42,
    dataset_dir: str = None,
):
    """
    Load model and evaluate. If dataset_dir is set, use local data (val split).
    Otherwise use IDMGSP: native test split if available, else val split from load_idmgsp.
    """
    config = _get_config()
    try:
        model = load_classifier(config_path="model_config.pkl", model_dir=".")
    except FileNotFoundError:
        raise FileNotFoundError("No model.pt or model.pkl. Run training first.")

    X_test, y_test = [], []

    if dataset_dir:
        out = load_local_dataset(dataset_dir=dataset_dir, val_size=val_fraction, random_state=random_state)
        if len(out) == 4:
            _, _, X_test, y_test = out
        else:
            X_test, y_test = out
    if not X_test and use_test_split:
        X_test, y_test = load_idmgsp_test(subset=idmgsp_subset, text_mode=text_mode)
    if not X_test:
        try:
            out = load_idmgsp(
                subset=idmgsp_subset,
                text_mode=text_mode,
                val_size=val_fraction,
                random_state=random_state,
            )
            if len(out) == 4:
                _, _, X_test, y_test = out
        except Exception:
            pass
    if not X_test:
        raise ValueError("No test/val data. Use --dataset-dir dataset (default) or add data in dataset/.")

    X_feat = build_features(X_test, config)
    y_pred = model.predict(X_feat)
    y_true = np.array(y_test)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"N:        {len(y_true)}")
    return {"accuracy": acc, "f1": f1, "n": len(y_true)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate model on IDMGSP or local data")
    p.add_argument("--dataset-dir", default="dataset", help="Local data directory (default: dataset)")
    p.add_argument("--idmgsp-subset", default="classifier_input", help="IDMGSP subset")
    p.add_argument("--idmgsp-text", default="abstract", choices=("abstract", "full"))
    p.add_argument("--val-only", action="store_true",
                   help="Use val split (same seed as training) instead of native test")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    evaluate(
        idmgsp_subset=args.idmgsp_subset,
        text_mode=args.idmgsp_text,
        use_test_split=not args.val_only,
        val_fraction=args.val_fraction,
        random_state=args.seed,
        dataset_dir=args.dataset_dir,
    )
