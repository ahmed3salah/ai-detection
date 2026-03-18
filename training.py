"""
Train the AI-detection classifier.
Uses local data in dataset/ by default (human + AI text or data.jsonl).
Use --idmgsp only if you have datasets<3.0 and want the IDMGSP benchmark.
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL", category=UserWarning)
warnings.filterwarnings("ignore", message="trust_remote_code", category=UserWarning)

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from features import (
    extract_features,
    extract_features_with_perplexity,
    extract_features_with_dual_perplexity,
    FEATURE_NAMES_WITH_PPL,
    FEATURE_NAMES_DUAL_PPL,
)
from data_loader import load_idmgsp, load_local_dataset


def train(
    use_idmgsp: bool = False,
    idmgsp_subset: str = "classifier_input",
    idmgsp_text_mode: str = "abstract",
    dataset_dir: str = "dataset",
    val_fraction: float = 0.2,
    random_state: int = 42,
    use_perplexity_features: bool = True,
    use_dual_perplexity: bool = False,
    use_tfidf: bool = False,
    max_tfidf_features: int = 5000,
    arxiv_metadata_max_samples: int = 100_000,
):
    """
    Load data (local dataset/ by default, or IDMGSP with --idmgsp), extract features,
    train classifier, save model and config.
    """
    X_train, y_train, X_val, y_val = [], [], [], []
    if use_idmgsp:
        try:
            out = load_idmgsp(
                subset=idmgsp_subset,
                text_mode=idmgsp_text_mode,
                val_size=val_fraction,
                random_state=random_state,
            )
            if len(out) == 4:
                X_train, y_train, X_val, y_val = out
            else:
                X_train, y_train = out
        except Exception as e:
            raise RuntimeError(
                "IDMGSP failed. Use local data in dataset/ (see DATASET.md). Error: %s" % e
            ) from e
    if not X_train:
        out = load_local_dataset(
            dataset_dir=dataset_dir,
            val_size=val_fraction,
            random_state=random_state,
            arxiv_metadata_max_samples=arxiv_metadata_max_samples,
        )
        if len(out) == 4:
            X_train, y_train, X_val, y_val = out
        else:
            X_train, y_train = out

    if not X_train:
        raise ValueError(
            "No training data in %s. Add human_text.txt + ai_text.txt (or data.jsonl). See DATASET.md." % dataset_dir
        )

    if use_dual_perplexity and not use_perplexity_features:
        raise ValueError("use_dual_perplexity requires use_perplexity_features=True")

    if use_dual_perplexity:
        extract_fn = extract_features_with_dual_perplexity
        feature_names = FEATURE_NAMES_DUAL_PPL
    elif use_perplexity_features:
        extract_fn = extract_features_with_perplexity
        feature_names = FEATURE_NAMES_WITH_PPL
    else:
        extract_fn = extract_features
        feature_names = getattr(extract_features, "feature_names")

    # Dense features
    X_train_dense = np.array([extract_fn(t) for t in X_train], dtype=np.float64)
    if X_val:
        X_val_dense = np.array([extract_fn(t) for t in X_val], dtype=np.float64)

    # Optional tf-idf
    vectorizer = None
    if use_tfidf:
        vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_train_feat = np.hstack([X_train_dense, X_train_tfidf.toarray()])
        if X_val:
            X_val_tfidf = vectorizer.transform(X_val)
            X_val_feat = np.hstack([X_val_dense, X_val_tfidf.toarray()])
        else:
            X_val_feat = None
    else:
        X_train_feat = X_train_dense
        X_val_feat = X_val_dense if X_val else None

    y_train_arr = np.array(y_train)
    # n_jobs=-1 uses all CPU cores; classifier runs on CPU (sklearn has no CUDA support)
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    model.fit(X_train_feat, y_train_arr)

    # Save model and config
    Path("model.pkl").parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "model.pkl")

    config = {
        "feature_names": list(feature_names),
        "use_perplexity_features": use_perplexity_features,
        "use_dual_perplexity": use_dual_perplexity,
        "use_tfidf": use_tfidf,
        "max_tfidf_features": max_tfidf_features if use_tfidf else None,
        "val_fraction": val_fraction,
        "random_state": random_state,
        "idmgsp": use_idmgsp,
        "idmgsp_subset": idmgsp_subset if use_idmgsp else None,
        "idmgsp_text_mode": idmgsp_text_mode if use_idmgsp else None,
    }
    joblib.dump(config, "model_config.pkl")

    if vectorizer is not None:
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    if X_val and X_val_feat is not None:
        val_acc = model.score(X_val_feat, y_val)
        print(f"Validation accuracy: {val_acc:.4f}")

    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train AI detection model (from dataset/ by default)")
    p.add_argument("--idmgsp", action="store_true", help="Use IDMGSP benchmark (requires datasets<3.0)")
    p.add_argument("--dataset-dir", default="dataset", help="Directory for local data (default: dataset)")
    p.add_argument("--idmgsp-subset", default="classifier_input", help="IDMGSP subset name")
    p.add_argument("--idmgsp-text", default="abstract", choices=("abstract", "full"),
                   help="Use abstract only or abstract+intro+conclusion")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (0=no split)")
    p.add_argument("--no-perplexity", action="store_true", help="Disable perplexity as a feature (faster training)")
    p.add_argument("--dual-perplexity", action="store_true",
                    help="Use both GPT-2 and SciBERT perplexity (2 GPUs when available)")
    p.add_argument("--tfidf", action="store_true", help="Add tf-idf (1-2 gram) features")
    p.add_argument("--max-tfidf", type=int, default=5000, help="Max tf-idf features (default 5000)")
    p.add_argument("--arxiv-max-samples", type=int, default=100_000,
                   help="Max samples from arxiv-metadata-pre-llm.jsonl (default 100000). 0 = skip.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(
        use_idmgsp=args.idmgsp,
        idmgsp_subset=args.idmgsp_subset,
        idmgsp_text_mode=args.idmgsp_text,
        dataset_dir=args.dataset_dir,
        val_fraction=args.val_fraction,
        random_state=args.seed,
        use_perplexity_features=not args.no_perplexity,
        use_dual_perplexity=args.dual_perplexity,
        use_tfidf=args.tfidf,
        max_tfidf_features=args.max_tfidf,
        arxiv_metadata_max_samples=args.arxiv_max_samples,
    )
