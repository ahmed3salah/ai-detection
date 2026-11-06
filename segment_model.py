"""
The feature sets a paragraph model can be trained on, and a trained model read back from its
artifacts directory.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np

from classifier_mlp import load_classifier
from features import extract_feature_rows, extract_features
from text_normalize import normalize_segment

Rows = List[Optional[List[float]]]

_RAW = list(extract_features.feature_names)

# Per-word or per-sentence rates only. The absolute counts (word_count, sentence_count) let a
# model learn length instead of writing, and the opener flags only mean something on a whole
# abstract.
STYLE_V2_COLUMNS = [
    "avg_sentence_length",
    "sentence_variance",
    "avg_word_length",
    "punctuation_per_word",
    "type_token_ratio",
    "passive_ratio",
    "hedging_ratio",
    "commas_per_sentence",
    "semicolons_per_sentence",
    "digit_ratio",
    "citation_like_per_word",
]


def _style_v2_rows(texts: Sequence[str]) -> Rows:
    rows: Rows = []
    for raw in extract_feature_rows(list(texts)):
        if raw is None:
            rows.append(None)
            continue
        by_name = dict(zip(_RAW, raw))
        words = by_name["word_count"]
        rows.append([
            by_name["avg_sentence_length"],
            by_name["sentence_variance"],
            by_name["avg_word_length"],
            by_name["punctuation_count"] / words if words else 0.0,
            by_name["type_token_ratio"],
            by_name["passive_ratio"],
            by_name["hedging_ratio"],
            by_name["commas_per_sentence"],
            by_name["semicolons_per_sentence"],
            by_name["digit_ratio"],
            by_name["citation_like"] / words if words else 0.0,
        ])
    return rows


def _legacy_zero_perplexity_rows(texts: Sequence[str]) -> Rows:
    """What the deployed model was trained on. Kept so tests can prove the trainer refuses it."""
    return extract_feature_rows(list(texts), dual_perplexity=True, measure_perplexity=False)


FEATURE_SETS: Dict[str, Tuple[Callable[[Sequence[str]], Rows], List[str]]] = {
    "style_v2": (_style_v2_rows, STYLE_V2_COLUMNS),
    "legacy_zero_perplexity": (
        _legacy_zero_perplexity_rows,
        _RAW + ["perplexity_gpt2", "log_perplexity_gpt2", "perplexity_scibert", "log_perplexity_scibert"],
    ),
}


def feature_rows(texts: Sequence[str], feature_set: str) -> Rows:
    return FEATURE_SETS[feature_set][0](texts)


def feature_names(feature_set: str) -> List[str]:
    return list(FEATURE_SETS[feature_set][1])


class SegmentModel:
    """A trained paragraph model, feature-based or transformer: texts in, calibrated probabilities out."""

    def __init__(self, directory: Union[str, Path], device: Optional[str] = None):
        self.directory = Path(directory)
        self.config = joblib.load(self.directory / "model_config.pkl")
        self.calibrator = joblib.load(self.directory / "calibrator.pkl")
        self.is_transformer = self.config.get("classifier_type") == "transformer"
        if self.is_transformer:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from train_transformer import pick_device

            self.device = pick_device(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory / "model")
            self.classifier = AutoModelForSequenceClassification.from_pretrained(self.directory / "model").to(self.device)
            self.scaler = None
            return
        self.classifier = load_classifier(self.directory / "model_config.pkl", self.directory)
        scaler_path = self.directory / "feature_scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    @property
    def thresholds(self) -> Dict[str, float]:
        return self.config["segment_thresholds"]

    def predict(self, texts: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
        """Calibrated P(AI) for the texts that could be measured, and each one's index into texts."""
        normalized = [normalize_segment(text) for text in texts]
        if self.is_transformer:
            from train_transformer import predict_raw

            kept = [i for i, text in enumerate(normalized) if text]
            if not kept:
                return np.zeros(0), []
            raw = predict_raw(self.classifier, self.tokenizer, [texts[i] for i in kept], self.config["max_length"], self.device)
            return self.calibrator.predict(raw), kept
        rows = feature_rows(normalized, self.config["feature_set"])
        kept = [i for i, row in enumerate(rows) if row is not None and normalized[i] and np.isfinite(row).all()]
        if not kept:
            return np.zeros(0), []
        matrix = np.array([rows[i] for i in kept], dtype=np.float64)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        return self.calibrator.predict(self.classifier.predict_proba(matrix)[:, 1]), kept
