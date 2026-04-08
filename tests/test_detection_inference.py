"""Unit tests for detection inference hardening (run_detection + feature dimension helper)."""
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


class TestRunDetection(unittest.TestCase):
    def test_happy_path_includes_aggregate_fields(self):
        from detector import run_detection

        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.7, 0.3]])

        def build_features(_t):
            return np.zeros((1, 8), dtype=np.float64)

        r = run_detection(
            "This is a sample paragraph with enough characters for detection.",
            clf,
            build_features,
            use_perplexity=False,
        )
        self.assertEqual(r["classifier_segments_failed"], 0)
        self.assertIn("classifier_overall", r)
        self.assertIsNone(r["perplexity_ai_signal"])
        self.assertTrue(r["paragraphs"][0]["scoring_ok"])
        self.assertAlmostEqual(r["classifier_overall"], 0.3, places=5)
        self.assertAlmostEqual(r["raw_overall_prob"], 0.3, places=4)

    def test_classifier_failure_sets_scoring_ok_and_count(self):
        from detector import run_detection

        clf = MagicMock()
        clf.predict_proba.side_effect = ValueError("simulated failure")

        def build_features(_t):
            return np.zeros((1, 8), dtype=np.float64)

        r = run_detection(
            "Another sample paragraph with sufficient length for the test.",
            clf,
            build_features,
            use_perplexity=False,
        )
        self.assertEqual(r["classifier_segments_failed"], 1)
        self.assertFalse(r["paragraphs"][0]["scoring_ok"])
        self.assertEqual(r["paragraphs"][0]["ai_probability"], 0.0)

    def test_empty_text_response_keys(self):
        from detector import run_detection

        clf = MagicMock()

        r = run_detection("   ", clf, lambda _t: np.zeros((1, 1)), use_perplexity=False)
        self.assertEqual(r["paragraph_count"], 0)
        self.assertEqual(r["classifier_segments_failed"], 0)
        self.assertEqual(r["classifier_overall"], 0.0)
        self.assertIsNone(r["perplexity_ai_signal"])

    def test_debug_flag_adds_error_class(self):
        import detector as det

        clf = MagicMock()
        clf.predict_proba.side_effect = RuntimeError("boom")

        def build_features(_t):
            return np.zeros((1, 4), dtype=np.float64)

        prev = det._DEBUG_SEGMENTS
        det._DEBUG_SEGMENTS = True
        try:
            r = det.run_detection(
                "Debug test paragraph with enough characters included.",
                clf,
                build_features,
                use_perplexity=False,
            )
            self.assertEqual(r["paragraphs"][0].get("error_class"), "RuntimeError")
        finally:
            det._DEBUG_SEGMENTS = prev

    def test_infinite_perplexity_json_serializable(self):
        """Perplexity can be inf for empty/degenerate text; API response must be JSON-safe."""
        from detector import run_detection

        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.5, 0.5]])

        def build_features(_t):
            return np.zeros((1, 4), dtype=np.float64)

        with patch("detector.calculate_perplexity", return_value=float("inf")):
            r = run_detection(
                "Some text here with enough length for one paragraph segment.",
                clf,
                build_features,
                use_perplexity=True,
            )
        json.dumps(r)
        self.assertIsNone(r["perplexity"])
        self.assertIsNone(r["perplexity_ai_signal"])

    def test_nan_classifier_prob_clamped_for_json(self):
        from detector import run_detection

        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[float("nan"), float("nan")]])

        r = run_detection(
            "Nan prob test paragraph with sufficient characters in the body.",
            clf,
            lambda _t: np.zeros((1, 3), dtype=np.float64),
            use_perplexity=False,
        )
        json.dumps(r)
        self.assertEqual(r["paragraphs"][0]["ai_probability"], 0.0)


class TestPytorchWeightsCorruptedError(unittest.TestCase):
    def test_skips_non_pytorch_classifier(self):
        import app as app_module

        self.assertIsNone(app_module._pytorch_weights_corrupted_error(MagicMock()))

    def test_detects_nan_weights(self):
        import app as app_module
        from classifier_mlp import MLPClassifier, PyTorchClassifierWrapper

        model = MLPClassifier(input_size=5, hidden_size=8)
        with torch.no_grad():
            model.net[0].weight.fill_(float("nan"))
        clf = PyTorchClassifierWrapper(model, torch.device("cpu"))
        msg = app_module._pytorch_weights_corrupted_error(clf)
        self.assertIsNotNone(msg)
        self.assertIn("NaN", msg or "")


class TestPytorchFeatureDimensionError(unittest.TestCase):
    def test_skips_non_pytorch_classifier(self):
        import app as app_module

        self.assertIsNone(app_module._pytorch_feature_dimension_error(MagicMock()))

    def test_mismatch_returns_message(self):
        import app as app_module
        from classifier_mlp import MLPClassifier, PyTorchClassifierWrapper

        model = MLPClassifier(input_size=99, hidden_size=8)
        clf = PyTorchClassifierWrapper(model, torch.device("cpu"))

        with patch.object(app_module, "_build_features", return_value=np.zeros((1, 3), dtype=np.float64)):
            msg = app_module._pytorch_feature_dimension_error(clf)
        self.assertIsNotNone(msg)
        self.assertIn("99", msg or "")
        self.assertIn("3", msg or "")

    def test_match_returns_none(self):
        import app as app_module
        from classifier_mlp import MLPClassifier, PyTorchClassifierWrapper

        model = MLPClassifier(input_size=5, hidden_size=8)
        clf = PyTorchClassifierWrapper(model, torch.device("cpu"))

        with patch.object(app_module, "_build_features", return_value=np.zeros((1, 5), dtype=np.float64)):
            self.assertIsNone(app_module._pytorch_feature_dimension_error(clf))


class TestSklearnFeatureDimensionError(unittest.TestCase):
    def test_skips_without_n_features_in_(self):
        import app as app_module

        self.assertIsNone(app_module._sklearn_feature_dimension_error(MagicMock()))

    def test_mismatch_returns_message(self):
        import app as app_module

        class _FakeEst:
            n_features_in_ = 17

        with patch.object(app_module, "_build_features", return_value=np.zeros((1, 21), dtype=np.float64)):
            msg = app_module._sklearn_feature_dimension_error(_FakeEst())
        self.assertIsNotNone(msg)
        self.assertIn("17", msg or "")
        self.assertIn("21", msg or "")

    def test_match_returns_none(self):
        import app as app_module

        class _FakeEst:
            n_features_in_ = 4

        with patch.object(app_module, "_build_features", return_value=np.zeros((1, 4), dtype=np.float64)):
            self.assertIsNone(app_module._sklearn_feature_dimension_error(_FakeEst()))


class TestFeatureDimensionError(unittest.TestCase):
    def test_combined_uses_pytorch_when_wrapper(self):
        import app as app_module
        from classifier_mlp import MLPClassifier, PyTorchClassifierWrapper

        model = MLPClassifier(input_size=2, hidden_size=3)
        clf = PyTorchClassifierWrapper(model, torch.device("cpu"))
        with patch.object(app_module, "_build_features", return_value=np.zeros((1, 9), dtype=np.float64)):
            msg = app_module._feature_dimension_error(clf)
        self.assertIsNotNone(msg)
        self.assertIn("2", msg or "")

    def test_combined_falls_back_to_sklearn(self):
        import app as app_module
        from classifier_mlp import PyTorchClassifierWrapper

        not_wrapper = MagicMock(spec=["predict_proba"])
        not_wrapper.n_features_in_ = 3
        type(not_wrapper).__name__ = "RandomForestClassifier"
        # Ensure not treated as PyTorch wrapper
        self.assertFalse(isinstance(not_wrapper, PyTorchClassifierWrapper))
        with patch.object(app_module, "_build_features", return_value=np.zeros((1, 99), dtype=np.float64)):
            msg = app_module._feature_dimension_error(not_wrapper)
        self.assertIsNotNone(msg)
        self.assertIn("3", msg or "")
        self.assertIn("99", msg or "")
