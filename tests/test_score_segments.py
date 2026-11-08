"""/v2/score-segments: the HTTP contract the backend relies on, and batch/single feature parity."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

import app as app_module
import segment_scoring

PARAGRAPH = (
    "We evaluate the proposed method on three public corpora and report accuracy for each one. "
    "The results show a consistent improvement over the baseline, although the margin narrows "
    "on the smallest corpus, where only a few hundred labelled examples are available."
)


def fake_classifier(probabilities):
    clf = MagicMock()
    clf.predict_proba.side_effect = lambda X: np.column_stack(
        [1 - np.array(probabilities[: len(X)]), np.array(probabilities[: len(X)])]
    )
    return clf


def fake_batch(texts, _config):
    return np.zeros((len(texts), 3)), list(range(len(texts)))


class ScoreSegmentsApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.config = patch.object(app_module, "_get_config", return_value={})
        self.config.start()
        self.addCleanup(self.config.stop)
        patcher = patch.object(app_module, "_model_version", "test-model")
        patcher.start()
        self.addCleanup(patcher.stop)
        # These tests drive the v1 fallback; ParagraphModelApiTest covers a deployed paragraph model.
        paragraph = patch.object(app_module, "_get_segment_model", return_value=None)
        paragraph.start()
        self.addCleanup(paragraph.stop)

    def post(self, segments, **extra):
        return self.client.post("/v2/score-segments", json={"segments": segments, **extra})

    def segments(self, count=3):
        return [{"id": 10 + i, "text": f"{i} {PARAGRAPH}", "section": "Intro"} for i in range(count)]

    def test_scores_each_segment_with_its_band_and_keeps_ids_and_order(self):
        with patch.object(app_module, "_get_classifier", return_value=fake_classifier([0.95, 0.7, 0.1])), patch.object(
            app_module, "_build_feature_batch", side_effect=fake_batch
        ):
            response = self.post(self.segments(), document_id="doc-1", text_type="full_paper")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_version"], "test-model")
        self.assertIs(body["calibrated"], False)
        self.assertEqual(body["thresholds"], {"likely_ai": 0.85, "possibly_ai": 0.6})
        self.assertEqual([s["id"] for s in body["segments"]], [10, 11, 12])
        self.assertEqual([s["band"] for s in body["segments"]], ["likely_ai", "possibly_ai", "human"])
        self.assertEqual(body["segments"][0]["ai_probability"], 0.95)
        self.assertTrue(all(s["scoring_ok"] for s in body["segments"]))

    def test_uses_a_models_calibrated_thresholds(self):
        self.config.stop()
        config = {"segment_thresholds": {"likely_ai": 0.9, "possibly_ai": 0.4}, "model_version": "x"}
        with patch.object(app_module, "_get_config", return_value=config), patch.object(
            app_module, "_get_classifier", return_value=fake_classifier([0.5])
        ), patch.object(app_module, "_build_feature_batch", side_effect=fake_batch):
            body = self.post(self.segments(1)).json()
        self.config.start()

        self.assertIs(body["calibrated"], True)
        self.assertEqual(body["segments"][0]["band"], "possibly_ai")

    def test_a_segment_too_short_or_too_long_fails_alone(self):
        segments = [
            {"id": "short", "text": "Too short."},
            {"id": "long", "text": "x" * (segment_scoring.MAX_SEGMENT_CHARS + 1)},
            {"id": "ok", "text": PARAGRAPH},
        ]
        with patch.object(app_module, "_get_classifier", return_value=fake_classifier([0.2])), patch.object(
            app_module, "_build_feature_batch", side_effect=fake_batch
        ):
            body = self.post(segments).json()

        by_id = {s["id"]: s for s in body["segments"]}
        self.assertEqual(by_id["short"]["error_class"], "too_short")
        self.assertIsNone(by_id["short"]["ai_probability"])
        self.assertIs(by_id["long"]["scoring_ok"], False)
        self.assertEqual(by_id["long"]["error_class"], "too_long")
        self.assertEqual(by_id["ok"]["ai_probability"], 0.2)

    def test_a_segment_whose_features_failed_has_no_probability_not_zero(self):
        def only_second(texts, _config):
            return np.zeros((1, 3)), [1]

        with patch.object(app_module, "_get_classifier", return_value=fake_classifier([0.3])), patch.object(
            app_module, "_build_feature_batch", side_effect=only_second
        ):
            body = self.post(self.segments(2)).json()

        self.assertIs(body["segments"][0]["scoring_ok"], False)
        self.assertIsNone(body["segments"][0]["ai_probability"])
        self.assertEqual(body["segments"][1]["ai_probability"], 0.3)

    def test_a_non_finite_probability_is_a_failure_and_the_answer_is_valid_json(self):
        with patch.object(app_module, "_get_classifier", return_value=fake_classifier([float("nan")])), patch.object(
            app_module, "_build_feature_batch", side_effect=fake_batch
        ):
            response = self.post(self.segments(1))

        json.loads(response.text)
        self.assertIs(response.json()["segments"][0]["scoring_ok"], False)

    def test_a_scoring_crash_fails_the_segments_but_still_answers(self):
        clf = MagicMock()
        clf.predict_proba.side_effect = RuntimeError("boom")
        with patch.object(app_module, "_get_classifier", return_value=clf), patch.object(
            app_module, "_build_feature_batch", side_effect=fake_batch
        ):
            response = self.post(self.segments(2))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(s["scoring_ok"] is False for s in response.json()["segments"]))

    def test_refuses_too_many_segments_and_too_much_text(self):
        with patch.object(app_module, "_get_classifier", return_value=MagicMock()) as classifier:
            many = self.post(self.segments(segment_scoring.MAX_SEGMENTS + 1))
            with patch.object(segment_scoring, "MAX_TOTAL_CHARS", 100):
                much = self.post(self.segments(2))

        self.assertEqual(many.status_code, 413)
        self.assertEqual(much.status_code, 413)
        classifier.assert_not_called()

    def test_validates_the_request(self):
        self.assertEqual(self.post([]).status_code, 422)
        self.assertEqual(self.post(self.segments(1), text_type="poem").status_code, 422)
        duplicate = [{"id": 1, "text": PARAGRAPH}, {"id": 1, "text": PARAGRAPH}]
        self.assertEqual(self.post(duplicate).status_code, 422)

    def test_accepts_exactly_what_the_backend_sends(self):
        """Modules/Core ScoreAiDetectionSegmentsJob: integer ids, null sections, uuid document id."""
        segments = [{"id": 7, "text": PARAGRAPH, "section": None}]
        with patch.object(app_module, "_get_classifier", return_value=fake_classifier([0.2])), patch.object(
            app_module, "_build_feature_batch", side_effect=fake_batch
        ):
            response = self.post(segments, text_type="full_paper", document_id="9f2c6a1e-0000-4000-8000-000000000000")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["segments"][0]["id"], 7)

    def test_requires_the_api_key_when_one_is_set(self):
        with patch.dict(os.environ, {"DETECTION_API_KEY": "secret"}):
            self.assertEqual(self.post(self.segments(1)).status_code, 401)
            self.assertEqual(self.client.get("/v2/model-info").status_code, 401)
            with patch.object(app_module, "_get_classifier", return_value=fake_classifier([0.2])), patch.object(
                app_module, "_build_feature_batch", side_effect=fake_batch
            ):
                response = self.client.post(
                    "/v2/score-segments",
                    json={"segments": self.segments(1)},
                    headers={"Authorization": "Bearer secret"},
                )
        self.assertEqual(response.status_code, 200)

    def test_answers_503_when_the_model_is_missing(self):
        from fastapi import HTTPException

        with patch.object(app_module, "_get_classifier", side_effect=HTTPException(status_code=503, detail="not trained")):
            self.assertEqual(self.post(self.segments(1)).status_code, 503)

    def test_answers_503_while_another_inference_holds_the_slot(self):
        release = threading.Event()
        holding = threading.Event()

        def hold():
            with app_module._inference_slot():
                holding.set()
                release.wait(5)

        thread = threading.Thread(target=hold)
        thread.start()
        holding.wait(5)
        try:
            with patch.object(app_module, "_get_classifier", return_value=MagicMock()), patch.object(
                app_module, "_INFERENCE_WAIT_SECONDS", 0.01
            ):
                busy = self.post(self.segments(1))
                health = self.client.get("/health")
        finally:
            release.set()
            thread.join()

        self.assertEqual(busy.status_code, 503)
        self.assertEqual(health.status_code, 200)

    def test_model_info_reports_version_thresholds_and_limits(self):
        with patch.object(app_module, "_get_classifier", return_value=MagicMock()):
            body = self.client.get("/v2/model-info").json()

        self.assertEqual(body["model_version"], "test-model")
        self.assertIs(body["calibrated"], False)
        self.assertEqual(body["limits"]["max_segments"], segment_scoring.MAX_SEGMENTS)

    def test_never_logs_manuscript_text_when_scoring_fails(self):
        clf = MagicMock()
        clf.predict_proba.side_effect = RuntimeError("boom")
        with patch.object(app_module, "_get_classifier", return_value=clf), patch.object(
            app_module, "_build_feature_batch", side_effect=fake_batch
        ), self.assertLogs("segment_scoring", level="ERROR") as logs:
            self.post(self.segments(1))

        self.assertNotIn("corpora", "\n".join(logs.output))


class FakeParagraphModel:
    config = {"model_version": "scibert-test", "classifier_type": "transformer", "segment_thresholds": {"likely_ai": 0.8, "possibly_ai": 0.1}}

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, texts):
        return np.array(self.probabilities[: len(texts)]), list(range(len(texts)))


class ParagraphModelApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        # A deployed paragraph model replaces the v1 classifier for v2 entirely.
        v1 = patch.object(app_module, "_get_classifier", side_effect=AssertionError("v1 classifier used"))
        v1.start()
        self.addCleanup(v1.stop)

    def post(self, count):
        return self.client.post("/v2/score-segments", json={"segments": [{"id": i, "text": f"{i} {PARAGRAPH}"} for i in range(count)]})

    def test_serves_the_paragraph_model_with_its_own_version_and_calibrated_bands(self):
        with patch.object(app_module, "_get_segment_model", return_value=FakeParagraphModel([0.9, 0.5, 0.05])):
            body = self.post(3).json()

        self.assertEqual(body["model_version"], "scibert-test")
        self.assertIs(body["calibrated"], True)
        self.assertEqual(body["thresholds"], {"likely_ai": 0.8, "possibly_ai": 0.1})
        self.assertEqual([s["band"] for s in body["segments"]], ["likely_ai", "possibly_ai", "human"])

    def test_model_info_describes_the_paragraph_model(self):
        with patch.object(app_module, "_get_segment_model", return_value=FakeParagraphModel([])):
            body = self.client.get("/v2/model-info").json()

        self.assertEqual(body["model_version"], "scibert-test")
        self.assertEqual(body["model_type"], "transformer")
        self.assertIs(body["calibrated"], True)
        self.assertIs(body["uses_perplexity"], False)

    def test_a_deployed_model_that_will_not_load_is_a_503_never_a_fallback_to_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model_config.pkl").write_text("not a pickle")
            with patch.object(app_module, "SEGMENT_MODEL_DIR", Path(directory)), patch.object(
                app_module, "_segment_model", None
            ), self.assertLogs("app", level="ERROR"):
                response = self.post(1)
                ready = self.client.get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(ready.status_code, 503)

    def test_without_a_deployed_model_v2_falls_back_to_v1(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app_module, "SEGMENT_MODEL_DIR", Path(directory)), patch.object(
            app_module, "_segment_model", None
        ):
            self.assertIsNone(app_module._get_segment_model())


class DockerImageTest(unittest.TestCase):
    def test_the_image_copies_every_local_module_the_app_imports(self):
        """The Dockerfile names its modules; one left out crashes the container on boot."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        local = {path.stem for path in root.glob("*.py")}
        copied = set()
        for line in (root / "Dockerfile").read_text().splitlines():
            if line.startswith("COPY "):
                copied.update(Path(part).stem for part in line.split()[1:-1] if part.endswith(".py"))

        pending = ["app"]
        seen = set()
        while pending:
            module = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            for node in ast.walk(ast.parse((root / f"{module}.py").read_text())):
                if isinstance(node, ast.Import):
                    pending.extend(alias.name for alias in node.names if alias.name in local)
                elif isinstance(node, ast.ImportFrom) and node.module in local:
                    pending.append(node.module)

        self.assertEqual(seen - copied, set())


class DetectStillWorksTest(unittest.TestCase):
    def test_v1_detect_answers_off_the_event_loop_with_the_same_shape(self):
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.7, 0.3]])
        client = TestClient(app_module.app)
        with patch.object(app_module, "_get_classifier", return_value=clf), patch.object(
            app_module, "_get_config", return_value={}
        ), patch.object(app_module, "_build_features", return_value=np.zeros((1, 3))), patch(
            "detector.calculate_perplexity", return_value=float("inf")
        ):
            response = client.post("/detect", json={"text": PARAGRAPH, "text_type": "abstract"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertAlmostEqual(body["classifier_overall"], 0.3, places=5)
        self.assertEqual(body["paragraph_count"], len(body["paragraphs"]))


class BatchedFeatureFailureTest(unittest.TestCase):
    """A text that can't be measured is None, never a row of zeros the model would score."""

    def test_a_failed_perplexity_drops_only_that_text(self):
        from features import extract_feature_rows

        with patch("features._style_feature_rows", return_value=[[1.0], [2.0]]), patch(
            "detector.batch_perplexity", side_effect=[[None, 5.0], [7.0, 9.0]]
        ):
            rows = extract_feature_rows(["a", "b"], dual_perplexity=True)

        self.assertIsNone(rows[0])
        self.assertEqual(len(rows[1]), 5)

    def test_build_feature_batch_reports_which_texts_it_kept(self):
        with patch.object(app_module, "extract_feature_rows", return_value=[None, [1.0, 2.0], [float("inf"), 1.0]]), patch.object(
            app_module, "_get_feature_scaler", return_value=None
        ):
            matrix, kept = app_module._build_feature_batch(["a", "b", "c"], {})

        self.assertEqual(kept, [1])
        self.assertEqual(matrix.shape, (1, 2))


class PerplexityOnlyWhenTheModelLearnedFromItTest(unittest.TestCase):
    """The deployed model was trained on zeros there; real values would give meaningless scores."""

    @staticmethod
    def scaler(perplexity_variance):
        scaler = MagicMock()
        scaler.var_ = np.array([1.0] * 19 + [perplexity_variance] * 4)
        return scaler

    def test_a_scaler_with_no_variance_means_the_model_never_saw_perplexity(self):
        with patch.object(app_module, "_get_feature_scaler", return_value=self.scaler(0.0)):
            self.assertFalse(app_module._model_learned_from_perplexity({"use_dual_perplexity": True}))

    def test_a_scaler_with_variance_means_it_did(self):
        with patch.object(app_module, "_get_feature_scaler", return_value=self.scaler(3.5)):
            self.assertTrue(app_module._model_learned_from_perplexity({"use_dual_perplexity": True}))

    def test_zeros_fill_the_slots_without_running_any_language_model(self):
        from features import extract_feature_rows

        with patch("features._style_feature_rows", return_value=[[1.0], None]), patch(
            "detector.batch_perplexity"
        ) as measure:
            rows = extract_feature_rows(["a", "b"], dual_perplexity=True, measure_perplexity=False)

        measure.assert_not_called()
        self.assertEqual(rows, [[1.0, 0.0, 0.0, 0.0, 0.0], None])


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "needs the real models; set RUN_MODEL_TESTS=1")
class RealModelParityTest(unittest.TestCase):
    """The deployed MLP only means anything if batching reproduces the features it was trained on."""

    TEXTS = [
        PARAGRAPH,
        "Short but valid sentence for the model to read here.",
        " ".join([PARAGRAPH] * 12),  # spans more than one SciBERT window
        "Transfer learning reduces the labelled data a classifier needs. " * 5,
        "x",  # one token: no perplexity at all on the GPT-2 side
        "   ",
    ]

    def test_the_deployed_model_is_scored_on_zeros_for_perplexity(self):
        config = app_module._get_config()
        self.assertFalse(app_module._model_learned_from_perplexity(config))

    def test_batched_gpt2_perplexity_matches_one_at_a_time(self):
        from detector import batch_perplexity, calculate_perplexity

        batched = batch_perplexity(self.TEXTS, model_type="gpt2", batch_size=4)
        for text, value in zip(self.TEXTS, batched):
            single = calculate_perplexity(text, model_type="gpt2")
            if np.isfinite(single):
                self.assertTrue(np.isclose(value, single, rtol=1e-3), f"{value} != {single}")
            else:
                self.assertFalse(np.isfinite(value))

    def test_batched_scibert_perplexity_matches_a_direct_forward_pass(self):
        """_perplexity_scibert itself raises on this transformers version, so compare with the model."""
        import torch
        from detector import _get_scibert, batch_perplexity

        model, tokenizer = _get_scibert()
        texts = self.TEXTS[:2] + [self.TEXTS[3]]
        batched = batch_perplexity(texts, model_type="scibert", batch_size=4)
        for text, value in zip(texts, batched):
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            labels = encoded["input_ids"].clone()
            labels[0, 0] = -100
            labels[0, -1] = -100
            with torch.no_grad():
                loss = model(**encoded, labels=labels).loss
            self.assertTrue(np.isclose(value, torch.exp(loss).item(), rtol=1e-3), f"{value} != {torch.exp(loss).item()}")

    def test_batched_features_give_the_probability_v1_gives(self):
        """Same features as the pipeline the model was trained and is served with, and same score."""
        config = app_module._get_config()
        classifier = app_module._get_classifier()
        texts = [t for t in self.TEXTS if len(t.strip()) >= segment_scoring.MIN_SEGMENT_CHARS]

        matrix, kept = app_module._build_feature_batch(texts, config)
        self.assertEqual(kept, list(range(len(texts))))

        single = np.vstack([app_module._build_features(t, config) for t in texts])
        np.testing.assert_allclose(matrix, single, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            classifier.predict_proba(matrix)[:, 1],
            classifier.predict_proba(single)[:, 1],
            atol=1e-6,
        )

    def test_real_request_through_the_endpoint(self):
        client = TestClient(app_module.app)
        response = client.post(
            "/v2/score-segments",
            json={"segments": [{"id": i, "text": t} for i, t in enumerate(self.TEXTS[:4])], "document_id": "d"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["model_version"].startswith("uncalibrated-"))
        self.assertTrue(all(s["scoring_ok"] for s in body["segments"]))


if __name__ == "__main__":
    unittest.main()
