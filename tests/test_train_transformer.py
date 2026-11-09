"""The transformer candidate: trains, keeps its best epoch, calibrates, and reads back like the others."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate_segments as ev
import segment_dataset as ds
import train_segments as ts
from segment_model import SegmentModel
from tests.test_train_segments import AI_SENTENCES, HUMAN_SENTENCES, make_rows, paragraph, write_rows

import random

TOKENIZER = "allenai/scibert_scivocab_uncased"


def tiny_base_model(directory: Path) -> str:
    """A two-layer BERT with random weights: enough to exercise the trainer, nothing to download."""
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_labels=2)
    BertForSequenceClassification(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


def tokenizer_cached() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        return True
    except Exception:
        return False


@unittest.skipUnless(tokenizer_cached(), f"needs the {TOKENIZER} tokenizer in the Hugging Face cache")
class TransformerTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import train_transformer as tt

        cls.tt = tt
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        cls.base = tiny_base_model(root / "base")
        cls.data = write_rows(root, make_rows(papers=120))
        cls.out = str(root / "artifacts")
        cls.artifacts = tt.train(cls.data, out_root=cls.out, version="tiny", base_model=cls.base, epochs=3, batch_size=16, learning_rate=1e-3, max_length=96, device="cpu")

    def test_writes_a_model_that_says_what_it_is(self):
        self.assertEqual(sorted(p.name for p in self.artifacts.iterdir()), ["calibrator.pkl", "model", "model_config.pkl"])
        config = SegmentModel(self.artifacts, device="cpu").config
        self.assertEqual(config["classifier_type"], "transformer")
        self.assertEqual(config["base_model"], self.base)
        self.assertEqual(config["text_normalization"], "v2")
        self.assertIn(config["best_epoch"], (1, 2, 3))
        self.assertEqual(config["dataset_sha256"], ds.load_segments(self.data)[1])

    def test_learns_the_toy_task_and_reads_back(self):
        model = SegmentModel(self.artifacts, device="cpu")
        rng = random.Random(8)
        probabilities, kept = model.predict([paragraph(HUMAN_SENTENCES, 100, rng), paragraph(AI_SENTENCES, 100, rng), "[1] [2]"])
        self.assertEqual(kept, [0, 1])
        self.assertLess(probabilities[0], probabilities[1])

    def test_evaluates_like_the_feature_models(self):
        report = ev.evaluate(str(self.artifacts), self.data, latency=True)
        self.assertGreater(report["metrics"]["auroc_test"], 0.9)
        self.assertIsNotNone(report["gates"]["cpu_latency"]["measured"]["seconds"])
        self.assertIsNotNone(report["gates"]["no_parens_digits_shortcut"]["measured"]["drop"])

    def test_the_api_serves_it_calibrated(self):
        from fastapi.testclient import TestClient

        import app as app_module

        rng = random.Random(9)
        segments = [{"id": "h", "text": paragraph(HUMAN_SENTENCES, 100, rng)}, {"id": "a", "text": paragraph(AI_SENTENCES, 100, rng)}]
        with patch.object(app_module, "SEGMENT_MODEL_DIR", self.artifacts), patch.object(app_module, "_segment_model", None), patch.object(
            app_module, "_get_classifier", side_effect=AssertionError("v1 classifier used")
        ):
            client = TestClient(app_module.app)
            body = client.post("/v2/score-segments", json={"segments": segments}).json()
            info = client.get("/v2/model-info").json()
            explained = client.post("/v2/explain-segments", json={"segments": segments}).json()

        self.assertEqual(body["model_version"], "tiny")
        self.assertIs(body["calibrated"], True)
        self.assertEqual(body["thresholds"], SegmentModel(self.artifacts, device="cpu").thresholds)
        by_id = {s["id"]: s for s in body["segments"]}
        self.assertTrue(by_id["h"]["scoring_ok"] and by_id["a"]["scoring_ok"])
        self.assertLess(by_id["h"]["ai_probability"], by_id["a"]["ai_probability"])
        self.assertEqual(info["model_type"], "transformer")
        self.assertEqual([s["id"] for s in explained["segments"]], ["h", "a"])
        self.assertTrue(all(s["explain_ok"] for s in explained["segments"]))
        for result, segment in zip(explained["segments"], segments):
            for span in result["evidence"]:
                self.assertTrue(0 <= span["start"] < span["end"] <= len(segment["text"]))

    def test_refuses_to_overwrite_and_leaves_nothing_after_a_crash(self):
        with self.assertRaisesRegex(ts.TrainingAborted, "never overwritten"):
            self.tt.train(self.data, out_root=self.out, version="tiny", base_model=self.base, epochs=1, device="cpu")
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=RuntimeError("out of memory")), self.assertRaises(RuntimeError):
            self.tt.train(self.data, out_root=self.out, version="crash", base_model=self.base, epochs=1, device="cpu", max_length=64)
        self.assertEqual(sorted(p.name for p in Path(self.out).iterdir()), ["tiny"])


if __name__ == "__main__":
    unittest.main()
