"""
The training pipeline's promises: it stops on bad features, keeps papers from crossing splits,
never touches the deployed model, and won't call a model ready on too little evidence.
"""
import hashlib
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import app as app_module
import evaluate_segments as ev
import segment_dataset as ds
import train_segments as ts
from segment_model import SegmentModel, feature_rows, feature_names
from text_normalize import normalize_segment

ROOT = Path(__file__).resolve().parent.parent

HUMAN_SENTENCES = [
    "We measured it twice, and the second run was worse!",
    "Nobody expected that; the old code had hidden a bug for years.",
    "Why did the loss spike? Our logs point to a bad batch.",
    "Three of the 12 runs crashed, so we threw them out.",
    "It works, mostly, but only on small graphs (under 40 nodes).",
    "That surprised us: 7 of 9 readers disagreed with the labels.",
    "The bad runs were removed by hand before anyone looked.",
]
AI_SENTENCES = [
    "The proposed framework demonstrates significant potential across a comprehensive range of applications.",
    "Furthermore, the methodology provides a robust foundation for subsequent investigations in this domain.",
    "These findings underscore the importance of integrating interdisciplinary perspectives into future research.",
    "Additionally, the analysis highlights several considerations that may inform practical implementation strategies.",
    "Moreover, the results suggest that the approach could facilitate substantial improvements in overall performance.",
    "Consequently, this work contributes valuable insights to the growing body of literature on the subject.",
    "A comprehensive evaluation was conducted to assess the overall effectiveness of the approach.",
]


def paragraph(sentences, words, rng):
    picked = []
    while sum(len(s.split()) for s in picked) < words:
        picked.append(rng.choice(sentences))
    return " ".join(picked)


def make_rows(papers=150, seed=1, generators=("gen-a", "gen-b")):
    rng = random.Random(seed)
    rows = []
    for p in range(papers):
        paper = f"paper-{p}"
        domain = "cs" if p % 2 else "physics"
        for k in range(3):
            words = rng.choice([60, 100, 150, 220])
            rows.append({"id": f"{paper}-h{k}", "text": paragraph(HUMAN_SENTENCES, words, rng), "label": 0, "paper_id": paper, "domain": domain, "section": "methods"})
            rows.append({"id": f"{paper}-a{k}", "text": paragraph(AI_SENTENCES, words, rng), "label": 1, "paper_id": paper, "domain": domain, "section": "methods", "generator": generators[k % len(generators)]})
    return rows


def write_rows(directory, rows, name="segments.jsonl"):
    path = Path(directory) / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


class NormalizeTest(unittest.TestCase):
    def test_strips_numeric_citations(self):
        for marker in ("[12]", "[1, 2]", "[1-3]", "[1]-[3]", "[4,5,6]"):
            self.assertEqual(normalize_segment(f"As shown before {marker}, it holds."), "As shown before, it holds.")

    def test_strips_author_year_citations_but_keeps_ordinary_parentheses(self):
        self.assertEqual(normalize_segment("It holds (Smith et al., 2019)."), "It holds.")
        self.assertEqual(normalize_segment("It holds (Smith and Jones, 2019; Lee, 2020a) here."), "It holds here.")
        self.assertEqual(normalize_segment("Smith et al. (2019) showed it."), "Smith et al. showed it.")
        self.assertEqual(normalize_segment("Small graphs (under 40 nodes) work."), "Small graphs (under 40 nodes) work.")
        self.assertEqual(normalize_segment("See Table 2 and [the appendix]."), "See Table 2 and [the appendix].")

    def test_removes_math_remnants_and_pdf_artefacts(self):
        self.assertEqual(normalize_segment("We set $x^2 = y$ for all."), "We set for all.")
        self.assertEqual(normalize_segment("A value \\(\\alpha\\) is \\textbf{used}."), "A value is.")
        self.assertEqual(normalize_segment("Given ∑ ≤ the bound."), "Given the bound.")
        self.assertEqual(normalize_segment("An exam-\nple of ﬁtting."), "An example of fitting.")

    def test_cleans_nothing_else_and_copes_with_empty_input(self):
        text = "We ran 3 trials in 2019, and accuracy rose by 4.5%."
        self.assertEqual(normalize_segment(text), text)
        self.assertEqual(normalize_segment(""), "")
        self.assertEqual(normalize_segment(None), "")

    def test_drops_formula_remnants_but_keeps_real_parentheticals(self):
        self.assertEqual(
            normalize_segment("The data {(x 1 , y 1 ), (x 2 , y 2 )} ignores (x n , y n ) so the unit (IMU) and (during separate sessions) stay."),
            "The data ignores so the unit (IMU) and (during separate sessions) stay.",
        )
        self.assertEqual(normalize_segment("Panel (a) shows f (x) for n = 4."), "Panel shows f for n 4.")

    def test_a_citation_only_paragraph_leaves_nothing(self):
        self.assertEqual(normalize_segment("[1] [2]"), "")


class DatasetLoadingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.row = {"id": "1", "text": "x", "label": 0, "paper_id": "p", "domain": "cs", "section": "intro"}

    def load(self, content):
        path = Path(self.directory.name) / "d.jsonl"
        path.write_text(content)
        return ds.load_segments(path)

    def test_bad_lines_stop_the_load_instead_of_being_skipped(self):
        for bad in (
            "not json\n",
            json.dumps({**self.row, "label": 2}) + "\n",
            json.dumps({**self.row, "label": True}) + "\n",
            json.dumps({k: v for k, v in self.row.items() if k != "paper_id"}) + "\n",
            json.dumps({**self.row, "text": ""}) + "\n",
            json.dumps({**self.row, "label": 1}) + "\n",
            json.dumps({**self.row, "test_set": "other"}) + "\n",
            json.dumps(self.row) + "\n" + json.dumps(self.row) + "\n",
            "",
        ):
            with self.assertRaises(ds.DatasetError, msg=bad[:40]):
                self.load(bad)

    def test_records_the_files_hash(self):
        content = json.dumps(self.row) + "\n"
        rows, digest = self.load(content)
        self.assertEqual(len(rows), 1)
        self.assertEqual(digest, hashlib.sha256(content.encode()).hexdigest())


class SplitTest(unittest.TestCase):
    def test_a_paper_never_crosses_splits(self):
        splits = ds.split_segments(make_rows(), 42)
        seen = {}
        for name, rows in splits.items():
            for row in rows:
                self.assertEqual(seen.setdefault(row["paper_id"], name), name)
        self.assertTrue(splits[ds.TRAIN] and splits[ds.CALIBRATION] and splits[ds.TEST])

    def test_assignment_ignores_row_order_and_follows_the_seed(self):
        rows = make_rows()
        shuffled = list(rows)
        random.Random(9).shuffle(shuffled)
        ids = lambda split: {row["id"] for row in split}
        self.assertEqual(ids(ds.split_segments(rows, 42)[ds.TEST]), ids(ds.split_segments(shuffled, 42)[ds.TEST]))
        self.assertNotEqual(ids(ds.split_segments(rows, 42)[ds.TEST]), ids(ds.split_segments(rows, 43)[ds.TEST]))

    def test_a_heldout_generator_never_reaches_fitting(self):
        splits = ds.split_segments(make_rows(), 42, heldout_generators=["gen-b"])
        for name in (ds.TRAIN, ds.CALIBRATION, ds.TEST):
            self.assertNotIn("gen-b", {row.get("generator") for row in splits[name]})
        self.assertEqual({row["generator"] for row in splits[ds.HELDOUT_GENERATOR]}, {"gen-b"})
        test_papers = {ds.assign_paper(row["paper_id"], 42) for row in splits[ds.HELDOUT_GENERATOR]}
        self.assertEqual(test_papers, {ds.TEST})

    def test_a_heldout_domain_goes_whole_to_its_own_set(self):
        splits = ds.split_segments(make_rows(), 42, heldout_domains=["physics"])
        self.assertEqual({row["domain"] for row in splits[ds.HELDOUT_DOMAIN]}, {"physics"})
        for name in (ds.TRAIN, ds.CALIBRATION, ds.TEST):
            self.assertNotIn("physics", {row["domain"] for row in splits[name]})

    def test_a_test_only_rows_paper_never_reaches_fitting(self):
        rows = make_rows()
        rows.append({"id": "polished-1", "text": "x", "label": 0, "paper_id": "paper-3", "domain": "cs", "section": "methods", "test_set": "polished"})
        splits = ds.split_segments(rows, 42)
        self.assertEqual([row["id"] for row in splits["polished"]], ["polished-1"])
        for name in (ds.TRAIN, ds.CALIBRATION):
            self.assertNotIn("paper-3", {row["paper_id"] for row in splits[name]})
        self.assertIn("paper-3", {row["paper_id"] for row in splits[ds.TEST]})

    def test_the_overlap_check_catches_a_leak(self):
        row = {"paper_id": "p"}
        with self.assertRaises(ds.DatasetError):
            ds.assert_no_paper_overlap({ds.TRAIN: [row], ds.CALIBRATION: [], ds.TEST: [row]})


class FeatureChecksTest(unittest.TestCase):
    NAMES = ["a", "b"]

    def test_a_constant_feature_stops_training_and_is_named(self):
        with self.assertRaisesRegex(ts.TrainingAborted, "never vary: b"):
            ts.checked_matrix([[1.0, 0.0], [2.0, 0.0]], self.NAMES, "train")

    def test_non_finite_values_stop_training(self):
        with self.assertRaisesRegex(ts.TrainingAborted, "non-finite values in a"):
            ts.checked_matrix([[float("inf"), 1.0], [2.0, 3.0]], self.NAMES, "train")

    def test_too_many_failed_rows_stop_training(self):
        rows = [[float(i), float(i)] for i in range(100)] + [None]
        with self.assertRaisesRegex(ts.TrainingAborted, "failed for 1 of 101"):
            ts.checked_matrix(rows, self.NAMES, "train")

    def test_a_few_failed_rows_are_dropped(self):
        rows = [[float(i), float(i)] for i in range(2000)] + [None]
        matrix, kept = ts.checked_matrix(rows, self.NAMES, "train")
        self.assertEqual(matrix.shape, (2000, 2))
        self.assertEqual(kept, list(range(2000)))


class ThresholdTest(unittest.TestCase):
    def test_flags_at_most_the_target_share_of_human_text(self):
        human = np.linspace(0.0, 0.99, 1000)
        threshold = ts.threshold_for_fpr(human, 0.01)
        self.assertLessEqual(np.mean(human >= threshold), 0.01)
        self.assertGreater(np.mean(human >= threshold), 0.005)

    def test_ties_never_push_the_rate_over_the_target(self):
        human = np.array([0.0] * 500 + [0.9] * 500)
        self.assertLessEqual(np.mean(human >= ts.threshold_for_fpr(human, 0.01)), 0.01)

    def test_flags_nothing_when_all_human_text_scores_the_top(self):
        self.assertGreater(ts.threshold_for_fpr(np.ones(100), 0.01), 1.0)


class FeatureSetTest(unittest.TestCase):
    def test_style_v2_is_length_free(self):
        rng = random.Random(3)
        short = paragraph(HUMAN_SENTENCES, 60, rng)
        rows = feature_rows([short, " ".join([short] * 4)], "style_v2")
        self.assertEqual(len(rows[0]), len(feature_names("style_v2")))
        self.assertNotIn("word_count", feature_names("style_v2"))
        self.assertAlmostEqual(rows[0][feature_names("style_v2").index("punctuation_per_word")], rows[1][feature_names("style_v2").index("punctuation_per_word")], places=2)


class TrainEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.data = write_rows(cls.directory.name, make_rows(papers=200))
        cls.out = str(Path(cls.directory.name) / "artifacts")
        cls.root_before = cls.root_hashes()
        cls.artifacts = ts.train(cls.data, out_root=cls.out, version="v-test", epochs=6)

    @staticmethod
    def root_hashes():
        return {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("model.pt", "model_config.pkl", "feature_scaler.pkl")
            if (ROOT / name).exists()
        }

    def test_writes_a_complete_self_describing_artifact(self):
        self.assertEqual(
            sorted(path.name for path in self.artifacts.iterdir()),
            ["calibrator.pkl", "feature_scaler.pkl", "model.pt", "model_config.pkl"],
        )
        config = SegmentModel(self.artifacts).config
        self.assertEqual(config["model_version"], "v-test")
        self.assertEqual(config["feature_set"], "style_v2")
        self.assertEqual(config["text_normalization"], "v2")
        self.assertEqual(config["split_seed"], 42)
        self.assertEqual(config["dataset_sha256"], ds.load_segments(self.data)[1])
        self.assertLessEqual(config["segment_thresholds"]["possibly_ai"], config["segment_thresholds"]["likely_ai"])

    def test_never_touches_the_deployed_model(self):
        self.assertEqual(self.root_hashes(), self.root_before)

    def test_refuses_to_overwrite_a_version(self):
        with self.assertRaisesRegex(ts.TrainingAborted, "never overwritten"):
            ts.train(self.data, out_root=self.out, version="v-test", epochs=1)

    def test_refuses_the_old_zero_perplexity_features_and_leaves_nothing_behind(self):
        """The check the deployed model would have failed, on the same features it was given."""
        with self.assertRaisesRegex(ts.TrainingAborted, "perplexity_gpt2.*log_perplexity_scibert"):
            ts.train(self.data, out_root=self.out, version="v-zeros", feature_set="legacy_zero_perplexity", epochs=1)
        self.assertFalse((Path(self.out) / "v-zeros").exists())

    def test_a_failure_while_training_leaves_nothing_behind(self):
        with patch.object(ts, "save_mlp", side_effect=RuntimeError("disk full")), self.assertRaises(RuntimeError):
            ts.train(self.data, out_root=self.out, version="v-crash", epochs=1)
        self.assertEqual(sorted(path.name for path in Path(self.out).iterdir()), ["v-test"])

    def test_cli_reports_an_abort_with_a_failing_exit_code(self):
        self.assertEqual(ts.main(["--data", self.data, "--out", self.out, "--version", "v-test"]), 1)

    def test_also_trains_gradient_boosting(self):
        artifacts = ts.train(self.data, out_root=self.out, version="v-gbm", model_kind="gbm")
        self.addCleanup(lambda: __import__("shutil").rmtree(artifacts))
        self.assertTrue((artifacts / "model.pkl").exists())
        self.assertFalse((artifacts / "feature_scaler.pkl").exists())
        probabilities, kept = SegmentModel(artifacts).predict([paragraph(AI_SENTENCES, 100, random.Random(1))])
        self.assertEqual(kept, [0])
        self.assertGreater(probabilities[0], 0.5)

    def test_the_model_reads_back_and_ranks_the_classes(self):
        model = SegmentModel(self.artifacts)
        rng = random.Random(5)
        probabilities, kept = model.predict([paragraph(HUMAN_SENTENCES, 100, rng), paragraph(AI_SENTENCES, 100, rng), "[1] [2]"])
        self.assertEqual(kept, [0, 1])
        self.assertLess(probabilities[0], probabilities[1])

    def test_evaluation_refuses_another_dataset(self):
        other = write_rows(self.directory.name, make_rows(papers=20, seed=2), "other.jsonl")
        with self.assertRaisesRegex(ValueError, "not the dataset"):
            ev.evaluate(str(self.artifacts), other)

    def test_evaluation_is_not_ready_without_its_test_only_sets(self):
        report = ev.evaluate(str(self.artifacts), self.data)
        json.dumps(report)
        self.assertFalse(report["ready_to_ship"])
        self.assertEqual(report["gates"]["human_document_flag_rate"]["status"], ev.NOT_MEASURED)
        self.assertEqual(report["gates"]["non_native_fpr"]["status"], ev.NOT_MEASURED)
        self.assertEqual(report["gates"]["cpu_latency"]["status"], ev.NOT_MEASURED)
        self.assertGreater(report["metrics"]["auroc_test"], 0.9)
        self.assertIn("methods", report["breakdowns"]["section"])
        self.assertTrue(ev.summary_lines(report)[-1].endswith("NO"))

    def test_latency_is_measured_when_asked(self):
        report = ev.evaluate(str(self.artifacts), self.data, latency=True)
        self.assertEqual(report["gates"]["cpu_latency"]["status"], ev.PASS)


class GateTest(unittest.TestCase):
    GOOD = dict(
        human_test={"n": 1000, "rate": 0.005},
        document={"n": 100, "rate": 0.01},
        ai_test={"n": 1000, "rate": 0.85},
        ai_generator={"n": 500, "rate": 0.75},
        bucket_recalls={name: {"n": 100, "rate": 0.85} for name in ("40-80", "80-120", "120-180", "180-300")},
        non_native={"n": 400, "rate": 0.01},
        feature_failure_rate=0.0,
        latency_seconds=5.0,
    )

    def statuses(self, **changes):
        gates = ev.evaluate_gates(**{**self.GOOD, **changes})
        return {name: gate["status"] for name, gate in gates.items()}

    def test_a_good_model_passes_every_gate(self):
        self.assertEqual(set(self.statuses().values()), {ev.PASS})

    def test_todays_length_problem_fails_the_spread_gate(self):
        buckets = {"40-80": {"n": 100, "rate": 0.967}, "80-120": {"n": 100, "rate": 0.5}, "120-180": {"n": 100, "rate": 0.107}, "180-300": {"n": 100, "rate": 0.05}}
        self.assertEqual(self.statuses(bucket_recalls=buckets)["ai_recall_length_spread"], ev.FAIL)

    def test_each_limit_fails_when_crossed(self):
        cases = {
            "human_paragraph_fpr": {"human_test": {"n": 1000, "rate": 0.011}},
            "human_document_flag_rate": {"document": {"n": 100, "rate": 0.02}},
            "ai_recall_heldout_papers": {"ai_test": {"n": 1000, "rate": 0.79}},
            "ai_recall_heldout_generator": {"ai_generator": {"n": 500, "rate": 0.69}},
            "non_native_fpr": {"non_native": {"n": 400, "rate": 0.021}},
            "feature_failures": {"feature_failure_rate": 0.01},
            "cpu_latency": {"latency_seconds": 60.0},
        }
        for gate, change in cases.items():
            self.assertEqual(self.statuses(**change)[gate], ev.FAIL, gate)

    def test_non_native_must_also_sit_close_to_the_rest(self):
        statuses = self.statuses(human_test={"n": 1000, "rate": 0.001}, non_native={"n": 400, "rate": 0.015})
        self.assertEqual(statuses["non_native_fpr"], ev.FAIL)

    def test_too_little_data_is_not_a_pass(self):
        cases = {
            "human_paragraph_fpr": {"human_test": {"n": 40, "rate": 0.0}},
            "human_document_flag_rate": {"document": {"n": 5, "rate": 0.0}},
            "ai_recall_heldout_papers": {"ai_test": {"n": 40, "rate": 1.0}},
            "ai_recall_heldout_generator": {"ai_generator": {"n": 40, "rate": 1.0}},
            "non_native_fpr": {"non_native": {"n": 40, "rate": 0.0}},
            "ai_recall_length_spread": {"bucket_recalls": {"40-80": {"n": 100, "rate": 0.9}}},
            "cpu_latency": {"latency_seconds": None},
        }
        for gate, change in cases.items():
            self.assertEqual(self.statuses(**change)[gate], ev.NOT_MEASURED, gate)

    def test_the_interval_shows_how_little_a_small_sample_proves(self):
        self.assertGreater(ev.wilson_upper(0, 40), 0.05)
        self.assertLess(ev.wilson_upper(0, 10000), 0.001)


class ServingNormalizationTest(unittest.TestCase):
    """Normalizing is opt-in: the deployed model was trained on raw text."""

    TEXT = "Our method improves the baseline on every benchmark we report [12]."

    def texts_seen(self, config, text=None):
        captured = {}

        def rows(texts, **_):
            captured["texts"] = list(texts)
            return [[1.0, 2.0] for _ in texts]

        with patch.object(app_module, "extract_feature_rows", side_effect=rows), patch.object(
            app_module, "_get_feature_scaler", return_value=None
        ):
            matrix, kept = app_module._build_feature_batch([text or self.TEXT], config)
        return captured["texts"], kept

    def test_a_model_without_the_setting_keeps_its_raw_text(self):
        self.assertEqual(self.texts_seen({})[0], [self.TEXT])

    def test_the_deployed_model_config_does_not_opt_in(self):
        import joblib

        path = ROOT / "model_config.pkl"
        if not path.exists():
            self.skipTest("no model_config.pkl here")
        self.assertFalse(joblib.load(path).get("text_normalization"))

    def test_a_model_trained_on_normalized_text_gets_normalized_text(self):
        self.assertEqual(
            self.texts_seen({"text_normalization": "v2"})[0],
            ["Our method improves the baseline on every benchmark we report."],
        )

    def test_nothing_left_after_cleaning_is_unscored_not_a_row_of_zeros(self):
        self.assertEqual(self.texts_seen({"text_normalization": "v2"}, "[1] [2] [3] [4] [5] [6]")[1], [])


if __name__ == "__main__":
    unittest.main()
