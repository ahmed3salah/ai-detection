"""The dataset scripts: what counts as eligible, what gets rejected, and that the budget holds."""
import importlib.util
import json
import random
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


human = load_script("build_human_paragraphs")
gen = load_script("generate_ai_paragraphs")
merge = load_script("merge_segments")
leak = load_script("leak_check")

BY = "http://creativecommons.org/licenses/by/4.0/"


def arxiv_page(records, token=""):
    body = ""
    for identifier, created, licence, categories in records:
        body += f"""<record><header><identifier>oai:arXiv.org:{identifier}</identifier></header><metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{identifier}</id><created>{created}</created><title>Paper {identifier}</title>
        <categories>{categories}</categories><license>{licence}</license></arXiv></metadata></record>"""
    resumption = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{body}{resumption}</ListRecords></OAI-PMH>'


def pmc_article(pmc_id="123", licence=BY, year="2019", body=True):
    p = " ".join(["The patients in this cohort were followed for a year and the results were recorded by the clinic for all of them."] * 2)
    main = f"""<body><sec><title>Introduction</title><p>{p} <xref ref-type="bibr">[1]</xref></p><fig><caption><p>Figure caption not prose at all.</p></caption></fig>
    <sec><title>Nested</title><p>{p}</p></sec></sec><sec><title>Results</title><p>{p}<inline-formula>x=1</inline-formula></p><table-wrap><table><tr><td>cell</td></tr></table></table-wrap></sec></body>""" if body else ""
    lic = f'<license xlink:href="{licence}"><license-p>Open.</license-p></license>' if licence else '<license license-type="open-access"><license-p>Open.</license-p></license>'
    return f"""<record><header><identifier>oai:pubmedcentral.nih.gov:{pmc_id}</identifier></header><metadata>
    <article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/" xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
    <title-group><article-title>A study {pmc_id}</article-title></title-group><pub-date><year>{year}</year></pub-date>
    <permissions>{lic}</permissions><abstract><p>{p}</p></abstract></article-meta></front>{main}<back><ref-list><ref>A cited work.</ref></ref-list></back></article></metadata></record>"""


def pmc_page(*articles):
    return f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{"".join(articles)}</ListRecords></OAI-PMH>'


class EligibilityTest(unittest.TestCase):
    def test_arxiv_keeps_only_cc_by_or_cc0_from_2021_or_before(self):
        page = arxiv_page([
            ("2001.00001", "2020-01-05", BY, "cs.LG stat.ML"),
            ("2001.00002", "2020-01-05", "http://creativecommons.org/publicdomain/zero/1.0/", "eess.SP"),
            ("2001.00003", "2020-01-05", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "cs.LG"),
            ("2001.00004", "2020-01-05", "http://creativecommons.org/licenses/by-nc-nd/4.0/", "cs.LG"),
            ("2001.00005", "2020-01-05", "http://creativecommons.org/licenses/by-sa/4.0/", "cs.LG"),
            ("2201.00006", "2022-01-01", BY, "cs.LG"),
        ], token="next")
        candidates, token, total = human.parse_arxiv_records(page)
        self.assertEqual([c.paper_id for c in candidates], ["arxiv:2001.00001", "arxiv:2001.00002"])
        self.assertEqual([c.domain for c in candidates], ["cs", "eess"])
        self.assertEqual(token, "next")
        self.assertEqual(total, 6)

    def test_first_submission_comes_from_the_id_not_the_created_field(self):
        candidates, _, _ = human.parse_arxiv_records(arxiv_page([
            ("1902.10899", "2021-05-31", BY, "cs.IT"),
            ("2203.00001", "2021-05-31", BY, "cs.IT"),
        ]))
        self.assertEqual([(c.paper_id, c.first_date) for c in candidates], [("arxiv:1902.10899", "2019")])

    def test_reads_both_arxiv_id_formats_and_keeps_hyphenated_archives(self):
        self.assertEqual(human.arxiv_first_year("cond-mat/0504132"), 2005)
        self.assertEqual(human.arxiv_first_year("hep-th/9901001"), 1999)
        self.assertEqual(human.arxiv_first_year("2008.03408"), 2020)
        candidates, _, _ = human.parse_arxiv_records(arxiv_page([("2001.00001", "2020-01-01", BY, "q-bio.GN cs.LG")]))
        self.assertEqual(candidates[0].domain, "q-bio")

    def test_the_harvest_window_keeps_out_papers_revised_after_2021(self):
        self.assertLessEqual(human.HARVEST_END.year, human.LAST_ALLOWED_YEAR)

    def test_pmc_needs_a_named_open_licence_and_an_old_enough_date(self):
        candidates, _, total = human.parse_pmc_records(pmc_page(
            pmc_article("1"),
            pmc_article("2", licence="https://creativecommons.org/licenses/by-nc/4.0/"),
            pmc_article("3", licence=""),
            pmc_article("4", year="2023"),
            pmc_article("5", body=False),
        ))
        self.assertEqual([c.paper_id for c in candidates], ["pmc:1"])
        self.assertEqual(candidates[0].domain, "pmc")
        self.assertEqual(total, 5)


class JatsTest(unittest.TestCase):
    def setUp(self):
        self.article = human.parse_pmc_records(pmc_page(pmc_article()))[0][0].jats

    def test_reads_abstract_and_sections_without_figures_tables_or_references(self):
        segments = human.segments_from_jats(self.article)
        self.assertEqual([s.section for s in segments], ["Abstract", "Introduction", "Nested", "Results"])
        text = " ".join(s.text for s in segments)
        for absent in ("caption", "cell", "A cited work", "x=1"):
            self.assertNotIn(absent, text)

    def test_rows_carry_provenance_and_a_section_class(self):
        candidate = human.parse_pmc_records(pmc_page(pmc_article()))[0][0]
        segments = human.segments_from_jats(candidate.jats) * 2
        rows = human.rows_for(candidate, segments)
        self.assertEqual(rows[0]["section"], "abstract")
        self.assertEqual(rows[1]["section"], "introduction")
        self.assertEqual(rows[3]["section"], "results")
        self.assertEqual(rows[0]["licence"], BY)
        self.assertEqual(rows[0]["label"], 0)
        self.assertEqual(rows[0]["paper_id"], "pmc:123")

    def test_a_paper_with_too_few_usable_paragraphs_gives_nothing(self):
        candidate = human.parse_pmc_records(pmc_page(pmc_article()))[0][0]
        self.assertEqual(human.rows_for(candidate, human.segments_from_jats(candidate.jats)[:2]), [])


class ProseFilterTest(unittest.TestCase):
    def test_keeps_english_prose_and_drops_tables_and_other_languages(self):
        self.assertTrue(human.is_prose("We show that the model is stable in the limit of the data and that it is fast."))
        self.assertFalse(human.is_prose("1.2 3.4 5.6 7.8 9.0 1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8"))
        self.assertFalse(human.is_prose("Nous montrons que le modele reste stable lorsque les donnees augmentent fortement ici."))

    def test_section_classes(self):
        self.assertEqual(human.section_class("1 Introduction"), "introduction")
        self.assertEqual(human.section_class("3. Experimental Results"), "results")
        self.assertEqual(human.section_class("Materials and Methods"), "methods")
        self.assertEqual(human.section_class("Conclusions and future work"), "conclusion")
        self.assertEqual(human.section_class("Acknowledgements"), "other")
        self.assertEqual(human.section_class(None), "other")

    def test_quotas_add_up_exactly(self):
        for total in (1, 7, 200, 5003):
            self.assertEqual(sum(human.quotas(total, human.DOMAIN_WEIGHTS).values()), total)
        self.assertEqual(human.quotas(200, human.DOMAIN_WEIGHTS)["cs"], 90)


def human_rows(papers=60, per_paper=8):
    rng = random.Random(4)
    rows = []
    for p in range(papers):
        for k in range(per_paper):
            words = rng.choice([50, 100, 150, 250])
            rows.append({
                "id": f"arxiv:{p}#{k}", "text": " ".join(["word"] * words), "label": 0, "paper_id": f"arxiv:{p}", "domain": "cs",
                "section": "methods" if k else "abstract", "heading": "3 Method", "position": k, "title": f"Paper {p}",
            })
    return rows


TRAINING = ["vendor-a/one", "vendor-b/two"]
HELDOUT = ["vendor-c/three"]
PRICING = {m: (1e-6, 2e-6) for m in TRAINING + HELDOUT}


def tasks(count=100, polish=0, rows=None):
    return gen.build_tasks(rows or human_rows(), count, 42, TRAINING, HELDOUT, polish)


def good_reply(task, cost=0.001):
    return {"text": " ".join(["word"] * (task.target_words - 1)) + " end.", "finish_reason": "stop", "cost": cost}


class TaskBuildingTest(unittest.TestCase):
    def test_every_length_bucket_gets_the_same_share(self):
        counts = {}
        for task in tasks(80):
            bucket = next(b for b in gen.LENGTH_BUCKETS if b[0] <= task.target_words <= b[1])
            counts[bucket] = counts.get(bucket, 0) + 1
        self.assertEqual(set(counts.values()), {20})

    def test_the_target_length_is_the_human_paragraphs_length(self):
        for task in tasks(40):
            self.assertEqual(task.target_words, len(task.source["text"].split()))
            self.assertIn(f"{task.target_words} words", task.prompt)

    def test_the_prompt_never_contains_the_paragraph_it_replaces(self):
        rows = human_rows()
        for i, row in enumerate(rows):
            row["text"] = f"UNIQUEMARK{i} " + row["text"]
        for task in gen.build_tasks(rows, 100, 42, TRAINING, HELDOUT):
            self.assertNotIn(task.source["text"].split()[0], task.prompt)

    def test_continuation_uses_the_real_previous_paragraph_and_the_first_gets_an_outline(self):
        rows = human_rows(per_paper=4)
        modes = {task.mode for task in gen.build_tasks(rows, 200, 42, TRAINING, HELDOUT)}
        self.assertEqual(modes, {"abstract", "continue", "outline"})
        by_id = {row["id"]: row for row in rows}
        for task in gen.build_tasks(rows, 200, 42, TRAINING, HELDOUT):
            if task.mode == "continue":
                position = task.source["position"]
                previous = by_id[f"{task.source['paper_id']}#{position - 1}"]
                self.assertIn(previous["text"], task.prompt)

    def test_a_heldout_generator_only_writes_for_test_papers(self):
        seen = [task for task in tasks(300) if task.model in HELDOUT]
        self.assertTrue(seen)
        self.assertEqual({gen.assign_paper(task.source["paper_id"], 42) for task in seen}, {gen.TEST})

    def test_tasks_are_stable_between_runs(self):
        self.assertEqual([t.id for t in tasks(50)], [t.id for t in tasks(50)])

    def test_polished_tasks_show_the_paragraph_and_avoid_replaced_ones(self):
        all_tasks = tasks(40, polish=10)
        polished = [t for t in all_tasks if t.mode == "polish"]
        self.assertEqual(len(polished), 10)
        self.assertTrue(all(t.source["text"] in t.prompt for t in polished))
        self.assertFalse({t.source["id"] for t in polished} & {t.source["id"] for t in all_tasks if t.mode != "polish"})


class RejectionTest(unittest.TestCase):
    OK = " ".join(["word"] * 99) + " end."

    def test_accepts_a_clean_reply(self):
        self.assertIsNone(gen.rejection_reason(self.OK, "stop", 100))

    def test_rejects_what_got_into_the_old_dataset(self):
        cases = {
            "finish_reason:length": (self.OK, "length"),
            "no_sentence_end": ("We present a computational model investigating tidally induced", "stop"),
            "empty": ("  ", "stop"),
            "meta_or_markdown": ("Here is the paragraph: " + self.OK, "stop"),
        }
        for reason, (text, finish) in cases.items():
            self.assertEqual(gen.rejection_reason(text, finish, 100 if reason != "no_sentence_end" else 8), reason)
        self.assertEqual(gen.rejection_reason("**Bold** " + self.OK, "stop", 100), "meta_or_markdown")
        self.assertEqual(gen.rejection_reason("# Heading\n" + self.OK, "stop", 100), "meta_or_markdown")

    def test_rejects_a_length_more_than_30_percent_off(self):
        self.assertEqual(gen.rejection_reason(self.OK, "stop", 200), "word_count")
        self.assertEqual(gen.rejection_reason(self.OK, "stop", 60), "word_count")
        self.assertIsNone(gen.rejection_reason(self.OK, "stop", 130))


class BudgetAndResumeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.output = self.dir / "ai.jsonl"
        self.ledger_path = self.dir / "ledger.jsonl"

    def run_tasks(self, todo, complete, budget=100.0):
        return gen.run(todo, output=self.output, ledger=gen.Ledger(self.ledger_path), budget=budget, pricing=PRICING, complete=complete, workers=4)

    def test_writes_rows_with_the_right_labels(self):
        todo = tasks(8, polish=2)
        report = self.run_tasks(todo, good_reply)
        self.assertEqual(report["statuses"], {"ok": 10})
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual({r["label"] for r in rows if r["mode"] != "polish"}, {1})
        polished = [r for r in rows if r["mode"] == "polish"]
        self.assertEqual({(r["label"], r["test_set"]) for r in polished}, {(0, "polished")})
        self.assertTrue(all(r["generator"] for r in rows))

    def test_never_spends_past_the_budget(self):
        todo = tasks(200)
        worst = max(gen.worst_case_cost(t, PRICING) for t in todo)
        # Every call costs its worst case, the hardest test of the reservation.
        report = self.run_tasks(todo, lambda t: good_reply(t, gen.worst_case_cost(t, PRICING)), budget=worst * 5)
        self.assertTrue(report["stopped_on_budget"])
        self.assertLessEqual(report["spent_usd"], worst * 5)
        self.assertGreater(report["statuses"]["ok"], 0)
        self.assertGreater(report["statuses"]["skipped"], 0)

    def test_the_budget_counts_earlier_runs(self):
        todo = tasks(20)
        self.run_tasks(todo[:10], lambda t: good_reply(t, 0.05))
        spent = gen.Ledger(self.ledger_path).spent
        self.assertAlmostEqual(spent, 0.5)
        calls = []
        report = self.run_tasks(todo[10:], lambda t: calls.append(t) or good_reply(t), budget=spent)
        self.assertEqual(calls, [])
        self.assertTrue(report["stopped_on_budget"])

    def test_a_rerun_repeats_nothing_it_paid_for(self):
        todo = tasks(12)
        rejected = {t.id for t in todo[:3]}
        self.run_tasks(todo, lambda t: {"text": "too short.", "finish_reason": "stop", "cost": 0.01} if t.id in rejected else good_reply(t))
        calls = []
        report = self.run_tasks(todo, lambda t: calls.append(t) or good_reply(t))
        self.assertEqual(calls, [])
        self.assertEqual(report["already_done"], 12)

    def test_a_provider_failure_costs_nothing_and_can_be_tried_again(self):
        todo = tasks(4)

        def fail(task):
            raise RuntimeError("HTTP 500")

        first = self.run_tasks(todo, fail)
        self.assertEqual(first["statuses"]["error"], 4)
        self.assertEqual(first["spent_usd"], 0)
        self.assertEqual(self.run_tasks(todo, good_reply)["statuses"], {"ok": 4})

    def test_a_broken_model_is_switched_off_instead_of_retried_forever(self):
        calls = []
        lock = threading.Lock()

        def fail(task):
            with lock:
                calls.append(task)
            raise RuntimeError("HTTP 404")

        todo = [t for t in tasks(100) if t.model == TRAINING[0]]
        report = gen.run(todo, output=self.output, ledger=gen.Ledger(self.ledger_path), budget=100, pricing=PRICING, complete=fail, workers=1)
        self.assertEqual(report["models_disabled"], [TRAINING[0]])
        self.assertEqual(len(calls), gen.MAX_CONSECUTIVE_MODEL_ERRORS)

    def test_a_rejected_replys_cost_still_counts(self):
        todo = tasks(3)
        report = self.run_tasks(todo, lambda t: {"text": "no", "finish_reason": "length", "cost": 0.02})
        self.assertAlmostEqual(report["spent_usd"], 0.06)
        self.assertEqual(report["statuses"], {"rejected": 3})
        self.assertFalse(self.output.exists())

    def test_the_ledger_holds_no_manuscript_text(self):
        todo = tasks(3)
        self.run_tasks(todo, good_reply)
        self.assertNotIn("word word", self.ledger_path.read_text())

    def test_dry_run_estimates_without_calling_anything(self):
        summary = gen.dry_run_summary(tasks(20), PRICING)
        self.assertEqual(summary["calls"], 20)
        self.assertGreater(summary["worst_case_usd"], summary["estimate_usd"] - 0.01)


class OpenRouterCallTest(unittest.TestCase):
    class Answer:
        def __init__(self, status, payload=None):
            self.status_code = status
            self.payload = payload or {}

        def json(self):
            return self.payload

    OK = {"choices": [{"message": {"content": "Text."}, "finish_reason": "stop"}], "usage": {"cost": 0.002, "prompt_tokens": 10, "completion_tokens": 5}}

    def session(self, *answers):
        test = self
        test.bodies = []
        queue = list(answers)

        class Session:
            def post(self, url, json, headers, timeout):
                test.bodies.append(dict(json))
                return queue.pop(0)

        return Session()

    def setUp(self):
        gen._MODELS_THAT_REFUSE_REASONING_SETTING.clear()
        self.addCleanup(gen._MODELS_THAT_REFUSE_REASONING_SETTING.clear)
        self.task = tasks(1)[0]

    def test_asks_for_no_reasoning_and_reads_the_cost(self):
        reply = gen.openrouter_completion(self.session(self.Answer(200, self.OK)), "key", self.task, PRICING)
        self.assertEqual(reply, {"text": "Text.", "finish_reason": "stop", "cost": 0.002})
        self.assertEqual(self.bodies[0]["reasoning"], {"enabled": False})

    def test_a_model_that_refuses_the_setting_goes_without_it_from_then_on(self):
        session = self.session(self.Answer(400), self.Answer(200, self.OK), self.Answer(200, self.OK))
        gen.openrouter_completion(session, "key", self.task, PRICING)
        gen.openrouter_completion(session, "key", self.task, PRICING)
        self.assertEqual(["reasoning" in body for body in self.bodies], [True, False, False])

    def test_a_400_without_the_setting_is_a_real_error(self):
        session = self.session(self.Answer(400), self.Answer(400))
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            gen.openrouter_completion(session, "key", self.task, PRICING)

    def test_works_out_the_cost_when_the_provider_gives_none(self):
        answer = {**self.OK, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        reply = gen.openrouter_completion(self.session(self.Answer(200, answer)), "key", self.task, PRICING)
        self.assertAlmostEqual(reply["cost"], 10 * 1e-6 + 5 * 2e-6)


class MergeTest(unittest.TestCase):
    def test_pairs_each_ai_row_with_the_paragraph_it_replaced(self):
        rows = human_rows(papers=5, per_paper=3)
        ai = [gen.row_for(t, "Text here.") for t in gen.build_tasks(rows, 6, 42, TRAINING, [])]
        merged = merge.merge(rows, ai)
        self.assertEqual(sum(r["label"] == 1 for r in merged), 6)
        self.assertEqual(sum(r["label"] == 0 for r in merged), 6)
        self.assertNotIn("title", merged[0])

    def test_a_polished_row_does_not_bring_its_source_along(self):
        rows = human_rows(papers=5, per_paper=3)
        task = gen.build_tasks(rows, 0, 42, TRAINING, [], 1)[0]
        merged = merge.merge(rows, [gen.row_for(task, "Edited.")])
        self.assertEqual([r["test_set"] for r in merged], ["polished"])

    def test_the_merged_file_loads_through_the_strict_loader(self):
        from segment_dataset import load_segments

        rows = human_rows(papers=5, per_paper=3)
        ai = [gen.row_for(t, "Text here.") for t in gen.build_tasks(rows, 6, 42, TRAINING, [], 2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in merge.merge(rows, ai)))
            loaded, _ = load_segments(path)
        self.assertEqual(len(loaded), 14)


class LeakCheckTest(unittest.TestCase):
    @staticmethod
    def rows(ai_text, human_text):
        rng = random.Random(1)
        rows = []
        for p in range(80):
            for label, make in ((0, human_text), (1, ai_text)):
                rows.append({"id": f"{p}-{label}", "text": make(rng), "label": label, "paper_id": f"p{p}", "domain": "cs", "section": "methods", **({"generator": "g"} if label else {})})
        return rows

    SENTENCE = "The method was tested on a large public corpus and the results were good. "

    def test_fails_when_length_separates_the_classes(self):
        report = leak.check(self.rows(lambda r: self.SENTENCE * r.randint(3, 4), lambda r: self.SENTENCE * r.randint(10, 14)))
        self.assertFalse(report["passed"])
        self.assertEqual(next(iter(report["single_feature_auroc"])), "word_count")

    def test_fails_when_pdf_traces_separate_the_classes_after_normalising(self):
        report = leak.check(self.rows(lambda r: self.SENTENCE * 4, lambda r: self.SENTENCE * 4 + "ﬁtted without an ending"))
        self.assertFalse(report["passed"])

    def test_passes_when_the_classes_match(self):
        report = leak.check(self.rows(lambda r: self.SENTENCE * r.randint(3, 12), lambda r: self.SENTENCE * r.randint(3, 12)))
        self.assertTrue(report["passed"], report)

    def test_citation_markers_alone_do_not_fail_it_since_normalising_removes_them(self):
        report = leak.check(self.rows(lambda r: self.SENTENCE * r.randint(3, 12), lambda r: (self.SENTENCE + "[12] ") * r.randint(3, 12)))
        self.assertTrue(report["passed"], report)


class EncodingTest(unittest.TestCase):
    """The GPU machine runs Windows, whose default encoding (cp1252) can't hold most of this text."""

    def test_jsonl_round_trips_any_text_and_splits_only_on_real_line_ends(self):
        from segment_dataset import append_jsonl, read_jsonl

        rows = [{"text": "\u03b1-helix\u2028next \u4e2d\u6587 \x85 caf\u00e9"}, {"text": "second"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            append_jsonl(path, rows[:1])
            append_jsonl(path, rows[1:])
            self.assertEqual(read_jsonl(path), rows)
            self.assertNotIn(b"\r", path.read_bytes())

    def test_no_script_uses_the_systems_default_encoding(self):
        import ast

        files = [ROOT / "scripts" / f"{name}.py" for name in ("build_human_paragraphs", "generate_ai_paragraphs", "merge_segments", "leak_check", "check_human_paragraphs")]
        files += [ROOT / name for name in ("segment_dataset.py", "segmenter.py", "segment_model.py", "train_segments.py", "evaluate_segments.py")]
        for path in files:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name in ("read_text", "write_text"):
                    self.assertIn("encoding", [k.arg for k in node.keywords], f"{path.name}:{node.lineno}")
                if name == "open":
                    mode = next((a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)), "r")
                    if "b" not in mode:
                        self.assertIn("encoding", [k.arg for k in node.keywords], f"{path.name}:{node.lineno}")

    def test_an_arxiv_page_is_read_as_utf8_from_bytes(self):
        page = arxiv_page([("2001.00001", "2020-01-01", BY, "cs.LG")]).replace("Paper 2001.00001", "\u03b1-stable caf\u00e9")
        candidates, _, _ = human.parse_arxiv_records(('<?xml version="1.0" encoding="UTF-8"?>' + page).encode("utf-8"))
        self.assertEqual(candidates[0].title, "\u03b1-stable caf\u00e9")


class GrobidCheckTest(unittest.TestCase):
    def test_a_page_that_is_not_tei_is_a_failure_not_an_empty_paper(self):
        class Answer:
            status_code = 206
            content = b"<!DOCTYPE html><html><body>Preparing Space</body></html>"

            def raise_for_status(self):
                pass

        class Session:
            def post(self, *args, **kwargs):
                return Answer()

        with self.assertRaisesRegex(RuntimeError, "did not answer with TEI"):
            human.grobid_tei(Session(), "http://grobid", b"%PDF")


if __name__ == "__main__":
    unittest.main()
