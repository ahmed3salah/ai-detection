"""The AI-typical wording list: what it keeps, what it drops, and what it is measured at."""
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("build_wording_lexicon", Path(__file__).resolve().parents[1] / "scripts" / "build_wording_lexicon.py")
lexicon = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexicon)

GENERATORS = [f"vendor-{i}/model" for i in range(6)]


def rows(papers=400):
    """Each paper: a human paragraph and an AI one on the same topic. AI text from every model says
    "delves into"; one model alone also says "tapestry"; both sides share the topic word."""
    out = []
    for n in range(papers):
        topic = f"topic{n % 7}"
        out.append({"id": f"h{n}", "paper_id": f"p{n}", "label": 0, "text": f"We measure the {topic} carefully and report every result in this study."})
        generator = GENERATORS[n % len(GENERATORS)]
        extra = " A rich tapestry emerges." if generator == GENERATORS[0] else ""
        out.append({"id": f"a{n}", "paper_id": f"p{n}", "label": 1, "generator": generator,
                    "text": f"This study delves into the {topic} and its measurement in depth.{extra}"})
    return out


class WordingLexiconTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = lexicon.build(rows(), stop={"the", "and", "its", "in", "we", "this", "a", "every", "of"})
        cls.phrases = {entry["phrase"]: entry for entry in cls.result["terms"]}

    def test_keeps_wording_many_models_share(self):
        self.assertIn("delves into", self.phrases)
        self.assertGreaterEqual(self.phrases["delves into"]["ratio"], lexicon.MIN_RATIO)
        self.assertEqual(self.phrases["delves into"]["generators"], len(GENERATORS))

    def test_drops_one_models_habit_and_shared_topic_words(self):
        self.assertNotIn("tapestry", self.phrases)
        # A topic word both sides use cancels out; only wording around it can stand out.
        self.assertFalse(any(phrase.startswith("topic") and " " not in phrase for phrase in self.phrases))

    def test_shows_a_word_as_the_phrase_it_sits_in(self):
        self.assertIn("delves into", self.phrases)
        self.assertNotIn("delves", self.phrases)

    def test_measures_how_often_it_fires_on_human_test_paragraphs(self):
        measured = self.result["measured"]
        self.assertEqual(measured["human_paragraphs_with_a_term_percent"], 0.0)
        self.assertGreater(measured["ai_paragraphs_with_a_term_percent"], 90)

    def test_matches_whole_words_across_any_spacing(self):
        pattern = lexicon.compile_phrase("delves into")
        self.assertTrue(pattern.search("It Delves\ninto it"))
        self.assertFalse(pattern.search("redelves intobox"))


if __name__ == "__main__":
    unittest.main()
