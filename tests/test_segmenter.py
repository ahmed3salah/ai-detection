"""segmenter.py must cut text exactly as the PHP ManuscriptSegmenter does in production."""
import json
import unittest
from pathlib import Path

import segmenter

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "segmenter"


def shape(segments):
    return [{"kind": s.kind, "section": s.section, "coords": s.coords, "text": s.text} for s in segments]


class MatchesProductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.expected = json.loads((FIXTURES / "expected.json").read_text())

    def test_the_fixture_exercises_every_rule(self):
        sections = [s["section"] for s in self.expected["tei"]]
        self.assertIn("Abstract", sections)
        self.assertIn(None, sections)
        self.assertGreater(sections.count("3 Results"), 1)
        text = " ".join(s["text"] for s in self.expected["tei"])
        self.assertNotIn("Caption text", text)
        self.assertNotIn("A cited work", text)
        self.assertIn("Short opener line.", text)

    def test_tei_matches_the_php_output(self):
        self.assertEqual(shape(segmenter.from_tei((FIXTURES / "manuscript.tei.xml").read_text())), self.expected["tei"])

    def test_plain_text_matches_the_php_output(self):
        self.assertEqual(shape(segmenter.from_text((FIXTURES / "pasted.txt").read_text())), self.expected["text"])

    def test_something_that_is_not_tei_reads_as_nothing(self):
        self.assertEqual(segmenter.from_tei("<html><body>Preparing Space</body></html>"), [])
        self.assertEqual(segmenter.from_tei("not xml"), [])


if __name__ == "__main__":
    unittest.main()
