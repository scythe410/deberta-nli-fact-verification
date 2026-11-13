"""Show known limits. Passing these tests does not mean the claims are true."""

import unittest

from factcheck import WordRecallThresholdFactChecker


class WordOverlapLimitTests(unittest.TestCase):
    def setUp(self):
        self.checker = WordRecallThresholdFactChecker()

    def test_negation_can_still_pass(self):
        fact = "Marianne McAndrew is not an actress."
        evidence = [{"text": "Marianne McAndrew is an actress."}]

        # Wrong factual answer: three out of four words still match.
        self.assertEqual(self.checker.get_overlap_score(fact, evidence), 0.75)
        self.assertEqual(self.checker.predict(fact, evidence), "S")

    def test_wrong_year_can_still_pass(self):
        fact = "Marianne McAndrew was born in 1980."
        evidence = [{"text": "Marianne McAndrew was born in 1942."}]

        # The name and 'born' outweigh the wrong year.
        self.assertEqual(self.checker.predict(fact, evidence), "S")

    def test_contraction_does_not_become_not(self):
        words = self.checker.get_words("isn't")

        self.assertEqual(words, {"isn", "t"})
        self.assertNotIn("not", words)

    def test_accented_name_loses_a_letter(self):
        words = self.checker.get_words("José")

        # ASCII tokenisation produces "jos"; stemming reduces it to "jo".
        self.assertEqual(words, {"jo"})
        original = WordRecallThresholdFactChecker(use_stemming=False)
        self.assertEqual(original.get_words("José"), {"jos"})


if __name__ == "__main__":
    unittest.main()
