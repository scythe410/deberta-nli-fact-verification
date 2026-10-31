import unittest

from factcheck import WordRecallThresholdFactChecker


class WordOverlapTests(unittest.TestCase):
    def setUp(self):
        # Start each test with a fresh checker.
        self.checker = WordRecallThresholdFactChecker()

    def test_cleaning_keeps_numbers_and_negation(self):
        # Remove tags and common words, but keep "not" and the year.
        words = self.checker.get_words("<s>The ACTOR was not born in 1980.</s>")

        self.assertEqual(words, {"actor", "not", "born", "1980"})

    def test_empty_inputs_are_not_supported(self):
        # Empty evidence or no useful fact words should give NS.
        self.assertEqual(self.checker.predict("An actor", []), "NS")
        self.assertEqual(self.checker.predict("", [{"text": "An actor"}]), "NS")
        self.assertEqual(self.checker.predict("the is", [{"text": "the is"}]), "NS")

    def test_passage_title_is_not_evidence(self):
        # Only the passage text should count as evidence.
        passages = [{"title": "actor", "text": "A singer"}]

        self.assertEqual(self.checker.predict("actor", passages), "NS")

    def test_repeated_words_do_not_increase_score(self):
        # Repeating "actor" still matches only one of the two fact words.
        passages = [{"text": "actor actor actor"}]
        score = self.checker.get_overlap_score("actor singer", passages)

        self.assertEqual(score, 0.5)

    def test_words_can_match_across_passages(self):
        # Matching words can come from different passages.
        passages = [{"text": "Marianne"}, {"text": "McAndrew actress"}]

        self.assertEqual(self.checker.predict("Marianne McAndrew is an actress", passages), "S")

    def test_threshold_includes_equal_score(self):
        # Three of four words match: a score of exactly 0.75.
        passages = [{"text": "actor singer dancer"}]

        self.assertEqual(self.checker.predict("actor singer dancer writer", passages), "S")

        # Raise the cutoff slightly. The same fact should now give NS.
        self.checker.threshold = 0.76

        self.assertEqual(self.checker.predict("actor singer dancer writer", passages), "NS")


if __name__ == "__main__":
    unittest.main()
