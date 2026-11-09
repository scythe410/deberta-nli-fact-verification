# Repeat the comparisons in docs/experiments.md.
# Run this file from the COURSEWORK folder.

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages


# 1. Load the facts and evidence.
passages = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages)


# 2. Compare scoring, common-word removal and passage handling.
# Each row means: score type, remove common words, combine passages.
methods = [
    ("Recall", False, False),
    ("Jaccard", False, False),
    ("Recall", False, True),
    ("Jaccard", False, True),
    ("Recall", True, False),
    ("Jaccard", True, False),
    ("Recall", True, True),
    ("Jaccard", True, True),
]

print("METHOD COMPARISONS")
print("Each method tries thresholds from 0.01 to 1.00.\n")

for method, remove_common_words, combine_passages in methods:
    checker = WordRecallThresholdFactChecker(use_stemming=False)

    if not remove_common_words:
        checker.stop_words = set()

    # 3. Work out one score for each fact.
    scores = []

    for example in examples:
        fact_words = checker.get_words(example.fact)
        evidence_sets = []

        for passage in example.passages:
            words = checker.get_words(passage["text"])
            evidence_sets.append(words)

        if combine_passages:
            combined_words = set()

            for words in evidence_sets:
                combined_words.update(words)

            evidence_sets = [combined_words]

        best_score = 0.0

        for words in evidence_sets:
            matching_words = fact_words & words
            total = len(fact_words)

            if method == "Jaccard":
                all_words = fact_words | words
                total = len(all_words)

            score = 0.0

            if total > 0:
                score = len(matching_words) / total

            best_score = max(best_score, score)

        scores.append(best_score)

    # 4. Try each cutoff and keep the one with the most correct answers.
    best_correct = -1
    best_threshold = 0.0

    for step in range(1, 101):
        threshold = step / 100
        correct = 0

        for score, example in zip(scores, examples):
            predicted_supported = score >= threshold
            actually_supported = example.label == "S"

            # Both NS and IR count as not supported.
            if predicted_supported == actually_supported:
                correct += 1

        # If results tie, keep the larger cutoff.
        if correct >= best_correct:
            best_correct = correct
            best_threshold = threshold

    accuracy = best_correct / len(examples)

    print(method)
    print(f"  Remove common words: {remove_common_words}")
    print(f"  Combine passages: {combine_passages}")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Correct: {best_correct}/{len(examples)} ({accuracy:.2%})")
    print()


# 5. Compare stemming with the original method at every cutoff.
print("STEMMING AND THRESHOLD COMPARISON")
print("Common words removed; recall over combined passages.")

for use_stemming in [False, True]:
    checker = WordRecallThresholdFactChecker(use_stemming=use_stemming)
    scores = [checker.get_overlap_score(e.fact, e.passages) for e in examples]
    best_correct = -1
    best_threshold = 0
    print(f"\nStemming: {use_stemming}")
    print("Cutoff  Correct  False positives  False negatives  Accuracy")

    for step in range(1, 101):
        threshold = step / 100
        false_positives = 0
        false_negatives = 0
        for score, example in zip(scores, examples):
            if score >= threshold and example.label != "S":
                false_positives += 1
            if score < threshold and example.label == "S":
                false_negatives += 1
        correct = len(examples) - false_positives - false_negatives
        if correct >= best_correct:
            best_correct = correct
            best_threshold = threshold
        if step in [50, 60, 65, 70, 75, 80, 85, 90, 100]:
            print(
                f"{threshold:.2f}    {correct:3d}      {false_positives:3d}"
                f"              {false_negatives:3d}           {correct / len(examples):.2%}"
            )
    print(f"Best tested cutoff: {best_threshold:.2f}; correct: {best_correct}/{len(examples)}")
