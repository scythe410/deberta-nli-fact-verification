# Repeat the comparisons in docs/experiments.md.
# Run this file from the COURSEWORK folder.

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages


# 1. Load the facts and evidence.
passages = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages)


# 2. List the eight methods to compare.
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

print("word overlap comparisons")
print("each method tries cutoffs from 0.01 to 1.00.\n")

for method, remove_common_words, combine_passages in methods:
    checker = WordRecallThresholdFactChecker()

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

    print(method.lower())
    print(f"  remove common words: {'yes' if remove_common_words else 'no'}")
    print(f"  combine passages: {'yes' if combine_passages else 'no'}")
    print(f"  cutoff: {best_threshold:.2f}")
    print(f"  correct: {best_correct}/{len(examples)} ({accuracy:.2%})")
    print()


# 5. Show how the current checker performs at a few cutoffs.
print("results at different cutoffs")
checker = WordRecallThresholdFactChecker()

for threshold in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
    checker.threshold = threshold
    correct = 0

    for example in examples:
        actual_label = example.label

        if actual_label == "IR":
            actual_label = "NS"

        prediction = checker.predict(example.fact, example.passages)

        if prediction == actual_label:
            correct += 1

    accuracy = correct / len(examples)
    print(f"{threshold:.2f}: {correct}/{len(examples)} ({accuracy:.2%})")
