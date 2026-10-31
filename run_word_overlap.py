import os
from pathlib import Path

project_folder = Path(__file__).resolve().parent
cache_folder = project_folder / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(cache_folder))

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages


def print_class_scores(class_name, correct_count, predicted_count, actual_count):
    # Start at zero so empty counts do not cause division errors.
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    # Precision: how often our predictions were right.
    if predicted_count > 0:
        precision = correct_count / predicted_count

    # Recall: how many actual answers we found.
    if actual_count > 0:
        recall = correct_count / actual_count

    # F1 combines precision and recall.
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    print(f"\n{class_name}")
    print(f"  Precision : {precision:.2%}")
    print(f"  Recall    : {recall:.2%}")
    print(f"  F1        : {f1:.2%}")


def main():
    print("\nWORD OVERLAP RESULTS")
    print("=" * 45)
    print("Loading facts and evidence...", flush=True)

    # Load the evidence and match it to the facts.
    passages_path = project_folder / "data/passages_bm25_ChatGPT_humfacts.jsonl"
    labels_path = project_folder / "data/dev_labeled_ChatGPT.jsonl"

    passages = read_passages(passages_path)
    examples = read_fact_examples(labels_path, passages)

    if not examples:
        raise ValueError("No labelled facts were found.")

    checker = WordRecallThresholdFactChecker()

    # Keep a separate count for each possible result.
    correct_supported = 0
    correct_not_supported = 0
    wrongly_supported = 0
    missed_supported = 0

    for example in examples:
        actual_label = example.label

        # The coursework counts irrelevant facts as not supported.
        if actual_label == "IR":
            actual_label = "NS"

        predicted_label = checker.predict(example.fact, example.passages)

        if actual_label == "S" and predicted_label == "S":
            correct_supported += 1
        elif actual_label == "NS" and predicted_label == "NS":
            correct_not_supported += 1
        elif predicted_label == "S":
            wrongly_supported += 1
        else:
            missed_supported += 1

    # Work out the overall result.
    total = len(examples)
    correct = correct_supported + correct_not_supported
    errors = wrongly_supported + missed_supported
    accuracy = correct / total

    print(f"\nFacts checked : {total}")
    print(f"Threshold     : {checker.threshold:.2f}")
    print(f"Accuracy      : {accuracy:.2%}")
    print(f"Correct       : {correct}")
    print(f"Errors        : {errors}")

    # Count the predicted and actual supported answers.
    predicted_supported = correct_supported + wrongly_supported
    actual_supported = correct_supported + missed_supported

    print_class_scores(
        "SUPPORTED FACTS", correct_supported, predicted_supported, actual_supported
    )

    # Do the same for not-supported answers.
    predicted_not_supported = correct_not_supported + missed_supported
    actual_not_supported = correct_not_supported + wrongly_supported

    print_class_scores(
        "NOT-SUPPORTED FACTS", correct_not_supported,
        predicted_not_supported, actual_not_supported
    )

    print("\nPREDICTION BREAKDOWN")
    print(f"  Correctly marked supported     : {correct_supported}")
    print(f"  Correctly marked not supported : {correct_not_supported}")
    print(f"  Wrongly marked supported       : {wrongly_supported}")
    print(f"  Missed supported facts         : {missed_supported}")
    print("=" * 45)


if __name__ == "__main__":
    main()
