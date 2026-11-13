"""Compare entailment settings or export the final checker predictions."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import EntailmentFactChecker, EntailmentModel, WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages


def predict_from_scores(candidates, cutoff, minimum_overlap):
    best_evidence = None
    best_probability = None
    calls = 0

    for sentence, label, probability, overlap in candidates:
        if overlap < minimum_overlap:
            continue
        calls += 1
        if best_probability is None or probability > best_probability:
            best_evidence = sentence
            best_probability = probability

        supported = label == "entailment" if cutoff is None else probability >= cutoff
        if supported:
            return "S", sentence, probability, calls

    return "NS", best_evidence, best_probability, calls


def save_predictions(output, rows):
    output.parent.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_final_predictions(checker, examples, output):
    rows = []
    correct = false_positives = false_negatives = 0
    for example in tqdm(examples, desc="checking claims"):
        gold = "NS" if example.label == "IR" else example.label
        prediction = checker.predict(example.fact, example.passages)
        correct += prediction == gold
        false_positives += prediction == "S" and gold == "NS"
        false_negatives += prediction == "NS" and gold == "S"
        rows.append({
            "fact": example.fact,
            "original_label": example.label,
            "gold_label": gold,
            "prediction": prediction,
            "passages": example.passages,
        })

    save_predictions(output, rows)
    print(f"accuracy: {correct}/{len(rows)} ({correct / len(rows):.2%})")
    print(f"false positives: {false_positives}")
    print(f"false negatives: {false_negatives}")
    print(f"saved final predictions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refine", action="store_true", help="Compare overlap cutoffs near 0.10.")
    modes.add_argument("--final", action="store_true", help="Export predictions from the final checker.")
    args = parser.parse_args()
    overlap_cutoffs = (0.0, 0.1, 0.2, 0.3)
    decision_cutoffs = (None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    if args.refine:
        overlap_cutoffs = tuple(step / 100 for step in range(6, 16))
        decision_cutoffs = (None,)

    # Limit CPU threads for these small, single-pair model calls.
    torch.set_num_threads(4)
    project = Path(__file__).resolve().parent
    passages = read_passages(project / "data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples(project / "data/dev_labeled_ChatGPT.jsonl", passages)

    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    entailment_model = EntailmentModel(model, tokenizer)
    checker = EntailmentFactChecker(entailment_model)
    if args.final:
        export_final_predictions(checker, examples, project / "outputs/entailment_final_predictions.jsonl")
        return

    overlap_checker = WordRecallThresholdFactChecker()

    # Score each sentence once so every comparison uses the same model outputs.
    start = perf_counter()
    scored_examples = []
    for example in tqdm(examples, desc="scoring evidence"):
        candidates = []
        for sentence in checker.evidence_sentences(example.passages):
            overlap = overlap_checker.get_overlap_score(example.fact, [{"text": sentence}])
            if overlap < overlap_cutoffs[0]:
                continue
            label, probability = entailment_model.check_entailment(sentence, example.fact)
            candidates.append((sentence, label, probability, overlap))
        scored_examples.append((example, candidates))

    scored_pairs = sum(len(candidates) for _, candidates in scored_examples)
    print(f"scored {scored_pairs} pairs in {perf_counter() - start:.1f}s on cpu")
    print("cutoff   min overlap   correct   accuracy   fp   fn   model calls*")
    export_rows = []
    best_result = (-1, 0)
    for minimum_overlap in overlap_cutoffs:
        for cutoff in decision_cutoffs:
            rows = []
            correct = false_positives = false_negatives = calls = 0
            for example, candidates in scored_examples:
                gold = "NS" if example.label == "IR" else example.label
                prediction, evidence, probability, example_calls = predict_from_scores(
                    candidates, cutoff, minimum_overlap
                )
                correct += prediction == gold
                false_positives += prediction == "S" and gold == "NS"
                false_negatives += prediction == "NS" and gold == "S"
                calls += example_calls
                rows.append({
                    "fact": example.fact,
                    "original_label": example.label,
                    "gold_label": gold,
                    "prediction": prediction,
                    "entailment_probability": probability,
                    "evidence_sentence": evidence,
                    "minimum_overlap": minimum_overlap,
                })
            rule = "argmax" if cutoff is None else f"{cutoff:.2f}"
            print(
                f"{rule:8} {minimum_overlap:11.2f} {correct:9d} {correct / len(examples):10.2%}"
                f" {false_positives:4d} {false_negatives:4d} {calls:14d}"
            )
            if args.refine and (correct, -calls) > best_result:
                best_result = (correct, -calls)
                export_rows = rows
            if cutoff is None and minimum_overlap == 0:
                export_rows = rows
    print("*model calls count sentences checked up to the first one that supports the claim")

    filename = "entailment_refined_predictions.jsonl" if args.refine else "entailment_baseline_predictions.jsonl"
    output = project / "outputs" / filename
    save_predictions(output, export_rows)
    print(f"saved predictions with a minimum overlap of {export_rows[0]['minimum_overlap']:.2f} to {output}")


if __name__ == "__main__":
    main()
