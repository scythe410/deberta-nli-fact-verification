"""Select ten errors of each type from the final entailment predictions."""

import json
import random
from pathlib import Path

project = Path(__file__).resolve().parent
predictions_path = project / "outputs/entailment_final_predictions.jsonl"
rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
random_generator = random.Random(3604)
selected = []

for error_type, gold, prediction in [("FP", "NS", "S"), ("FN", "S", "NS")]:
    indices = [
        index for index, row in enumerate(rows)
        if row["gold_label"] == gold and row["prediction"] == prediction
    ]
    sample = sorted(random_generator.sample(indices, min(10, len(indices))))
    print(f"{error_type}: {len(indices)} total, {len(sample)} selected")
    for index in sample:
        selected.append({"row": index, "error_type": error_type, **rows[index]})
        print(f"  {index}: {rows[index]['fact']}")

output = project / "outputs/part3_sample.jsonl"
with output.open("w", encoding="utf-8") as file:
    for row in selected:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Saved the selected claims and full passages to {output}")
