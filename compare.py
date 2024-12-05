import pandas as pd

# File paths
snli_file = "eval_snli_output/eval_predictions.jsonl"
anli_finetuned_file = "eval_anli_finetuned_output/eval_predictions.jsonl"

# Load predictions
snli_data = pd.read_json(snli_file, lines=True)
anli_finetuned_data = pd.read_json(anli_finetuned_file, lines=True)

# Predictions mapping
PREDICTIONS = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Identify errors in SNLI predictions
snli_errors = snli_data[snli_data["label"] != snli_data["predicted_label"]]

# Compare with ANLI finetuned predictions
anli_correct = anli_finetuned_data[
    anli_finetuned_data["label"] == anli_finetuned_data["predicted_label"]
]

# Errors in SNLI that were fixed by ANLI-finetuned
fixed_errors = snli_errors[snli_errors.index.isin(anli_correct.index)]

# Output results
print(f"Total number of errors fixed: {len(fixed_errors)}")
print("=" * 50)

# # Display all fixed errors
# for idx, row in fixed_errors.iterrows():
#     print(f"Row Index: {idx}")
#     print(f"Premise: {row['premise']}")
#     print(f"Hypothesis: {row['hypothesis']}")
#     print(f"Gold Label: {PREDICTIONS[row['label']]}")
#     print(f"Predicted (SNLI): {PREDICTIONS[row['predicted_label']]}")
#     print(
#         f"Fixed (ANLI-Finetuned): Correctly predicted as {PREDICTIONS[row['label']]}"
#     )
#     print("=" * 50)
