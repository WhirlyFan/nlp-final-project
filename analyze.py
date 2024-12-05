import os

import pandas as pd

input_dir = "eval_anli_output/"
file_name = "eval_predictions.jsonl"
file_path = os.path.join(input_dir, file_name)

# Load predictions
data = pd.read_json(file_path, lines=True)

print(f"File path: {file_path}")
PREDICTIONS = {0: "entailment", 1: "neutral", 2: "contradiction"}

errors = data[data["label"] != data["predicted_label"]]
correct = data[data["label"] == data["predicted_label"]]


def display_row_info(df, row_number, error_type):
    row = df.iloc[row_number]
    print("=" * 50)
    print(f"Error Type: {error_type}")
    print(f"Row Index: {row.name}")
    print(f"Premise: {row['premise']}")
    print(f"Hypothesis: {row['hypothesis']}")
    print(f"Gold Label: {PREDICTIONS[row['label']]}")
    print(f"Predicted: {PREDICTIONS[row['predicted_label']]}")


negation_errors = errors[
    errors["hypothesis"].str.contains(r"\b(?:not|never|no)\b", case=False)
]
print(f"Number of negation errors: {len(negation_errors)}")

logical_words = ["all", "some", "every", "none", "if", "then"]
logical_errors = errors[
    errors["premise"].str.contains("|".join(logical_words), case=False)
    | errors["hypothesis"].str.contains("|".join(logical_words), case=False)
]
print(f"Number of logical errors: {len(logical_errors)}")


numerical_errors = errors[
    errors["premise"].str.contains(r"\d") | errors["hypothesis"].str.contains(r"\d")
]
print(f"Number of numerical errors: {len(numerical_errors)}")

temporal_words = ["before", "after", "during", "while"]
temporal_errors = errors[
    errors["premise"].str.contains("|".join(temporal_words), case=False)
    | errors["hypothesis"].str.contains("|".join(temporal_words), case=False)
]
print(f"Number of temporal errors: {len(temporal_errors)}")

print("=" * 50)
print(f"Accuracy: {len(correct)}/{len(data)} ({len(correct) / len(data) * 100:.2f}%)")
print(f"Total number of errors: {len(errors)}")

# Display the information of the first temporal error
# display_row_info(temporal_errors, 1, "Temporal Error")
# display_row_info(numerical_errors, 1, "Numerical Error")
# display_row_info(logical_errors, 4, "Logical Error")
# display_row_info(logical_errors, 1, "Logical Error")
# display_row_info(logical_errors, 7, "Logical Error")
# display_row_info(negation_errors, 1, "Negation Error")
# display_row_info(errors, random.randint(0, len(errors) - 1), "General Error")
# display_row_info(correct, random.randint(0, len(correct) - 1), "Correct Prediction")

# def random_sample_logical(data, output_file, n=10, label=None):
#     if label is not None:
#         data = data[data["label"] == label]
#     logical_samples = data[
#         data["premise"].str.contains("|".join(logical_words), case=False)
#         | data["hypothesis"].str.contains("|".join(logical_words), case=False)
#     ]
#     random_samples = logical_samples.sample(n=min(len(logical_samples), n), random_state=42)
#     random_samples.to_json(output_file, orient="records", lines=True)
#     return random_samples

# random_sample_logical(errors, output_dir + "logical_sample.jsonl", n=10)

# Function to randomly sample one unique row for each label type
# def sample_unique_errors(errors_df, already_sampled):
#     samples = []
#     for label in range(3):  # 0: entailment, 1: neutral, 2: contradiction
#         label_errors = errors_df[
#             (errors_df["label"] == label) & (~errors_df.index.isin(already_sampled))
#         ]
#         if not label_errors.empty:
#             # Randomly sample one row
#             sample = label_errors.sample(1)
#             samples.append(sample.iloc[0])
#             already_sampled.add(sample.index[0])  # Track the sampled index
#         else:
#             print(f"No more unique errors available for label {PREDICTIONS[label]} ({label}).")
#     return samples

# # Keep track of already sampled indices
# already_sampled_indices = set()

# # Get a new set of samples
# samples = sample_unique_errors(logical_errors, already_sampled_indices)

# # Display the new set
# for sample in samples:
#     display_row_info(pd.DataFrame([sample]), 0, "Logical Error")

# Function to find rows by their indices
# def find_rows_by_indices(df, indices):
#     return df.loc[indices]

# # Example indices to retrieve
# row_indices = [9433, 730, 7777]

# # Retrieve rows using the indices
# selected_rows = find_rows_by_indices(errors, row_indices)

# # Display the retrieved rows
# for idx, row in selected_rows.iterrows():
#     display_row_info(pd.DataFrame([row]), 0, "Logical Error")
