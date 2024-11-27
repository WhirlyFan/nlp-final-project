import pandas as pd

# Load predictions
data = pd.read_json("eval_output/eval_predictions.jsonl", lines=True)

PREDICTIONS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# # Display the first few rows
# print(data.head())

# # Check the number of rows and columns
# print(f"{data.shape=}")

# # Display column names
# print(f"{data.columns=}")

# # Check label distribution
# print(f"{data['label'].value_counts()=}")

# Filter misclassified examples
errors = data[data['label'] != data['predicted_label']]

# # Display some errors
# print(errors[['premise', 'hypothesis', 'label', 'predicted_label']].head())

negation_errors = errors[errors['hypothesis'].str.contains(r'\b(?:not|never|no)\b', case=False)]
print(f"Number of negation errors: {len(negation_errors)}")

# Display a few examples
print(negation_errors[['premise', 'hypothesis', 'label', 'predicted_label']].head())

def display_row_info(df, row_number):
    """
    Displays the information of a specific row in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - row_number (int): The row index to display.

    Returns:
    - None: Prints the row's content in a readable format.
    """
    if row_number < 0 or row_number >= len(df):
        print(f"Row number {row_number} is out of range. Please provide a valid row index (0 to {len(df) - 1}).")
        return

    row = df.iloc[row_number]  # Get the specific row
    print(f"Row {row_number} Information:")
    for column, value in row.items():
        print(f"{column}: {value}")

# Display the information of the first negation error
display_row_info(negation_errors, 0)
