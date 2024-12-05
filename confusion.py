import json
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load your JSONL file into a DataFrame
file_path = "eval_snli_finetuned_output/eval_predictions.jsonl"  # Replace with the path to your data file
data = []
with open(file_path, "r") as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Extract true labels and predicted labels
true_labels = df["label"].tolist()
predicted_labels = df["predicted_label"].tolist()

# Define class names
class_names = ["Entailment", "Neutral", "Contradiction"]

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)

# Add title and labels
plt.title("Confusion Matrix of SNLI w/ Fine-tuned Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_snli_finetuned.png", bbox_inches="tight")  # Saves as PNG
# Show the plot
plt.show()
