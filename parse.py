import os
import random

import pandas as pd

# Load predictions
data = pd.read_json("eval_output/eval_predictions.jsonl", lines=True)

# Create a directory for data samples
output_dir = "anli_data_samples/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


PREDICTIONS = {0: "entailment", 1: "neutral", 2: "contradiction"}

errors = data[data["label"] != data["predicted_label"]]
correct = data[data["label"] == data["predicted_label"]]

def random_sample(data, output_file, n=None, label=None):
    if label is not None:
        data = data[data["label"] == label]
    if n is None:
        n = len(data)
    random_samples = data.sample(n=min(len(data), n), random_state=42)
    random_samples.to_json(output_file, orient="records", lines=True)
    return random_samples


random_sample(data, output_dir + "random_sample.jsonl", n=1000)
random_sample(data, output_dir + "entailment_sample.jsonl", n=500, label=0)
random_sample(data, output_dir + "contradiction_sample.jsonl", n=500, label=2)
random_sample(data, output_dir + "neutral_sample.jsonl", n=500, label=1)
