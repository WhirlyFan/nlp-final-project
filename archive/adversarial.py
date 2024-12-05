import json
import os
import random
import urllib.request
import zipfile

import nltk
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine

# Download required NLTK data
nltk.download("wordnet")
nltk.download("omw-1.4")  # Open Multilingual WordNet
nltk.download("averaged_perceptron_tagger")

# Load SpaCy model for consistency
nlp = spacy.load("en_core_web_sm")

PREDICTIONS = {0: "entailment", 1: "neutral", 2: "contradiction"}
EXCLUDED_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "has",
    "have",
    "had",
    "does",
    "do",
    "did",
}
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = "glove"
GLOVE_FILE = "glove.6B.300d.txt"


def download_and_extract_glove():
    """Download and extract GloVe embeddings if not already present."""
    if not os.path.exists(GLOVE_DIR):
        os.makedirs(GLOVE_DIR)

    glove_zip = os.path.join(GLOVE_DIR, "glove.6B.zip")
    glove_txt_path = os.path.join(GLOVE_DIR, GLOVE_FILE)

    if not os.path.exists(glove_txt_path):
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(GLOVE_URL, glove_zip)
        print("Extracting GloVe embeddings...")
        with zipfile.ZipFile(glove_zip, "r") as zip_ref:
            zip_ref.extractall(GLOVE_DIR)
        os.remove(glove_zip)
    else:
        print("GloVe embeddings already downloaded and extracted.")

    return glove_txt_path


def load_glove_embeddings(glove_file_path):
    """Load GloVe embeddings into a dictionary."""
    embeddings = {}
    print("Loading GloVe embeddings...")
    with open(glove_file_path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            embeddings[word] = vector
    print("GloVe embeddings loaded.")
    return embeddings


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is not None and vec2 is not None:
        return 1 - cosine(vec1, vec2)
    else:
        return None


def filter_by_similarity(word, replacements, embeddings, threshold=0.3):
    """Filter replacements based on cosine similarity threshold."""
    word_vector = embeddings.get(word.lower())
    if word_vector is None:
        return []
    filtered_replacements = []
    for replacement in replacements:
        replacement_vector = embeddings.get(replacement.lower())
        if replacement_vector is not None:
            similarity = cosine_similarity(word_vector, replacement_vector)
            if similarity is not None and similarity >= threshold:
                filtered_replacements.append(replacement)
    return filtered_replacements


def get_synonyms_and_hypernyms(word, pos_tag):
    """Get synonyms or hypernyms of a word."""
    synonyms = set()
    for syn in wn.synsets(word, pos=pos_tag):
        # Add synonyms
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
        # Add hypernyms
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                hypernym_word = lemma.name().replace("_", " ")
                if hypernym_word.lower() != word.lower():
                    synonyms.add(hypernym_word)
    return list(synonyms)


def pos_tag_spacy_to_wordnet(spacy_tag):
    """Convert SpaCy POS tags to WordNet POS tags."""
    if spacy_tag.startswith("N"):
        return wn.NOUN
    elif spacy_tag.startswith("V"):
        return wn.VERB
    elif spacy_tag.startswith("J"):  # Adjective
        return wn.ADJ
    elif spacy_tag.startswith("R"):  # Adverb
        return wn.ADV
    return None


def generate_adversarial_hypotheses(premise, original_hypothesis, label, embeddings):
    """Generate adversarial hypotheses by replacing one word in the hypothesis with a synonym."""
    doc = nlp(original_hypothesis)
    generated_samples = []

    for token in doc:
        if token.is_alpha and token.text.lower() not in EXCLUDED_WORDS:
            wordnet_pos = pos_tag_spacy_to_wordnet(token.tag_)
            if not wordnet_pos:
                continue

            word = token.text
            replacements = get_synonyms_and_hypernyms(word, wordnet_pos)
            if replacements:
                # Filter replacements by cosine similarity
                replacements = filter_by_similarity(
                    word, replacements, embeddings, threshold=0.3
                )
                if replacements:
                    replacement = random.choice(replacements)
                    # Replace one word and construct the new hypothesis
                    modified_hypothesis = original_hypothesis.replace(
                        word, replacement, 1
                    )
                    generated_samples.append(
                        {
                            "premise": premise,
                            "original_hypothesis": original_hypothesis,
                            "hypothesis": modified_hypothesis,
                            "label": label,
                        }
                    )
                    # Only replace one word per hypothesis
                    break

    return generated_samples


def process_file(input_file, embeddings):
    """Process the input file and generate adversarial examples."""
    samples = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            premise = data.get("premise")
            original_hypothesis = data.get("hypothesis")
            label = data.get("label")
            if premise and original_hypothesis and label:
                samples.extend(
                    generate_adversarial_hypotheses(
                        premise, original_hypothesis, label, embeddings
                    )
                )
    return samples


def main():
    # Ensure GloVe embeddings are available
    glove_file_path = download_and_extract_glove()
    embeddings = load_glove_embeddings(glove_file_path)

    # File paths
    sample_file = "data_samples/neutral_sample.jsonl"
    output_file = "data_samples/adversarial_set.jsonl"

    # Process each file
    print("Processing samples...")
    random_sample = process_file(sample_file, embeddings)

    with open(output_file, "w") as f:
        for sample in random_sample:
            f.write(json.dumps(sample) + "\n")

    print(f"Adversarial set created at {output_file}")


if __name__ == "__main__":
    main()
