import os
import json
import random
import urllib.request
import zipfile
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import spacy
from scipy.spatial.distance import cosine

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual WordNet
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')  # For tokenization

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Load SpaCy model for consistency
nlp = spacy.load('en_core_web_sm')

PREDICTIONS = {0: "entailment", 1: "neutral", 2: "contradiction"}
EXCLUDED_WORDS = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "does", "do", "did",
    "the", "a", "an"
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
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    print("GloVe embeddings loaded.")
    return embeddings


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is not None and vec2 is not None:
        return 1 - cosine(vec1, vec2)
    else:
        return None


def get_synonyms(word, pos_tag):
    """Get synonyms of a word."""
    synonyms = set()
    for syn in wn.synsets(word, pos=pos_tag):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


def get_antonyms(word, pos_tag):
    """Get antonyms of a word from its primary synset."""
    antonyms = set()
    for syn in wn.synsets(word, pos=pos_tag):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonym = ant.name().replace('_', ' ')
                if antonym.lower() != word.lower():
                    antonyms.add(antonym)
    return list(antonyms)


def pos_tag_spacy_to_wordnet(spacy_tag):
    """Convert SpaCy POS tags to WordNet POS tags."""
    if spacy_tag.startswith('N'):
        return wn.NOUN
    elif spacy_tag.startswith('V'):
        return wn.VERB
    elif spacy_tag.startswith('J'):  # Adjective
        return wn.ADJ
    elif spacy_tag.startswith('R'):  # Adverb
        return wn.ADV
    return None


def calculate_word_importance(sentence, token, embeddings):
    """
    Calculate the semantic importance of a word based on its average similarity
    with all other words in the sentence.
    """
    word_vector = embeddings.get(token.text.lower())
    if word_vector is None:
        return 0  # If no embedding, return 0 importance

    total_similarity = 0
    count = 0
    for other_token in sentence:
        if other_token.text.lower() != token.text.lower() and other_token.text.lower() in embeddings:
            other_vector = embeddings.get(other_token.text.lower())
            similarity = cosine_similarity(word_vector, other_vector)
            if similarity is not None:
                total_similarity += similarity
                count += 1

    if count == 0:
        return 0  # No other embeddings to compare to

    # Higher average similarity implies higher importance
    return total_similarity / count


def select_replacement_word(doc, embeddings):
    """
    Select the word to replace based on semantic importance.
    """
    candidate_tokens = []
    for token in doc:
        if token.is_alpha and token.text.lower() not in EXCLUDED_WORDS:
            wordnet_pos = pos_tag_spacy_to_wordnet(token.tag_)
            if wordnet_pos:
                candidate_tokens.append(token)

    if not candidate_tokens:
        return None  # No replaceable tokens

    # Rank tokens by importance (higher importance = more central to sentence)
    ranked_tokens = sorted(
        candidate_tokens,
        key=lambda token: calculate_word_importance(doc, token, embeddings),
        reverse=True,
    )

    # Return the most important token
    return ranked_tokens[0]


def adjust_replacement_form(replacement, word_token):
    """
    Adjust the replacement word to match the morphological features of the original word.
    """
    # Get the POS tag of the original word
    word_tag = word_token.tag_  # e.g., 'VBD', 'NN', etc.
    pos = word_token.pos_  # e.g., 'VERB', 'NOUN', etc.

    # Simple rules for inflection without lemminflect
    if pos == 'VERB':
        # Use NLTK's conjugation functions (limited support)
        replacement_inflected = conjugate_verb(replacement, word_tag)
    elif pos == 'NOUN':
        # Adjust noun number (singular/plural)
        replacement_inflected = inflect_noun(replacement, word_tag)
    else:
        # For adjectives and adverbs, we can use basic rules
        replacement_inflected = replacement

    # Preserve capitalization
    if word_token.text[0].isupper():
        replacement_inflected = replacement_inflected.capitalize()

    return replacement_inflected


def conjugate_verb(verb, tag):
    """Conjugate verb to match the original verb's tense."""
    verb = verb.lower()
    lemma = lemmatizer.lemmatize(verb, 'v')
    if tag == 'VBD' or tag == 'VBN':
        # Past tense or past participle
        if lemma == verb:
            # Regular verb
            if verb.endswith('e'):
                return verb + 'd'
            elif verb.endswith('y'):
                return verb[:-1] + 'ied'
            else:
                return verb + 'ed'
        else:
            # Irregular verb
            return lemma  # Best effort
    elif tag == 'VBZ':
        # Third person singular present
        if verb.endswith(('s', 'x', 'z', 'o', 'ch', 'sh')):
            return verb + 'es'
        elif verb.endswith('y'):
            return verb[:-1] + 'ies'
        else:
            return verb + 's'
    elif tag == 'VBG':
        # Present participle / gerund
        if verb.endswith('e') and len(verb) > 2:
            return verb[:-1] + 'ing'
        else:
            return verb + 'ing'
    else:
        # Base form or other tenses
        return verb


def inflect_noun(noun, tag):
    """Inflect noun to match singular or plural form."""
    noun = noun.lower()
    if tag == 'NNS' or tag == 'NNPS':
        # Plural form
        if noun.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return noun + 'es'
        elif noun.endswith('y') and noun[-2] not in 'aeiou':
            return noun[:-1] + 'ies'
        else:
            return noun + 's'
    else:
        # Singular form
        return noun


def generate_adversarial_hypotheses(premise, hypothesis, label, embeddings):
    """Generate adversarial hypotheses by replacing one word in the premise."""
    doc = nlp(premise)
    generated_samples = []

    # Select the most important word to replace
    token_to_replace = select_replacement_word(doc, embeddings)
    if not token_to_replace:
        # Use the original hypothesis if no token to replace
        generated_samples.append({
            "premise": premise,
            "hypothesis": hypothesis if hypothesis else premise,
            "label": label,
        })
        return generated_samples

    wordnet_pos = pos_tag_spacy_to_wordnet(token_to_replace.tag_)
    word = token_to_replace.text

    # Initialize replacements set
    replacements = set()

    if label == 0:  # Entailment
        replacements.update(get_synonyms(word, wordnet_pos))
    elif label == 2:  # Contradiction
        replacements.update(get_antonyms(word, wordnet_pos))
    else:
        # Use the original hypothesis for neutral or other labels
        generated_samples.append({
            "premise": premise,
            "hypothesis": hypothesis if hypothesis else premise,
            "label": label,
        })
        return generated_samples

    # Note: Removed filtering using have_same_root and is_substring
    filtered_replacements = list(replacements)

    if not filtered_replacements:
        # Use the original hypothesis if no valid replacements
        generated_samples.append({
            "premise": premise,
            "hypothesis": hypothesis if hypothesis else premise,
            "label": label,
        })
        return generated_samples

    if label == 2:  # Contradiction
        # For contradictions, select antonym with lowest cosine similarity
        similarity_scores = []
        word_vector = embeddings.get(word.lower())
        if word_vector is None:
            # Use the original hypothesis if word vector not found
            generated_samples.append({
                "premise": premise,
                "hypothesis": hypothesis if hypothesis else premise,
                "label": label,
            })
            return generated_samples
        for replacement in filtered_replacements:
            replacement_vector = embeddings.get(replacement.lower())
            if replacement_vector is not None:
                similarity = cosine_similarity(word_vector, replacement_vector)
                if similarity is not None:
                    similarity_scores.append((replacement, similarity))
        if similarity_scores:
            similarity_scores.sort(key=lambda x: x[1])  # Sort by similarity
            replacement = similarity_scores[0][0]
        else:
            # Use the original hypothesis if no valid embeddings for replacements
            generated_samples.append({
                "premise": premise,
                "hypothesis": hypothesis if hypothesis else premise,
                "label": label,
            })
            return generated_samples
    else:  # Entailment
        # For entailments, pick a synonym with high cosine similarity
        filtered_replacements = filter_by_similarity(word, filtered_replacements, embeddings, threshold=0.1)
        if not filtered_replacements:
            # Use the original hypothesis if no valid replacements after filtering
            generated_samples.append({
                "premise": premise,
                "hypothesis": hypothesis if hypothesis else premise,
                "label": label,
            })
            return generated_samples
        replacement = random.choice(filtered_replacements)

    # Adjust 'replacement' to match the morphological features of 'word'
    replacement_inflected = adjust_replacement_form(replacement, token_to_replace)

    # Replace the selected word
    modified_premise = premise.replace(word, replacement_inflected, 1)
    generated_samples.append({
        "premise": premise,
        "hypothesis": modified_premise,
        "label": label,
    })

    return generated_samples


def filter_by_similarity(word, replacements, embeddings, threshold=0.1):
    """Filter replacements based on cosine similarity threshold."""
    word_vector = embeddings.get(word.lower())
    if word_vector is None:
        return replacements  # Return all replacements if word vector not found
    filtered_replacements = []
    for replacement in replacements:
        replacement_vector = embeddings.get(replacement.lower())
        if replacement_vector is not None:
            similarity = cosine_similarity(word_vector, replacement_vector)
            if similarity is not None and similarity >= threshold:
                filtered_replacements.append(replacement)
    return filtered_replacements


def process_file(input_file, label, embeddings):
    """Process the input file and generate adversarial examples."""
    samples = []
    skipped = 0
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            premise = data.get("premise")
            hypothesis = data.get("hypothesis")  # Get original hypothesis if available
            if premise:
                generated = generate_adversarial_hypotheses(premise, hypothesis, label, embeddings)
                if generated:
                    samples.extend(generated)
                else:
                    skipped += 1
    print(f"Skipped {skipped} examples")
    return samples


def main():
    # Ensure GloVe embeddings are available
    glove_file_path = download_and_extract_glove()
    embeddings = load_glove_embeddings(glove_file_path)

    # File paths
    contradiction_file = "anli_data_samples/contradiction_sample.jsonl"
    entailment_file = "anli_data_samples/entailment_sample.jsonl"
    output_file = "anli_data_samples/adversarial_set.jsonl"

    # Process each file
    print("Processing contradiction samples...")
    contradiction_samples = process_file(
        contradiction_file, 2, embeddings
    )  # Label for contradiction

    print("Processing entailment samples...")
    entailment_samples = process_file(
        entailment_file, 0, embeddings
    )  # Label for entailment

    # Combine samples and write to output
    all_samples = contradiction_samples + entailment_samples
    with open(output_file, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Adversarial set created at {output_file}")


if __name__ == "__main__":
    main()
