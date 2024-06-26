# Imports
from datasets import load_dataset
import os
import json

def create_subset(dataset_name, config_name, split, output_file, num_samples=100):
    """
    Create a small subset of a dataset and save it to a JSON file.

    Parameters:
    dataset_name (str): The name of the dataset to load.
    config_name (str): The configuration name of the dataset.
    split (str): The split of the dataset to load (e.g., 'train', 'test').
    output_file (str): The path to the output JSON file where the subset will be saved.
    num_samples (int, optional): The number of samples to include in the subset. Default is 100.

    Returns:
    None
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, config_name, split=split)

    # Take a small subset
    subset = dataset.select(range(num_samples))

    # Save the subset to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset.to_dict(), f, ensure_ascii=False, indent=4)

def get_dataset(file_path, dataset_name='wmt14', config_name='de-en', split='train', num_samples=100):
    """
    Load a dataset from a JSON file, creating a small subset if the file does not exist.

    Parameters:
    file_path (str): The path to the JSON file containing the dataset.
    dataset_name (str, optional): The name of the dataset to load if the file does not exist. Default is 'wmt14'.
    config_name (str, optional): The configuration name of the dataset. Default is 'de-en'.
    split (str, optional): The split of the dataset to load if the file does not exist. Default is 'train'.
    num_samples (int, optional): The number of samples to include in the subset if the file does not exist. Default is 100.

    Returns:
    list: A list of translation pairs from the dataset.
    """
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Creating a small subset...")
        create_subset(dataset_name, config_name, split, file_path, num_samples)
        print(f"Subset saved to {file_path}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['translation']

def pad_sequences(sequences, pad_token=0):
    """
    Pad sequences to the same length with the given pad token.

    Parameters:
    sequences (list): List of sequences to be padded.
    pad_token (int, optional): The token to use for padding. Default is 0.

    Returns:
    list: List of padded sequences.
    """
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [pad_token] * (max_length - len(seq)) for seq in sequences]
    return padded_sequences

if __name__ == "__main__":
    # Define the path to the dataset file
    data_path = os.path.join(os.path.dirname(__file__), '../data/small_wmt14_en_de.json')

    # Load the dataset
    dataset = get_dataset(data_path)

    # Print the first 5 translation pairs for verification
    for pair in dataset[:5]:
        print(f"\nEnglish: {pair['en']},\nGerman: {pair['de']}")
