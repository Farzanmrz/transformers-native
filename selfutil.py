import json
import os
from math import sqrt

import torch
import torch.nn.functional as F
from datasets import load_dataset


def get_dataset(data_dir='data/', dataset_name='wmt14', config_name='de-en', split='train', num_samples=100, create_subset=False):
    """
    Loads a dataset from the Hugging Face datasets library, optionally creating a subset and saving it as a JSON file.

    Args:
        data_dir (str): The directory where the dataset is stored or will be stored. Default is 'data/'.
        dataset_name (str): The name of the dataset to load. Default is 'wmt14'.
        config_name (str): The configuration name of the dataset. Default is 'de-en'.
        split (str): The split of the dataset to load (e.g., 'train', 'test'). Default is 'train'.
        num_samples (int): The number of samples to load from the dataset. Default is 100.
        create_subset (bool): Whether to create a subset of the dataset if it doesn't already exist. Default is False.

    Returns:
        list: A list of dictionaries containing the language names 'en' and 'de' as keys for sentences, or None if the dataset could not be loaded.

    Raises:
        ValueError: If the data directory does not exist.
    """

    # Raise error if the data directory does not exist
    if not os.path.exists(data_dir):
        raise ValueError(f"{data_dir} not a valid directory")

    # Set the file name as per convention
    file_name = f"{dataset_name}_{config_name}_{split}_{num_samples}.json"

    # Determine the file path
    file_path = os.path.join(data_dir, file_name)

    # Check if the subset file exists 
    if os.path.exists(file_path):

        # If exists then print success message and load the dataset from the file
        print(f"JSON EXISTS, loading from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            ds = json.load(f)

    # If file doesn't exist       
    else:

        # If user wants to create the subset
        if create_subset:

            # Print message about creating subset
            print(f"JSON DNE, Creating...")

            # Try-catch block to handle ds or config not found errors
            try:
                # Load the dataset
                ds = load_dataset(dataset_name, config_name, split=split)

                # Check if the number of samples is less than actual samples
                if len(ds) < num_samples:

                    # Adjust filename and path
                    file_name = f"{dataset_name}_{config_name}_{split}_{len(ds)}.json"
                    file_path = os.path.join(data_dir, file_name)

                    # Print warning message
                    print(f"WARNING: Truncating to {len(ds)} < {num_samples}, file renamed {file_name}")

                    # Adjust the number of samples to fit the dataset
                    num_samples = len(ds)

                    # If error from HF then print              
            except Exception as e:
                print(f"\nHF-ERROR:\t{e}")
                return None

            # Reduce to number of samples required
            ds = ds.select(range(num_samples))

            # Save the subset to a JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(ds.to_dict(), f, ensure_ascii=False, indent=4)

            # Print success message of saving the subset
            print(f"\nSUCCESS")

        # If user doesn't want to create the subset
        else:

            # Print error message and return none
            print(f"JSON DNE, EXITING: Set 'create_subset' = True for new subset")
            return None

    # Access outer key to return actual sentences
    return ds['translation']


def scaled_dot_product_attention(query, key, value):
    """
    Computes the scaled dot-product attention.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, seq_len, embedding_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, seq_len, embedding_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, seq_len, embedding_dim).

    Returns:
        torch.Tensor: The result of the attention mechanism applied to the value tensor, 
                      of shape (batch_size, seq_len, embedding_dim).
    """

    # Matmul Q, K^T and scale by the last dim (embedding dim) 
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(query.size(-1))

    # Apply softmax to get attention weights along cols, prob dist
    weights = F.softmax(scores, dim = -1)

    # Calculate context-aware attention by matmul W and V tensors
    return torch.bmm(weights, value)