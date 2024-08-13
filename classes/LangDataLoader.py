import torch
from tqdm import tqdm


class LangDataLoader(torch.utils.data.Dataset):
    """
    DataLoader class for preparing input data for the model.

    Args:
            data (list): List of dictionaries containing the data.
            tokenizer (Tokenizer): Tokenizer used for tokenization.
            pad_length (int, optional): Maximum length for padding sentences. Defaults to 256.
    """

    def __init__(self, data, tokenizer, pad_length=256):
        """
        Initializes the LangDataLoader with data, tokenizer, and padding length.

        Args:
                data (list): List of dictionaries containing the data.
                tokenizer (Tokenizer): Tokenizer used for tokenization.
                pad_length (int, optional): Maximum length for padding sentences. Defaults to 256.
        """
        self._raw_data = data
        self._tokenizer = tokenizer
        self._pad_length = pad_length

        # Preprocess and tokenize data with progress tracking
        en_toks, de_toks = [], []
        for example in tqdm(data, desc="Tokenizing and Padding"):
            en_tokens, de_tokens = self._featurize(example, tokenizer)
            en_toks.append(en_tokens)
            de_toks.append(de_tokens)

        # Convert lists of tensors into 2D tensors of size batch_size x seq_len = 100x256
        self._en_toks = torch.cat(en_toks, dim=0)
        self._de_toks = torch.cat(de_toks, dim=0)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
                int: Number of samples in the dataset.
        """
        return self._en_toks.size(0)

    def __getitem__(self, item):
        """
        Returns a sample from the dataset.

        Args:
                item (int): Index of the sample to retrieve.

        Returns:
                dict: Dictionary containing the English and German tokens.
        """
        return {"en": self._en_toks[item], "de": self._de_toks[item]}

    def _featurize(self, example, tokenizer):
        """
        Tokenizes and pads sentences, returning tokens as tensors.

        Args:
                example (dict): Dictionary containing the English and German sentences.
                tokenizer (Tokenizer): Tokenizer used for tokenization.

        Returns:
                tuple: Tuple containing the English and German tokens as tensors.
        """
        en_tokens = tokenizer(
            example["en"],
            padding="max_length",
            max_length=self._pad_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
        de_tokens = tokenizer(
            example["de"],
            padding="max_length",
            max_length=self._pad_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids

        return en_tokens, de_tokens
