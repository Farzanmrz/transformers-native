import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    A class to create input embeddings for a sequence of tokens, including both token embeddings and positional embeddings.

    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for token embeddings.
        position_embeddings (nn.Parameter): Parameter containing positional embeddings.
    """

    def __init__(self, vocab_size, d_model, max_len=10000):
        """
        Initializes the InputEmbeddings class with token and positional embeddings.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the embeddings.
            max_len (int, optional): Maximum length of the input sequences. Defaults to 10000.
        """
        super(InputEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)  # padding_idx is set to 0
        self.position_embeddings = self.create_position_embedding(d_model, max_len)

    def create_position_embedding(self, d_model, max_len):
        """
        Creates positional embeddings using sine and cosine functions.

        Args:
            d_model (int): Dimension of the embeddings.
            max_len (int): Maximum length of the input sequences.

        Returns:
            nn.Parameter: Positional embeddings as a non-trainable parameter.
        """
        position_encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        position_encoding[:, 0::2] = torch.sin(positions * div_term)
        position_encoding[:, 1::2] = torch.cos(positions * div_term)

        position_encoding = position_encoding.unsqueeze(0)  # For batch compatibility
        return nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, x):
        """
        Forward pass to compute the combined token and positional embeddings.

        Args:
            x (torch.Tensor): Input tensor containing token indices with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Combined token and positional embeddings with shape (batch_size, sequence_length, d_model).
        """
        seq_len = x.size(1)
        x = self.token_embeddings(x)
        x += self.position_embeddings[:, :seq_len, :]
        return x

