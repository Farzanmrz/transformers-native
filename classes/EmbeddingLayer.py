import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    EmbeddingLayer class to create token and position embeddings for input sequences.

    Args:
        config: Configuration object containing the following attributes:
            - vocab_size (int): Size of the vocabulary.
            - hidden_size (int): Dimensionality of the embeddings.
            - max_position_embeddings (int): Maximum number of position embeddings.
    """

    def __init__(self, config):
        """
        Initializes the EmbeddingLayer with token and position embeddings, layer normalization, and dropout.

        Args:
            config: Configuration object containing the following attributes:
                - vocab_size (int): Size of the vocabulary.
                - hidden_size (int): Dimensionality of the embeddings.
                - max_position_embeddings (int): Maximum number of position embeddings.
        """
        super().__init__()

        # Define dimensions
        self._vocab_size = config.vocab_size  # 30522
        self._hidden_size = config.hidden_size  # 768
        self._position_embed_size = config.max_position_embeddings  # 512

        # Create token and position embedding layer using config
        self.token_embeddings = nn.Embedding(
            self._vocab_size, self._hidden_size
        )  # 30522x768
        self.position_embeddings = nn.Embedding(
            self._position_embed_size, self._hidden_size
        )  # 512x768

        # Normalize across hidden dimension
        self.layer_norm = nn.LayerNorm(self._hidden_size, eps=1e-12)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        """
        Forward pass for the EmbeddingLayer.

        Args:
            input_ids (torch.Tensor): A tensor of shape (batch_size, seq_length) containing the input token IDs.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, seq_length, hidden_size) containing the combined token and position embeddings,
            after applying layer normalization and dropout.
        """
        # Get length of sequence by checking how many columns in input_ids = 256
        seq_length = input_ids.size(1)  # 256

        # Create position ID tensor of shape [1 x seq_length] = 1x256
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

        # Create token embedding of shape batch_size x seq_length x hidden_size = 100x256x768
        token_embeddings = self.token_embeddings(input_ids)

        # Create position embeddings tensor of shape [batch_size x seq_length x hidden_size] = 100x256x768
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings

        # Apply normalization, dropout and return the result
        embeddings = self.dropout(self.layer_norm(embeddings))
        return embeddings
