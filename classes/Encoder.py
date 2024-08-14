# Imports
import torch
import torch.nn as nn
from tqdm import tqdm


from classes.MultiHeadAttention import MultiHeadAttention
from classes.PoswiseFeedForward import PoswiseFeedForward
from classes.EmbeddingLayer import EmbeddingLayer


class EncoderLayer(nn.Module):
    """
    EncoderLayer is a single layer of the Transformer encoder.

    This layer consists of a multi-head attention mechanism followed by a position-wise feed-forward network.
    Each of these sub-layers has a residual connection around it, followed by pre-layer normalization.

    Attributes:
        _embed_dim (int): The dimension of the embeddings.
        _norm1 (nn.LayerNorm): Layer normalization applied before the multi-head attention.
        _multihead_attn (MultiHeadAttention): Multi-head attention mechanism.
        _norm2 (nn.LayerNorm): Layer normalization applied before the feed-forward network.
        _feed_forward (PoswiseFeedForward): Position-wise feed-forward network.
    """

    def __init__(self, config):
        """
        Initializes the EncoderLayer with the given configuration.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()

        # Set the hidden size
        self._embed_dim = config.hidden_size  # hidden_size = 768

        # Setup layers in order output size for each layer is batch_size x seq_len x embed_dim = 100x256x768
        self._norm1 = nn.LayerNorm(self._embed_dim)
        self._multihead_attn = MultiHeadAttention(config)
        self._norm2 = nn.LayerNorm(self._embed_dim)
        self._feed_forward = PoswiseFeedForward(config)

    def forward(self, x):
        """
        Forward pass for the EncoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim) after applying multi-head attention,
                          feed-forward network, and skip connections.
        """

        # Get the multihead attention on normalized input x
        multihead_attn = self._multihead_attn(self._norm1(x))

        # Apply skip connection by adding the input x and the output of multihead attention and creating new x2
        x2 = x + multihead_attn

        # Get the FFN output on normalized new input x2
        ffn_output = self._feed_forward(self._norm2(x2))

        # Apply skip connection by adding the input of ffn x2 and the output of ffn to get x3
        x3 = x2 + ffn_output

        # Return the final x3
        return x3


class Encoder(nn.Module):
    """
    Encoder is a stack of N encoder layers.

    This class implements the full encoder block of the Transformer model, which consists of multiple
    EncoderLayer instances stacked on top of each other.

    Attributes:
        _embeddings (Embeddings): Embedding layer that combines token and positional embeddings.
        _layers (nn.ModuleList): List of EncoderLayer instances.
    """

    def __init__(self, config):
        """
        Initializes the Encoder with the given configuration.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()

        # Get the number of hidden layers from configuration
        self._num_hidden_layers = config.num_hidden_layers  # num_hidden_layers = 12

        # Setup the embeddings which are token + positional normalized
        self._embeddings = EmbeddingLayer(config)

        # Setup the layers which have the full encoder block from EncoderLayer object
        self._layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(self._num_hidden_layers)]
        )


    def forward(self, x):
        """
        Forward pass for the Encoder.

        This function takes an input tensor `x`, converts it to embeddings, and then passes it through
        a series of encoder layers. Each encoder layer applies multi-head attention, normalization, and
        feed-forward operations with skip connections.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor after passing through all encoder layers, of shape (batch_size, seq_len, embed_dim).
        """
        # Convert input x to embeddings
        x = self._embeddings(x)

        # Loop through each layer
        for layer in tqdm(self.layers, desc="Encoding layers"):

            # Pass x through the layers setting it to the output
            x = layer(x)

        # Return final x
        return x
