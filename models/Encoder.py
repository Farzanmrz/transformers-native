import torch
import torch.nn as nn
from models.layers.EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    """
    Implements the Transformer Encoder as described in the "Attention is All You Need"
    paper by Vaswani et al. This module consists of a stack of N encoder layers.

    Attributes:
        embedding (nn.Embedding): The input embedding layer.
        layers (nn.ModuleList): A list of encoder layers.
        norm (nn.LayerNorm): The final layer normalization.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        """
        Initializes the Encoder module.

        Args:
            vocab_size (int): The size of the input vocabulary.
            d_model (int): The dimension of the input and output features.
            num_layers (int): The number of encoder layers.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float, optional): The dropout rate. Default is 0.1.
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        """
        Defines the forward pass of the Encoder module.

        Args:
            src (torch.Tensor): The input tensor of shape (batch_size, seq_length).
            mask (torch.Tensor, optional): The attention mask of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_length, d_model).
        """
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
