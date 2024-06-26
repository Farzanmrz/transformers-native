import torch
import torch.nn as nn
from models.layers.DecoderLayer import DecoderLayer

class Decoder(nn.Module):
    """
    Implements the Transformer Decoder as described in the "Attention is All You Need"
    paper by Vaswani et al. This module consists of a stack of N decoder layers.

    Attributes:
        embedding (nn.Embedding): The input embedding layer.
        layers (nn.ModuleList): A list of decoder layers.
        norm (nn.LayerNorm): The final layer normalization.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        """
        Initializes the Decoder module.

        Args:
            vocab_size (int): The size of the input vocabulary.
            d_model (int): The dimension of the input and output features.
            num_layers (int): The number of decoder layers.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float, optional): The dropout rate. Default is 0.1.
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """
        Defines the forward pass of the Decoder module.

        Args:
            tgt (torch.Tensor): The target tensor of shape (batch_size, seq_length).
            enc_output (torch.Tensor): The encoder output tensor of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor, optional): The source mask for the encoder output.
            tgt_mask (torch.Tensor, optional): The target mask for the decoder input.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_length, d_model).
        """
        x = self.embedding(tgt)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.norm(x)
        return x