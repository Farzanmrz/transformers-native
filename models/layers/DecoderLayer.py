import torch
import torch.nn as nn
from models.layers.MaskedMultiHeadAttention import MaskedMultiHeadAttention
from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer Decoder as described in the
    "Attention is All You Need" paper by Vaswani et al. This module consists of
    two multi-head attention mechanisms followed by a position-wise feed-forward
    neural network, with residual connections and layer normalization.

    Attributes:
        self_attn (MaskedMultiHeadAttention): The first multi-head attention layer for self-attention.
        enc_dec_attn (MultiHeadAttention): The second multi-head attention layer for encoder-decoder attention.
        feed_forward (PositionwiseFeedForward): The position-wise feed-forward layer.
        norm1 (nn.LayerNorm): The first layer normalization.
        norm2 (nn.LayerNorm): The second layer normalization.
        norm3 (nn.LayerNorm): The third layer normalization.
        dropout (nn.Dropout): The dropout layer for regularization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initializes the DecoderLayer module.

        Args:
            d_model (int): The dimension of the input and output features.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float, optional): The dropout rate. Default is 0.1.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Defines the forward pass of the DecoderLayer module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).
            enc_output (torch.Tensor): The encoder output tensor of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor, optional): The source mask for the encoder output.
            tgt_mask (torch.Tensor, optional): The target mask for the decoder input.

        Returns:
            torch.Tensor: The output tensor of the same shape as the input (batch_size, seq_length, d_model).
        """
        _x = x
        x, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x) + _x
        x = self.norm1(x)

        _x = x
        x, _ = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.dropout(x) + _x
        x = self.norm2(x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout(x) + _x
        x = self.norm3(x)

        return x