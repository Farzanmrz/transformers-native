import torch
import torch.nn as nn
from models.layers.ScaledDotProductAttention import ScaledDotProductAttention

class MaskedMultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism with masking for the Transformer Decoder.
    """

    def __init__(self, d_model, num_heads):
        """
        Initializes the MaskedMultiHeadAttention module.

        Args:
            d_model (int): The dimension of the input and output features.
            num_heads (int): The number of attention heads.
        """
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Linear layers to project the input features to queries, keys, and values
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Scaled dot-product attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for the MaskedMultiHeadAttention module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = Q.size(0)

        # Linear projections and reshaping for multi-head attention
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        scores, attn = self.attention(Q, K, V, mask)

        # Concatenate the attention output and apply the final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        output = self.out(concat)

        return output, attn