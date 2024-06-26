import torch
import torch.nn as nn
from models.layers.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention is All You Need".

    This module performs attention over multiple heads, allowing the model to focus on different parts of the input sequence simultaneously.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize the MultiHeadAttention module.

        Args:
            d_model (int): The dimensionality of the input and output feature vectors.
            num_heads (int): The number of attention heads.

        Raises:
            AssertionError: If d_model is not divisible by num_heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            batch_size (int): The size of the batch.

        Returns:
            torch.Tensor: The reshaped tensor of shape (batch_size, num_heads, seq_len, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        """
        Perform the forward pass of the MultiHeadAttention module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: The attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = Q.size(0)

        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scaled_attention, attention_weights = self.attention(Q, K, V, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out(concat_attention)

        return output, attention_weights