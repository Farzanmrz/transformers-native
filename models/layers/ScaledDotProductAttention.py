import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    This class implements the scaled dot-product attention mechanism as described in the
    "Attention is All You Need" paper by Vaswani et al. (2017). It computes the attention
    scores, applies a softmax to obtain the attention weights, and then computes the
    weighted sum of the values.
    """
    def __init__(self, d_k):
        """
        Initialize the ScaledDotProductAttention module.

        Args:
            d_k (int): The dimension of the keys and queries.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        """
        Perform the forward pass of the scaled dot-product attention.

        Args:
            Q (torch.Tensor): The query matrix of shape (batch_size, num_heads, seq_len, d_k).
            K (torch.Tensor): The key matrix of shape (batch_size, num_heads, seq_len, d_k).
            V (torch.Tensor): The value matrix of shape (batch_size, num_heads, seq_len, d_v).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, 1, 1, seq_len)
                                           to prevent attention to certain positions. Default is None.

        Returns:
            output (torch.Tensor): The output matrix of shape (batch_size, num_heads, seq_len, d_v).
            attn (torch.Tensor): The attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-1)

        # Compute the output as weighted sum of values
        output = torch.matmul(attn, V)

        return output, attn