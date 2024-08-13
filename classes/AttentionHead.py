import torch.nn as nn

from selfutil import scaled_dot_product_attention


class AttentionHead(nn.Module):
    """
    AttentionHead class to perform attention mechanism on input sequences.

    Args:
        hidden_size (int): Dimensionality of the input embeddings.
        head_dim (int): Dimensionality of each attention head.
    """

    def __init__(self, hidden_size, head_dim):
        """
        Initializes the AttentionHead with linear layers for query, key, and value transformations.

        Args:
            hidden_size (int): Dimensionality of the input embeddings.
            head_dim (int): Dimensionality of each attention head.
        """
        super().__init__()

        # Set the instance variables
        self.hidden_size = hidden_size  # 768
        self.head_dim = head_dim  # 64

        # Linear transform to Q,K,V tensors of [batch_size x seqlen x head_dim] = 1x5x64 for each head to be concatenated along last dim
        self.q = nn.Linear(self.hidden_size, self.head_dim)
        self.k = nn.Linear(self.hidden_size, self.head_dim)
        self.v = nn.Linear(self.hidden_size, self.head_dim)

    def forward(self, hidden_state):
        """
        Forward pass for the AttentionHead.

        Args:
            hidden_state (torch.Tensor): A tensor of shape (batch_size, seq_length, hidden_size) containing the input embeddings.

        Returns:
            torch.Tensor: A tensor containing the attention output of shape (batch_size, seq_length, head_dim).
        """
        # Linear transform input hidden_state to get Q,K,V tensors = 1x5x64
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)

        # Calculate scaled dot product attention using func and return
        attn_output = scaled_dot_product_attention(q, k, v)
        return attn_output
