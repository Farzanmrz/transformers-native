import torch
import torch.nn as nn

from classes.AttentionHead import AttentionHead


class MultiHeadAttention(nn.Module):
    """
    Implements multi-headed attention mechanism for a Transformer model.

    Attributes:
        heads (nn.ModuleList): List of AttentionHead modules, one for each attention head.
        output_linear (nn.Linear): Linear layer to combine the outputs from all heads back to the embedding dimension.
    """

    def __init__(self, config):
        """
        Initializes the Simple MultiHeadAttentionLayer with the given configuration.

        Args:
            config (transformers.PretrainedConfig): Configuration object containing hyperparameters such as hidden_size and num_attention_heads.
        """
        super().__init__()

        # Set the embedding dimension and number of heads from configuration
        self._embed_dim = config.hidden_size  # hidden_size = 768
        self._num_heads = (
            config.num_attention_heads
        )  # num_attention_heads model has = 12

        # Compute the dimension of each head
        self._head_dim = self._embed_dim // self._num_heads  # 768/12 = 64

        # Initialize num_heads number of single AttentionHead modules each of size batch_size x seq_len x head_dim = 100x256x64
        self._heads = nn.ModuleList(
            [
                AttentionHead(self._embed_dim, self._head_dim)
                for _ in range(self._num_heads)
            ]
        )  # 12 AttentionHeads of 100x256x64

        # Linear layer doesn't impact first 2 dims of batch_size and seq_len just multiplies by the last dim
        self._output_linear = nn.Linear(
            self._embed_dim, self._embed_dim
        )  # hidden_size x hidden_size = 768x768

    def forward(self, hidden_state):
        """
        Performs the forward pass of the multi-headed attention mechanism.

        Args:
            hidden_state (torch.Tensor): The input tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            torch.Tensor: The output tensor after applying multi-headed attention, of shape [batch_size, seq_len, embed_dim].
        """
        # Process the hidden_state through each attention head and concatenate the results along last head_dim
        x = torch.cat(
            [head(hidden_state) for head in self._heads], dim=-1
        )  # batch_size x seq_len x hidden_size = 100x256x768

        # Apply the output linear layer that basically just does [batch_size x seq_len x embed_dim] x [embed_dim x embed_dim] = [batch_size x seq_len x embed_dim]
        return self._output_linear(x)  # batch_size x seq_len x embed_dim = 100x256x768
