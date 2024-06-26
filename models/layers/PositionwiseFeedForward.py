import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Neural Network as described in the
    "Attention is All You Need" paper by Vaswani et al. This module consists of
    two linear transformations with a ReLU activation in between, followed by a
    dropout layer for regularization.

    Attributes:
        linear1 (nn.Linear): The first linear transformation layer.
        dropout (nn.Dropout): The dropout layer for regularization.
        linear2 (nn.Linear): The second linear transformation layer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initializes the PositionwiseFeedForward module.

        Args:
            d_model (int): The dimension of the input and output features.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float, optional): The dropout rate. Default is 0.1.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Defines the forward pass of the PositionwiseFeedForward module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: The output tensor of the same shape as the input (batch_size, seq_length, d_model).
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))