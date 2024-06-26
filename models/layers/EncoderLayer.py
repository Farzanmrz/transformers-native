import torch
import torch.nn as nn
from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
	"""
	Implements a single layer of the Transformer Encoder as described in the
	"Attention is All You Need" paper by Vaswani et al. This module consists of
	a multi-head attention mechanism followed by a position-wise feed-forward
	neural network, with residual connections and layer normalization.

	Attributes:
		attention (MultiHeadAttention): The multi-head attention layer.
		feed_forward (PositionwiseFeedForward): The position-wise feed-forward layer.
		norm1 (nn.LayerNorm): The first layer normalization.
		norm2 (nn.LayerNorm): The second layer normalization.
		dropout (nn.Dropout): The dropout layer for regularization.
	"""

	def __init__( self, d_model, num_heads, d_ff, dropout = 0.1 ):
		"""
		Initializes the EncoderLayer module.

		Args:
			d_model (int): The dimension of the input and output features.
			num_heads (int): The number of attention heads.
			d_ff (int): The dimension of the inner feed-forward layer.
			dropout (float, optional): The dropout rate. Default is 0.1.
		"""
		super(EncoderLayer, self).__init__()
		self.attention = MultiHeadAttention(d_model, num_heads)
		self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward( self, x, mask = None ):
		"""
		Defines the forward pass of the EncoderLayer module.

		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).
			mask (torch.Tensor, optional): The attention mask of shape (batch_size, 1, seq_length, seq_length).

		Returns:
			torch.Tensor: The output tensor of the same shape as the input (batch_size, seq_length, d_model).
		"""
		attn_output, _ = self.attention(x, x, x, mask)
		x = x + self.dropout(attn_output)
		x = self.norm1(x)

		ff_output = self.feed_forward(x)
		x = x + self.dropout(ff_output)
		x = self.norm2(x)

		return x
