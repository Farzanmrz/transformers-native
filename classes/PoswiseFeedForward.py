import torch.nn as nn
class PoswiseFeedForward(nn.Module):
	def __init__(self, config):
		super().__init__()

		# Set the instance variables
		self._embed_dim = config.hidden_size # hidden_size = 768
		self._expanded_dim = config.intermediate_size # intermediate_size = 3072
		self._drop_prob = config.hidden_dropout_prob # hidden_dropout_prob = 0.1

		# Linear layer to expand from hidden_size to intermediate_size
		self._linear_1 = nn.Linear(self._embed_dim, self._expanded_dim) # hidden_size x intermediate_size = 768x3072

		# GELU activation function to introduce non-linearity and capture complex patterns in the data
		self.gelu = nn.GELU()

		# Linear layer to project back to hidden_size
		self.linear_2 = nn.Linear(self._expanded_dim, self._embed_dim) # intermediate_size x hidden_size = 3072x768

		# Dropout layer for regularization with a dropout probability of 0.1 meaning 10% of input elements are set to zero
		self.dropout = nn.Dropout(self._drop_prob)

	def forward(self, x):

		# First linear transformation to input tensor of shape [batch_size x seq_len x embed_dim] = 100x256x768
		lin1 = self._linear_1(x) # batch_size x seq_len x intermediate_size = 100x256x3072

		# GELU activation function
		nonlin = self.gelu(lin1) # batch_size x seq_len x intermediate_size = 100x256x3072

		# Second linear transformation
		lin2 = self.linear_2(nonlin) # batch_size x seq_len x intermediate_size = 100x256x768

		# Use dropout for regularization
		output = self.dropout(lin2) # batch_size x seq_len x hidden_size = 100x256x768

		# Return the output after dropout layer, shape remains the same as the input
		return output

