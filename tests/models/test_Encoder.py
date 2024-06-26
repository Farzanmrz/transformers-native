import unittest
import torch
from models.Encoder import Encoder


class TestEncoder(unittest.TestCase):
	"""
	Unit test class for the Encoder.
	"""

	def setUp( self ):
		"""
		Set up the test environment before each test method is run.

		Initializes the Encoder with specific dimensions and creates random input tensors.
		"""
		self.vocab_size = 10000  # Size of the input vocabulary
		self.d_model = 512  # Dimension of the model
		self.num_layers = 6  # Number of encoder layers
		self.num_heads = 8  # Number of attention heads
		self.d_ff = 2048  # Dimension of the feed-forward layer
		self.encoder = Encoder(self.vocab_size, self.d_model, self.num_layers, self.num_heads, self.d_ff)

		# Random input tensor with shape (batch_size, sequence_length)
		self.src = torch.randint(0, self.vocab_size, (1, 10))
		# Mask tensor with shape (batch_size, 1, sequence_length, sequence_length)
		self.mask = torch.ones(1, 1, 10, 10)

	def test_output_shape( self ):
		"""
		Test if the output shape of the Encoder matches the expected shape.

		This ensures that the Encoder correctly processes the input tensor and returns
		an output tensor with the correct shape.
		"""
		output = self.encoder(self.src, self.mask)
		self.assertEqual(output.shape, (1, 10, self.d_model))


if __name__ == '__main__':
	unittest.main()
