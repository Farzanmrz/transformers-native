import unittest
import torch
from models.layers.EncoderLayer import EncoderLayer


class TestEncoderLayer(unittest.TestCase):
	"""
	Unit test class for the EncoderLayer.
	"""

	def setUp( self ):
		"""
		Set up the test environment before each test method is run.

		Initializes the EncoderLayer with specific dimensions and creates random input tensors.
		"""
		self.d_model = 512  # Dimension of the model
		self.num_heads = 8  # Number of attention heads
		self.d_ff = 2048  # Dimension of the feed-forward layer
		self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff)

		# Random input tensor with shape (batch_size, sequence_length, d_model)
		self.x = torch.rand(1, 10, self.d_model)
		# Mask tensor with shape (batch_size, 1, sequence_length, sequence_length)
		self.mask = torch.ones(1, 1, 10, 10)

	def test_output_shape( self ):
		"""
		Test if the output shape of the EncoderLayer matches the input shape.

		This ensures that the layer correctly processes the input tensor and returns
		an output tensor with the same shape.
		"""
		output = self.encoder_layer(self.x, self.mask)
		self.assertEqual(output.shape, self.x.shape)


if __name__ == '__main__':
	unittest.main()
