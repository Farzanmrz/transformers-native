import unittest
import torch
from models.layers.DecoderLayer import DecoderLayer


class TestDecoderLayer(unittest.TestCase):
	"""
	Unit test class for the DecoderLayer.
	"""

	def setUp( self ):
		"""
		Set up the test environment before each test method is run.

		Initializes the DecoderLayer with specific dimensions and creates random input tensors.
		"""
		self.d_model = 512  # Dimension of the model
		self.num_heads = 8  # Number of attention heads
		self.d_ff = 2048  # Dimension of the feed-forward layer
		self.decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff)

		# Random input tensor with shape (batch_size, sequence_length, d_model)
		self.x = torch.rand(1, 10, self.d_model)
		self.enc_output = torch.rand(1, 10, self.d_model)
		self.src_mask = torch.ones(1, 1, 10, 10)
		self.tgt_mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)

	def test_output_shape( self ):
		"""
		Test if the output shape of the DecoderLayer matches the input shape.

		This ensures that the layer correctly processes the input tensor and returns
		an output tensor with the same shape.
		"""
		output = self.decoder_layer(self.x, self.enc_output, self.src_mask, self.tgt_mask)
		self.assertEqual(output.shape, self.x.shape)


if __name__ == '__main__':
	unittest.main()
