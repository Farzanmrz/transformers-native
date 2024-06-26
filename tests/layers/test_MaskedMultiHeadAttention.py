import unittest
import torch
from models.layers.MaskedMultiHeadAttention import MaskedMultiHeadAttention


class TestMaskedMultiHeadAttention(unittest.TestCase):
	"""
	Unit test class for the MaskedMultiHeadAttention.
	"""

	def setUp( self ):
		"""
		Set up the test environment before each test method is run.

		Initializes the MaskedMultiHeadAttention with specific dimensions and creates random input tensors.
		"""
		self.d_model = 512  # Dimension of the model
		self.num_heads = 8  # Number of attention heads
		self.attention_layer = MaskedMultiHeadAttention(self.d_model, self.num_heads)

		self.q = torch.rand(1, 10, self.d_model)
		self.k = torch.rand(1, 10, self.d_model)
		self.v = torch.rand(1, 10, self.d_model)
		self.mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)  # Lower triangular mask

	def test_output_shape( self ):
		"""
		Test if the output shape of the MaskedMultiHeadAttention matches the expected shape.

		This ensures that the layer correctly processes the input tensor and returns
		an output tensor with the correct shape.
		"""
		output, attn = self.attention_layer(self.q, self.k, self.v, self.mask)
		self.assertEqual(output.shape, self.q.shape)
		self.assertEqual(attn.shape, (1, self.num_heads, 10, 10))

	def test_no_mask( self ):
		"""
		Test if the layer works correctly without a mask.
		"""
		output, attn = self.attention_layer(self.q, self.k, self.v)
		self.assertEqual(output.shape, self.q.shape)


if __name__ == '__main__':
	unittest.main()
