import unittest
import torch
from models.layers.ScaledDotProductAttention import ScaledDotProductAttention


class TestScaledDotProductAttention(unittest.TestCase):

	def setUp( self ):
		self.d_k = 64
		self.attention_layer = ScaledDotProductAttention(self.d_k)

		self.Q = torch.rand(1, 8, 10, self.d_k)
		self.K = torch.rand(1, 8, 10, self.d_k)
		self.V = torch.rand(1, 8, 10, self.d_k)
		self.mask = torch.ones(1, 8, 10, 10)

	def test_output_shape( self ):
		output, attn = self.attention_layer(self.Q, self.K, self.V, self.mask)
		self.assertEqual(output.shape, self.V.shape)
		self.assertEqual(attn.shape, (1, 8, 10, 10))

	def test_no_mask( self ):
		output, attn = self.attention_layer(self.Q, self.K, self.V)
		self.assertEqual(output.shape, self.V.shape)

	def test_with_mask( self ):
		mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)
		output, attn = self.attention_layer(self.Q, self.K, self.V, mask)
		self.assertEqual(output.shape, self.V.shape)

	def test_attention_weights( self ):
		output, attn = self.attention_layer(self.Q, self.K, self.V, self.mask)
		self.assertAlmostEqual(attn.sum(-1).mean().item(), 1.0, places = 5)


if __name__ == '__main__':
	unittest.main()
