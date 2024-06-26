import unittest
import torch
from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.ScaledDotProductAttention import ScaledDotProductAttention

class TestMultiHeadAttention(unittest.TestCase):
    """
    Unit test class for testing the MultiHeadAttention layer.
    """

    def setUp(self):
        """
        Set up the test environment before each test method is run.
        Initializes the MultiHeadAttention layer and sample input tensors.
        """
        self.d_model = 512
        self.num_heads = 8
        self.attention_layer = MultiHeadAttention(self.d_model, self.num_heads)

        self.q = torch.rand(1, 10, self.d_model)
        self.k = torch.rand(1, 10, self.d_model)
        self.v = torch.rand(1, 10, self.d_model)
        self.mask = torch.ones(1, 1, 10, 10)

    def test_output_shape(self):
        """
        Test that the output shape of the MultiHeadAttention layer matches the input shape.
        Also checks the shape of the attention weights.
        """
        output, attn = self.attention_layer(self.q, self.k, self.v, self.mask)
        self.assertEqual(output.shape, self.q.shape)
        self.assertEqual(attn.shape, (1, self.num_heads, 10, 10))

    def test_no_mask(self):
        """
        Test the MultiHeadAttention layer without providing a mask.
        Ensures that the output shape matches the input shape.
        """
        output, attn = self.attention_layer(self.q, self.k, self.v)
        self.assertEqual(output.shape, self.q.shape)

    def test_scaled_dot_product_attention_integration(self):
        """
        Test the integration of the ScaledDotProductAttention within the MultiHeadAttention layer.
        Verifies that the scaled attention and attention weights have the correct shapes.
        """
        scaled_dot_product_attn = ScaledDotProductAttention(self.d_model // self.num_heads)

        batch_size = self.q.size(0)
        q_split = self.q.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k_split = self.k.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        v_split = self.v.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        scaled_attention, attention_weights = scaled_dot_product_attn(q_split, k_split, v_split, self.mask)
        self.assertEqual(scaled_attention.shape, (batch_size, self.num_heads, 10, self.d_model // self.num_heads))
        self.assertEqual(attention_weights.shape, (batch_size, self.num_heads, 10, 10))

if __name__ == '__main__':
    unittest.main()