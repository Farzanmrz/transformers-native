import unittest
import torch
from models.Decoder import Decoder


class TestDecoder(unittest.TestCase):
    """
    Unit test class for the Decoder.
    """

    def setUp(self):
        """
        Set up the test environment before each test method is run.

        Initializes the Decoder with specific dimensions and creates random input tensors.
        """
        self.vocab_size = 10000  # Size of the input vocabulary
        self.d_model = 512  # Dimension of the model
        self.num_layers = 6  # Number of decoder layers
        self.num_heads = 8  # Number of attention heads
        self.d_ff = 2048  # Dimension of the feed-forward layer
        self.decoder = Decoder(self.vocab_size, self.d_model, self.num_layers, self.num_heads, self.d_ff)

        # Random input tensors with shape (batch_size, sequence_length)
        self.tgt = torch.randint(0, self.vocab_size, (1, 10))
        self.enc_output = torch.rand(1, 10, self.d_model)
        self.src_mask = torch.ones(1, 1, 10, 10)
        self.tgt_mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)

    def test_output_shape(self):
        """
        Test if the output shape of the Decoder matches the expected shape.

        This ensures that the Decoder correctly processes the input tensor and returns
        an output tensor with the correct shape.
        """
        output = self.decoder(self.tgt, self.enc_output, self.src_mask, self.tgt_mask)
        self.assertEqual(output.shape, (1, 10, self.d_model))


if __name__ == '__main__':
    unittest.main()