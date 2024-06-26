import unittest
import torch
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward

class TestPositionwiseFeedForward(unittest.TestCase):
    """
    Unit test class for the PositionwiseFeedForward layer.
    """

    def setUp(self):
        """
        Set up the test environment before each test method is run.

        Initializes the PositionwiseFeedForward layer with specific dimensions
        and creates a random input tensor.
        """
        self.d_model = 512  # Dimension of the model
        self.d_ff = 2048    # Dimension of the feed-forward layer
        self.ff_layer = PositionwiseFeedForward(self.d_model, self.d_ff)

        # Random input tensor with shape (batch_size, sequence_length, d_model)
        self.x = torch.rand(1, 10, self.d_model)

    def test_output_shape(self):
        """
        Test if the output shape of the PositionwiseFeedForward layer matches the input shape.

        This ensures that the layer correctly processes the input tensor and returns
        an output tensor with the same shape.
        """
        output = self.ff_layer(self.x)
        self.assertEqual(output.shape, self.x.shape)

if __name__ == '__main__':
    unittest.main()