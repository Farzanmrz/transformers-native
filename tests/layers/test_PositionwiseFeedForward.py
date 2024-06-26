import unittest
import torch
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward

class TestPositionwiseFeedForward(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.d_ff = 2048
        self.ff_layer = PositionwiseFeedForward(self.d_model, self.d_ff)

        self.x = torch.rand(1, 10, self.d_model)

    def test_output_shape(self):
        output = self.ff_layer(self.x)
        self.assertEqual(output.shape, self.x.shape)

if __name__ == '__main__':
    unittest.main()
