import unittest
import torch
from net.inception_resnet_b import InceptionResnetB

class TestInceptionResnetB(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 1024, 17, 17).float()

    def test_forward_returns_correct_size(self):
        irb = InceptionResnetB()
        output_tensor = irb(self.input_tensor)
        self.assertEqual(output_tensor.size(), (1, 1024, 17, 17))

    def test_backward(self):
        pass
