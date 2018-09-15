import unittest
import torch
from net.inception_resnet_c import InceptionResnetC

class TestInceptionResnetC(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 2016, 8, 8)

    def test_forward_returns_correct_size(self):
        irc = InceptionResnetC()
        output_tensor = irc(self.input_tensor)
        self.assertEqual(output_tensor.size(), self.input_tensor.size())

    def test_backward(self):
        pass
