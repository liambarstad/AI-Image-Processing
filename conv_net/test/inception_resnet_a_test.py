import unittest
import torch
from net.inception_resnet_a import InceptionResnetA

class TestInceptionResnetA(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 384, 35, 35).float()

    def test_forward_returns_correct_size(self):
        ira = InceptionResnetA()
        output_tensor = ira(self.input_tensor)
        self.assertEqual(output_tensor.size(), self.input_tensor.size())

    def test_backward(self):
        pass
