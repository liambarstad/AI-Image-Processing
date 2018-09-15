import unittest
import torch
from net.stem import Stem

class TestStem(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 3, 299, 299).float()

    def test_forward_returns_correct_size(self):
        stem = Stem()
        output_tensor = stem(self.input_tensor)
        self.assertEqual(output_tensor.size(), (1, 384, 35, 35))

    def test_backward(self):
        pass
