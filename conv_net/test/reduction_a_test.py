import unittest
import torch
from net.reduction_a import ReductionA

class TestReductionA(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 384, 35, 35).float()

    def test_forward_returns_correct_size(self):
        ra = ReductionA(192, 192, 256, 384)
        output_tensor = ra(self.input_tensor)
        self.assertEqual(output_tensor.size(), (1, 1024, 17, 17))

    def test_backward(self):
        pass
