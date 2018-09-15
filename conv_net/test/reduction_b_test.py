import unittest
import torch
from net.reduction_b import ReductionB

class TestReductionB(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 1024, 17, 17).float()

    def test_forward_returns_correct_size(self):
        rb = ReductionB()
        output_tensor = rb(self.input_tensor)
        self.assertEqual(output_tensor.size(), (1, 2016, 8, 8))

    def test_backward(self):
        pass
