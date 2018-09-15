import torch
import torch.nn as nn
import torch.nn.functional as F

class ReductionA(nn.Module):

    def __init__(self, k, l, m, n):
        super(ReductionA, self).__init__()
        self.conv1a = nn.Conv2d(384, k, 1, padding=1)
        self.conv1b = nn.Conv2d(k, l, 3)
        self.conv1c = nn.Conv2d(l, m, 3, stride=2)
        self.conv2 = nn.Conv2d(384, n, 3, stride=2)
        self.max_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        branches = [
            self.branch_conv_1_3_3(x),
            self.conv2(x),
            self.max_pool(x),
        ]
        return F.relu(torch.cat(branches, 1))

    def branch_conv_1_3_3(self, x):
        branch = self.conv1a(x)
        branch = self.conv1b(branch)
        return self.conv1c(branch)

