import torch
import torch.nn as nn
import torch.nn.functional as F

class ReductionB(nn.Module):

    def __init__(self):
        super(ReductionB, self).__init__()
        self.conv1a = nn.Conv2d(1024, 256, 1, padding=1) 
        self.conv1b = nn.Conv2d(256, 288, 3)
        self.conv1c = nn.Conv2d(288, 320, 3, stride=2)
        self.conv2 = nn.Conv2d(1024, 256, 1)
        self.conv2a = nn.Conv2d(256, 288, 3, stride=2)
        self.conv2b = nn.Conv2d(256, 384, 3, stride=2)
        self.max_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        branches = [
            self.branch_conv_1_3_3(x),
            self.branch_288_conv_1_3(x),
            self.branch_384_conv_1_3(x),
            self.max_pool(x)
        ]
        return torch.cat(branches, 1)

    def branch_conv_1_3_3(self, x):
        branch = self.conv1a(x)
        branch = self.conv1b(branch)
        return self.conv1c(branch)

    def branch_288_conv_1_3(self, x):
        branch = self.conv2(x)
        return self.conv2a(branch)

    def branch_384_conv_1_3(self, x):
        branch = self.conv2(x)
        return self.conv2b(branch)
