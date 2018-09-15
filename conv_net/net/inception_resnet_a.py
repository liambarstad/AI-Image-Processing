import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionResnetA(nn.Module):

    def __init__(self):
        super(InceptionResnetA, self).__init__()
        self.conv1a = nn.Conv2d(384, 32, 1, padding=1)
        self.conv1b = nn.Conv2d(32, 48, 3, padding=1) 
        self.conv1c = nn.Conv2d(48, 64, 3)
        self.conv2a = nn.Conv2d(384, 32, 1, padding=1)
        self.conv2b = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(384, 32, 1)
        self.fcconv = nn.Conv2d(128, 384, 1)

    def forward(self, x):
        prev = x
        branches = [
            self.branch_conv_1_3_3(x),
            self.branch_conv_1_3(x),
            F.relu(self.conv3(x)),
        ]
        x = torch.cat(branches, 1)
        x = self.fcconv(x)
        return F.relu(x + prev)

    def branch_conv_1_3_3(self, x):
        branch = self.conv1a(x)
        branch = self.conv1b(branch)
        return self.conv1c(branch)

    def branch_conv_1_3(self, x):
        branch = self.conv2a(x)
        return self.conv2b(branch)

