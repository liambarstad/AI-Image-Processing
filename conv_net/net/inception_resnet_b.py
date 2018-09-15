import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionResnetB(nn.Module):

    def __init__(self):
        super(InceptionResnetB, self).__init__()
        self.conv1a = nn.Conv2d(1024, 128, 1, padding=1)
        self.conv1b = nn.Conv2d(128, 160, kernel_size=(7, 1), padding=1) 
        self.conv1c = nn.Conv2d(160, 192, kernel_size=(1, 7), padding=1)
        self.conv2 = nn.Conv2d(1024, 192, 1)
        self.fcconv = nn.Conv2d(384, 1024, 1) 

    def forward(self, x):
        prev = x
        branches = [
            self.branch_conv_1_17_71(x),
            self.conv2(x),
        ]
        x = torch.cat(branches, 1)
        x = self.fcconv(x)
        return F.relu(x + prev)

    def branch_conv_1_17_71(self, x):
        branch = self.conv1a(x)
        branch = self.conv1b(branch)
        return self.conv1c(branch)

