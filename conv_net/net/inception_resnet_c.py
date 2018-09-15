import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionResnetC(nn.Module):

    def __init__(self):
        super(InceptionResnetC, self).__init__()
        self.conv1a = nn.Conv2d(2016, 192, 1, padding=1)
        self.conv1b = nn.Conv2d(192, 224, kernel_size=(1, 3)) 
        self.conv1c = nn.Conv2d(224, 256, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(2016, 192, 1)
        self.fcconv = nn.Conv2d(448, 2016, 1)

    def forward(self, x):
        prev = x
        branches = [
            self.branch_conv_1_13_31(x),
            self.conv2(x),
        ]
        x = self.fcconv(torch.cat(branches, 1))
        return F.relu(prev + x)

    def branch_conv_1_13_31(self, x):
        branch = self.conv1a(x)
        branch = self.conv1b(branch)
        return self.conv1c(branch)
