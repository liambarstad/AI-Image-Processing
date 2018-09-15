import torch
import torch.nn as nn
import torch.nn.functional as F

class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 96, 3, stride=2)
        self.conv5a = nn.Conv2d(160, 64, 1, padding=3)
        self.conv5b = nn.Conv2d(64, 64, kernel_size=(7, 1))
        self.conv5c = nn.Conv2d(64, 64, kernel_size=(1, 7))
        self.conv5d = nn.Conv2d(64, 96, 3)
        self.conv6a = nn.Conv2d(160, 64, 1)
        self.conv6b = nn.Conv2d(64, 96, 3)
        self.conv7 = nn.Conv2d(192, 192, 3, stride=2)
        self.max_pool = nn.MaxPool2d(3, stride=2) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        branches = [
            self.conv4(x),
            self.max_pool(x),
        ]
        x = F.relu(torch.cat(branches, 1))

        branches = [
            self.branch_conv_1_71_17_3(x),
            self.branch_conv_1_3(x),
        ]
        x = torch.cat(branches, 1)

        branches = [
            self.max_pool(x),
            self.conv7(x),
        ]
        x = F.relu(torch.cat(branches, 1))
        return x

    def branch_conv_1_3(self, x):
        branch = self.conv6a(x)
        return F.relu(self.conv6b(branch))

    def branch_conv_1_71_17_3(self, x):
        branch = self.conv5a(x)
        branch = self.conv5b(branch)
        branch = self.conv5c(branch)
        return F.relu(self.conv5d(branch))

