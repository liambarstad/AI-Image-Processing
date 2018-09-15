import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

arr = ndimage.imread('./sample.jpg')
tens = torch.from_numpy(arr)

print(arr.shape)
