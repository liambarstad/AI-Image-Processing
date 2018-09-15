import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from inception_resnet_v2 import InceptionResnetV2

arr = ndimage.imread('./sample.jpg')
arr = np.swapaxes(arr, 0, 2)
input_tensor = torch.from_numpy(arr).float().unsqueeze(0)

net = InceptionResnetV2()
output_tensor = net(input_tensor)
