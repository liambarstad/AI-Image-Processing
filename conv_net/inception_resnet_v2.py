import torch
import torch.nn as nn
import torch.nn.functional as F
from net.stem import Stem
from net.inception_resnet_a import InceptionResnetA
from net.inception_resnet_b import InceptionResnetB
from net.inception_resnet_c import InceptionResnetC
from net.reduction_a import ReductionA
from net.reduction_b import ReductionB

class InceptionResnetV2(nn.Module):

    def __init__(self):
        super(InceptionResnetV2, self).__init__()
        self.stem = Stem()
        self.inception_a1 = InceptionResnetA()
        self.inception_a2 = InceptionResnetA()
        self.inception_a3 = InceptionResnetA()
        self.inception_a4 = InceptionResnetA()
        self.inception_a5 = InceptionResnetA()

        self.reduction_a = ReductionA(192, 192, 256, 384)

        self.inception_b1 = InceptionResnetB()
        self.inception_b2 = InceptionResnetB()
        self.inception_b3 = InceptionResnetB()
        self.inception_b4 = InceptionResnetB()
        self.inception_b5 = InceptionResnetB()
        self.inception_b6 = InceptionResnetB()
        self.inception_b7 = InceptionResnetB()
        self.inception_b8 = InceptionResnetB()
        self.inception_b9 = InceptionResnetB()
        self.inception_b10 = InceptionResnetB()
        
        self.reduction_b = ReductionB()

        self.inception_c1 = InceptionResnetC()
        self.inception_c2 = InceptionResnetC()
        self.inception_c3 = InceptionResnetC()
        self.inception_c4 = InceptionResnetC()
        self.inception_c5 = InceptionResnetC()
    
    def forward(self, image):
        x = self.stem(image) 
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        x = self.inception_a4(x)
        x = self.inception_a5(x)

        x = self.reduction_a(x)

        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        x = self.inception_b5(x)
        x = self.inception_b6(x)
        x = self.inception_b7(x)
        x = self.inception_b8(x)
        x = self.inception_b9(x)
        x = self.inception_b10(x)

        x = self.reduction_b(x)

        x = self.inception_c1(x)
        x = self.inception_c2(x)
        x = self.inception_c3(x)
        x = self.inception_c4(x)
        x = self.inception_c5(x)
        return x
        # average pooling
        # dropout (keep 0.8)
        # softmax
        # ^^^^ this can be replaced by YOLO ^^^
