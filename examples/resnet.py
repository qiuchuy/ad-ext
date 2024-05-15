from typing import Any
import ailang as al
import ailang.nn as nn


class myConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5)
        self.bn1 = nn.BatchNorm(out_channels)
        # x = al.tensor((1,3,224,224), "Float")
        # print(type(c1(x)))
        # print(al.compile_ir(c1(x),x))

    def __call__(self, x):
        # basicblcok
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.bn1(out2)
        # basicblcok
        out3 = self.conv1(out2)
        out3 = self.conv2(out3)
        out3 = self.bn1(out3)
        # basicblcok
        out3 = self.conv1(out3)
        out3 = self.conv2(out3)
        out3 = self.bn1(out3)
        # basicblcok
        out3 = self.conv1(out3)
        out3 = self.conv2(out3)
        out3 = self.bn1(out3)
        # print(out3.shape())
        return out3


model_ = myConv(224, 224, 3)

import numpy as np

# print(model_.parameters())
y = al.zeros((1, 3, 244, 244))
z = model_(y)
print(z.shape)
