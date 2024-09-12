import ailang as al
import ailang.nn as nn
import numpy as np


class net(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=2)
        self.maxpool2d = nn.Maxpool2d(3, 3)
        self.bn = nn.Batchnorm2d(5)

    def __call__(self, x):
        y = self.conv(x)
        print(y.shape)
        z = self.maxpool2d(y)
        u = self.bn(z)
        return u


import torch

N = net()
arr = np.random.randn(1, 3, 224, 224).astype(dtype=np.float32)
x = al.from_numpy(arr)
t = torch.from_numpy(arr)
# mean = al.standard.var(x, [0, 2, 3])
# mean_n = np.var(arr, 1)
# mean_t = torch.var(t, [0, 2, 3])
# print(mean)
# print(mean_n)
# print(mean_t)
print(N(x))
