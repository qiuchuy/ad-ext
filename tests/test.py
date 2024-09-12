import ailang as al
import ailang.nn as nn
import numpy as np


class net(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=2)
        self.maxpool2d = nn.Maxpool2d(3, 3)

    def __call__(self, x):
        y = self.conv(x)
        print(y.shape)
        z = self.maxpool2d(y)
        return z


N = net()
arr = np.random.randn(1, 3, 2, 2).astype(dtype=np.float32)
x = al.from_numpy(arr)

mean = al.standard.mean(x, [1])
mean_n = np.mean(arr, 1)
print(mean)
print("@", mean_n)
