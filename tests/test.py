import ailang as al
import ailang.nn as nn
import numpy as np
import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv1.weight
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(residual)
        x += shortcut
        x = self.relu(x)

        return x


class alBasicResnet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.Batchnorm2d(out_channels)
        self.relu = nn.Relu()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.Batchnorm2d(out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride),
                nn.Batchnorm2d(out_channels),
            )

    def __call__(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(residual)
        x = al.standard.add(x, shortcut)
        x = self.relu(x)
        return x


import torch

T = ResidualBlock(3, 3)
N = alBasicResnet(3, 3)
arr = np.random.randn(1, 3, 224, 224).astype(dtype=np.float32)
x = al.from_numpy(arr)
t = torch.from_numpy(arr)
print(T(t).shape)
# mean = al.standard.var(x, [0, 2, 3])
# mean_n = np.var(arr, 1)
# mean_t = torch.var(t, [0, 2, 3])
# print(mean)
# print(mean_n)
# print(mean_t)
print(N(x).shape)
