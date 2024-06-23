import numpy as np
import ailang as al

# print("============")
# a = np.random.randn(2, 2, 2)
# b = al.from_numpy(a)
# c = al.cos(b)
# a = al.ones((3, 4))
# print(a)
# a1 = np.random.randn(2,1,2)
# b1 = al.from_numpy(a1)
# c1 = al.cos(b1)
# e= al.zeros((2,3))
# print(e)
# print(c)
# print(c1)

# d = al.Add(c,c1)
# print(d)
#################


# a = np.random.randn(2, 3, 5)
# b = al.from_numpy(a)
# c = al.mean(b, False)
# e = al.ones((2, 2))
# f = al.Add(e, c)
# print(f)
# ###
# a = np.ones((2, 3, 5, 5))
# b = al.from_numpy(a)
# c = np.ones((5, 3, 3, 3))
# d = al.from_numpy(c)
# e = al.conv(b, d, (1, 1), (1, 1), (1, 1))
# print(e)
# print(e.shape)

# # //broadcast
# a = al.ones((1, 2))
# b = al.ones((2, 2))
# c = al.Add(a, b)
# print(c)

# ##
# a = np.random.randn(2, 3, 5)
# b = al.from_numpy(a)
# c = al.mean(b, (1, 2, 0), False)
# print(c)
a = np.random.randn(2, 3)
b = al.from_numpy(a)
c = np.random.randn(2, 3)
d = al.from_numpy(c)
e = al.Sub(b, d)
f = al.square(e)
# print(e)
# print(f)
# a = np.random.randn(2, 3, 5)
# b = al.from_numpy(a)
# c = al.var(b, False)
# print(c)
# c = al.Add(a, b)
# d = al.sqrt(c)
# print(al.Multiply(d, d))
# e = al.zeros((1,))
# f = al.maximum(d, al.zeros((1,)))
# print(f)


# a = al.ones((1,))
# b = al.ones((3, 2))
# c = al.ones((2, 3))
# d = al.Multiply(b, c)
# print(d)

from typing import Any
import ailang as al
import ailang.nn as nn


class BasicBlock(nn.Module):
    expansion = 1  # expand scale

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm(out_channels)
        self.downsample = downsample

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = al.Add(identity, out)
        out = nn.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: BasicBlock, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 4
        self.conv1 = nn.Conv2d(4, 4, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(4)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 4, layers[0])
        self.layer2 = self._make_layer(block, 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 3, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 4, layers[3], stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(242 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)
            # x = self.avgpool(x)
        x = al.flatten(x)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


if __name__ == "__main__":
    model = resnet18(num_classes=50)
    x = al.from_numpy(np.random.randn(1, 4, 224, 224))
    print(model(x))
