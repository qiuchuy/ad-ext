# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_parameters_same(model, params_dict):
    model_params = (
        model.named_parameters()
        if isinstance(model, torch.nn.Module)
        else model.parameters()
    )
    for name, param in model_params:
        if name in params_dict:
            if isinstance(param, torch.Tensor):
                param.data = torch.from_numpy(params_dict[name])
            else:
                param.data = al.from_numpy(params_dict[name])


def create_numpy_params(model):
    """为模型生成一个 NumPy 随机参数字典"""
    params_dict = {}
    model_params = (
        model.named_parameters()
        if isinstance(model, torch.nn.Module)
        else model.parameters()
    )
    for name, param in model_params:
        params_dict[name] = param.detach().numpy()
    return params_dict


# 定义基本的残差块
class TorchBasicBlock(torch.nn.Module):
    expansion = 1  # 扩展比例，ResNet18/34的为1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(TorchBasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        # 第二个卷积层
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample  # 如果需要调整尺寸则添加下采样层
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样，调整输入的尺寸
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # 残差连接
        out = self.relu(out)

        return out


# 定义ResNet18结构
class TorchResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(TorchResNet, self).__init__()
        self.in_planes = 64

        # 初始的卷积和池化层
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的四个主要阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化层和全连接层
        self.avgpool = torch.nn.AvgPool2d(7, 1, 0)
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果输入和输出尺寸不同（通道数或空间维度），则需要下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_planes, planes, stride, downsample)
        )  # 第一层包含可能的下采样
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))  # 其他层保持相同尺寸

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


# 创建ResNet18模型
def TorchResNet18(num_classes=1000):
    return TorchResNet(TorchBasicBlock, [2, 2, 2, 2], num_classes)


t_model = TorchResNet18()
params_dict = create_numpy_params(t_model)
# print(params_dict.keys())
###################################################
#####################AILANG########################
###################################################


from typing import Any
import ailang as al
import ailang.nn as nn


class AilangBasicBlock(nn.Module):
    expansion = 1  # 扩展比例，ResNet18/34的为1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(AilangBasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_planes, planes, stride=stride, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.Batchnorm2d(planes)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            planes, planes, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.Batchnorm2d(planes)
        self.downsample = downsample  # 如果需要调整尺寸则添加下采样层
        self.relu = nn.ReLU()

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样，调整输入的尺寸
        if self.downsample is not None:
            identity = self.downsample(x)
        out = al.standard.add(out, identity)
        out = self.relu(out)

        return out


class AilangResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(AilangResNet, self).__init__()
        self.in_planes = 64

        # 初始的卷积和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.Batchnorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Maxpool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的四个阶段，每个阶段包含多个残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化层和全连接层
        self.avgpool = nn.Maxpool2d(7, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 判断是否需要下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.Batchnorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def __call__(self, x):
        # 前向推理流程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过每个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # # # 全局池化和全连接
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)  # 展平
        # x = self.fc(x)

        return x


# 定义 ResNet18 的结构
def AilangResNet18(num_classes=1000):
    return AilangResNet(AilangBasicBlock, [2, 2, 2, 2], num_classes)


def test_basic_block():

    a_model = AilangBasicBlock(3, 64)
    t_model = TorchBasicBlock(3, 64)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)


def test_resnet():
    a_model = AilangResNet18()
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    t = torch.from_numpy(x)
    a = al.from_numpy(x)
    t_model.eval()
    a_model.eval()
    # a_out = a_model(a)
    t_out = t_model(t)
    print(t_out.shape)


# 测试推理过程
if __name__ == "__main__":
    # model = TorchResNet18(num_classes=1000)  # 假设有1000类
    # x = torch.randn(1, 3, 224, 224)  # 输入是224x224的彩色图像
    # model.eval()
    # output = model(x)
    # print(output)  # 输出是(batch_size, num_classes)，即(1, 1000)

    test_resnet()
