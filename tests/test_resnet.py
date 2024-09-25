# Torch
import torch
import ailang as al
import ailang.nn as nn
import torch.nn.functional as F
import numpy as np


class TorchResNet(torch.nn.Module):
    def __init__(self):
        super(TorchResNet, self).__init__()
        self.in_planes = 64

        # 初始的卷积和池化层
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64, momentum=None, affine=False)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.conv2 = torch.nn.Conv2d(
            64, 64, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(
            64, 64, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()

        # Layer 2
        self.conv4 = torch.nn.Conv2d(
            64, 128, stride=2, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(
            128, 128, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn5 = torch.nn.BatchNorm2d(128)

        # 下采样层，用于调整尺寸
        self.downsample2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(128),
        )
        self.relu3 = torch.nn.ReLU()

        # Layer 3
        self.conv6 = torch.nn.Conv2d(
            128, 256, stride=2, kernel_size=3, padding=1, bias=False
        )
        self.bn6 = torch.nn.BatchNorm2d(256)
        self.conv7 = torch.nn.Conv2d(
            256, 256, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn7 = torch.nn.BatchNorm2d(256)

        self.downsample3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(256),
        )
        self.relu4 = torch.nn.ReLU()

        # Layer 4
        self.conv8 = torch.nn.Conv2d(
            256, 512, stride=2, kernel_size=3, padding=1, bias=False
        )
        self.bn8 = torch.nn.BatchNorm2d(512)
        self.conv9 = torch.nn.Conv2d(
            512, 512, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn9 = torch.nn.BatchNorm2d(512)

        self.downsample4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
        )
        self.relu5 = torch.nn.ReLU()

    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Layer 1
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity  # 残差连接
        out = self.relu2(out)
        # Layer 2
        x = out
        identity2 = x
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu3(out)

        out = self.conv5(out)
        out = self.bn5(out)

        # 下采样并加上残差连接
        identity2 = self.downsample2(identity2)
        out += identity2
        out = self.relu3(out)
        # Layer 3
        x = out
        identity3 = x
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu4(out)

        out = self.conv7(out)
        out = self.bn7(out)

        identity3 = self.downsample3(identity3)
        out += identity3
        out = self.relu4(out)
        # Layer 4
        x = out
        identity4 = x
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu5(out)

        out = self.conv9(out)
        out = self.bn9(out)

        identity4 = self.downsample4(identity4)
        out += identity4
        out = self.relu4(out)
        return out


# 测试网络
model = TorchResNet()
model.eval()


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


pd = create_numpy_params(model)
# print(pd.keys())


class AilangResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        # 初始的卷积和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = al.from_numpy(pd["conv1.weight"])
        self.bn1 = nn.Batchnorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Maxpool2d(kernel_size=3, stride=2, padding=1)
        # layer 1
        self.conv2 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv2.weight = al.from_numpy(pd["conv2.weight"])
        self.bn2 = nn.Batchnorm2d(64)
        # 第二个卷积层
        self.conv3 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv3.weight = al.from_numpy(pd["conv3.weight"])

        self.bn3 = nn.Batchnorm2d(64)
        self.relu2 = nn.ReLU()
        # layer2
        self.conv4 = nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1, bias=False)
        self.conv4.weight = al.from_numpy(pd["conv4.weight"])
        self.bn4 = nn.Batchnorm2d(128)
        # 第二个卷积层
        self.conv5 = nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv5.weight = al.from_numpy(pd["conv5.weight"])

        self.bn5 = nn.Batchnorm2d(128)
        downsample = nn.Sequential(
            nn.Conv2d(
                self.in_planes,
                128 * 1,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.Batchnorm2d(128 * 1),
        )
        downsample.layers[0].weight = al.from_numpy(pd["downsample2.0.weight"])
        self.downsample2 = downsample  # 如果需要调整尺寸则添加下采样层
        self.relu3 = nn.ReLU()
        # Layer 3
        self.conv6 = nn.Conv2d(128, 256, stride=2, kernel_size=3, padding=1, bias=False)
        self.conv6.weight = al.from_numpy(pd["conv6.weight"])
        self.bn6 = nn.Batchnorm2d(256)
        # 第二个卷积层
        self.conv7 = nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv7.weight = al.from_numpy(pd["conv7.weight"])

        self.bn7 = nn.Batchnorm2d(256)
        downsample = nn.Sequential(
            nn.Conv2d(
                128,
                256 * 1,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.Batchnorm2d(256 * 1),
        )
        downsample.layers[0].weight = al.from_numpy(pd["downsample3.0.weight"])
        self.downsample3 = downsample  # 如果需要调整尺寸则添加下采样层
        self.relu4 = nn.ReLU()

        # Layer 4
        self.conv8 = nn.Conv2d(256, 512, stride=2, kernel_size=3, padding=1, bias=False)
        self.conv8.weight = al.from_numpy(pd["conv8.weight"])
        self.bn8 = nn.Batchnorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv9.weight = al.from_numpy(pd["conv9.weight"])

        self.bn9 = nn.Batchnorm2d(512)
        downsample = nn.Sequential(
            nn.Conv2d(
                256,
                512 * 1,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.Batchnorm2d(512 * 1),
        )
        downsample.layers[0].weight = al.from_numpy(pd["downsample4.0.weight"])
        self.downsample4 = downsample  # 如果需要调整尺寸则添加下采样层
        self.relu5 = nn.ReLU()

    @al.jit
    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = al.add(out, identity)
        out = self.relu2(out)
        x = out
        # layer2
        identity2 = x
        out = self.conv4(x)
        out = self.bn4(out)
        out = self.relu3(out)

        out = self.conv5(out)
        out = self.bn5(out)

        # 如果存在下采样，调整输入的尺寸
        identity2 = self.downsample2(x)
        out = al.add(out, identity2)
        out = self.relu3(out)

        # Layer 3
        x = out
        identity3 = x
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu4(out)

        out = self.conv7(out)
        out = self.bn7(out)

        identity3 = self.downsample3(identity3)
        out = al.add(out, identity3)
        out = self.relu4(out)
        # Layer 4
        x = out
        identity4 = x
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu5(out)

        out = self.conv9(out)
        out = self.bn9(out)

        identity4 = self.downsample4(identity4)
        out = al.add(out, identity4)

        out = self.relu4(out)
        return out


def numeric_check(a: al.array, b: np.ndarray):
    return np.allclose(a.tolist(), b.tolist(), rtol=1e-04, atol=1e-06)


def test_resnet():
    a_model = AilangResNet()
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    t = torch.from_numpy(x)
    a = al.from_numpy(x)
    a_model.eval()
    model.eval()
    a_out = a_model(a)
    t_out = model(t)
    assert numeric_check(a_out, t_out.detach().numpy())


if __name__ == "__main__":
    test_resnet()
