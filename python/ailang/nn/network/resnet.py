import torch
from torchvision import models

test_device = torch.device('cuda')
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.cuda().half().to(memory_format=torch.channels_last)
resnet50.eval()
resnet50.to(test_device)

def resnet(x):
    return resnet50(x)

