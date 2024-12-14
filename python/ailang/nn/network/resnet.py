import torch
from torchvision import models

test_device = torch.device('cuda')
torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
torch_resnet50.cuda().half().to(memory_format=torch.channels_last)
torch_resnet50.eval()
torch_resnet50.to(test_device)

def resnet50(x):
    return torch_resnet50(x)

