import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64)

class linearExModel(nn.Module):
    def __init__(self):
        super(linearExModel, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

linear = linearExModel()
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = linear(output)
    print(output.shape)