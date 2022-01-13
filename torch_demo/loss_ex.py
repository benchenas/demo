import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)


class lossExModel(nn.Module):
    def __init__(self):
        super(lossExModel, self).__init__()
        self.seq = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        return x
loss = nn.CrossEntropyLoss()
loss_ex = lossExModel()
for data in dataloader:
    img, target = data
    output = loss_ex(img)
    result_loss = loss(output, target)
    result_loss.backward()
    print(result_loss)
