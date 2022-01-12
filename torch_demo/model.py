from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import numpy as np


class ExModel(nn.Module):
    def __init__(self):
        super(ExModel, self).__init__()
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
net = ExModel()
x = np.ones(3,32,32)

