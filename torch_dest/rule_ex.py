import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor(),download=True)

dateloader = DataLoader(dataset, batch_size=64)

class reluEx(nn.Module):
    def __init__(self):
        super(reluEx, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,x):
        x = self.sigmoid1(x)
        return x

relu = reluEx()
step = 0
writer = SummaryWriter('sigmoid')
for data in dateloader:
    imgs,targets = data
    output = relu(imgs)
    writer.add_images('input',imgs,step)
    writer.add_images('output', output, step)
    step += 1

writer.close()