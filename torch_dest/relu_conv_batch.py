import numpy as np
import paddle
from paddle.vision import transforms, datasets
import paddle.nn as nn
import paddle.nn.functional as F

class EvoCNNModel(paddle.nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()

        # resnet and densenet unit
        self.bn1 = nn.BatchNorm2D(3)
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, bias_attr=False, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2D(16)
        # self.conv2 = nn.Conv2D(in_channels=16, out_channels=64, bias_attr=False, kernel_size=3, padding=1)


        # linear unit
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out_0 = self.bn1(x)
        out_1 = F.relu(out_0)
        out_2 = self.conv1(out_1)

        out = out_2

        out = paddle.reshape(out, [out.shape[0], -1])
        print(out.shape)
        out = self.linear(out)
        return out





normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

# define transform
train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

dataset = datasets.Cifar10(mode='train',
                           download=True, transform=train_transform,
                           )

data_loader = paddle.io.DataLoader(
    dataset, batch_size=32, shuffle=False
)
paddle.set_device('gpu:2')
net = EvoCNNModel()
for epoch_id in range(50):
    net.train()
    optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                     parameters=net.parameters())
    train_loss = list()
    train_acc = list()

    for i, data in enumerate(data_loader()):
        inputs, labels = data
        inputs = paddle.to_tensor(inputs)

        labels = paddle.to_tensor(labels)
        labels = paddle.reshape(labels, [-1, 1])

        predicts = net(inputs)  # 前向传播

        loss = F.cross_entropy(predicts, labels)
        train_loss.append(loss.numpy())
        loss.backward()  # 反向传播
        optimizer.step()
        optimizer.clear_grad()
        acc = paddle.metric.accuracy(input=predicts, label=labels)
        train_acc.append(acc.numpy())
        if i == 1:
            break
    loss_mean = np.mean(train_loss)
    acc_mean = np.mean(train_acc)
    print('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f' % (1, loss_mean, acc_mean))
