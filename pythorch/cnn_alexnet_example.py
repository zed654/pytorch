'''

    This code is CNN AlexNet Model writed by MLP Example style

'''
import torch
import torch.nn  as nn
from  torchvision import transforms,utils
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import pdb

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
#device = torch.device("cuda:0")

class AlexNet(torch.nn.Module):

    def __init__(self, num_classes=5):

        super(AlexNet, self).__init__()

        self.feature = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(256*6*6),
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            # torch.nn.BatchNorm1d(4096),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), 256*6*6)  #    x = torch.randn(2, 2, 1)     x.size(0)  y = x.view(1, 4)
        x = self.classifier(x)
        #print(x.size())
        return x


x = torch.randn(12, 3, 224, 224).to(device)    # x is equal to inputs
y = torch.randn(12, 5).to(device)


net = AlexNet().to(device)
#net = AlexNet()

learning_rate = 0.0002

criterion = torch.nn.MSELoss(size_average=False, reduce=True)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)   # 왠지 얘도 클래스안에 넣을 수 있을것같다.


for t in range(500):
    y_pred = net.forward(x).to(device)  # y_pred is equal to outputs

    optimizer.zero_grad()

    loss = criterion(y_pred, y)
    print(t, loss.item())

    loss.backward()

    optimizer.step()
