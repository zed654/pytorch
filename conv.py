import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):

    def __init__(self):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in
        # here
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)    # torch.nn.Module.conv1 이라는 변수를 만든 것.
        self.pool1 = nn.MaxPool2d(2, 2)     # 가로x세로 2x2의 maxpool작업이라는 의미인듯
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)       # 마지막 단인 Fully connected 구현
        self.fc2 = nn.Linear(50, 10)        # 마지막 단인 Fully connected 구현

    # it's the forward function that defines the network structure
    # we're accepting only a single input in here, but if you want,
    # feel free to use more
    def forward(self, input):
        x1 = self.pool1(F.relu(self.conv1(input)))
        x2 = self.pool2(F.relu(self.conv2(x1)))

        # in your model definition you can go full crazy and use arbitrary
        # python code to define your model structure
        # all these are perfectly legal, and will be handled correctly
        # by autograd:
        # if x.gt(0) > x.numel() / 2:
        #      ...
        #
        # you can even do a loop and reuse the same module inside it
        # modules no longer hold ephemeral state, so you can use them
        # multiple times during your forward pass
        # while x.norm(2) < 10:
        #    x = self.conv1(x)

        x3 = x2.view(x2.size(0), -1)
        x4 = F.relu(self.fc1(x3))       # 여기서 torch.nn.ReLU(self.fc1(x3))은 안먹힘. 얘는 torch.nn.Sequential() 에서만 쓸 수 있는듯.
        result_pred = F.relu(self.fc2(x4))
        return result_pred

net = MNISTConvNet()
print(net)

input = torch.randn(1, 1, 28, 28)   # (nSamples, nChannels, Height, Width)
out = net(input)
print(out.size())

target = torch.tensor([3], dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()  # LogSoftmax + ClassNLL Loss
err = loss_fn(out, target)
err.backward()

print(err)

print(net.conv1.weight.grad.size())

print(net.conv1.weight.data.norm())  # norm of the weight
print(net.conv1.weight.grad.data.norm())  # norm of the gradients