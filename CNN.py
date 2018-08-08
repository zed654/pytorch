import torch
import torch.nn.init
import torchvision

from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

## torch.device("cuda:0")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=10)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



##############################
########### Viewer ###########
##############################

import matplotlib.pylab as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(''.join('%5s' % classes[labels[j]] for j in range(4)))

#####################################
# Define Couvolution Neural Network #
#####################################

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# Python은 Class에서 self라는 방법을 사용.
#   클래스의 입력이 self로 대체된다고 생각하면 됨.
class Net(nn.Module):                       # 얘는 nn 클래스에 Module 클래스 (이중 클래스)를 상속받은 것.
    def __init__(self):                     # 이 self는 nn.Module을 뜻함.
        super(Net, self).__init__()         # super()는 상속에 대한 오버라이딩을 무시하겠다는 뜻
                                            #   Super()와 Super(Net, self)는 같은 뜻 이다.
                                            #   즉, Net이 상속받은(nn.Module로부터) 인스턴스의 메소드(여기선 __init__)를 사용하겠다는 뜻

        self.conv1 = nn.Conv2d(3, 24, 5)    # 이건 nn.Module.conv1을 뜻 함.
        self.b1 = nn.BatchNorm2d(24)        # 이건 nn.Module.b1을 뜻 함.
        self.pool = nn.MaxPool2d(2, 2)      # 이건 nn.Module.pool을 뜻 함.

        self.conv2 = nn.Conv2d(24, 64, 5)   # 이건 nn.Module.conv2을 뜻 함.
        self.b2 = nn.BatchNorm2d(64)        # 이건 nn.Module.b2을 뜻 함.

        self.fc1 = nn.Linear(64 * 5 * 5, 240)   # 이건 nn.Module.fc1을 뜻 함.
        self.fc2 = nn.Linear(240, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.b1((self.conv1(x)))))
        x = self.pool(F.relu(self.b2(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net



#############################
########### optim ###########
#############################

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())        # learning_rate default value is 0.01

for epoch in range(10):
    running_loss = 0.0                          # loss 값
    for i, data in enumerate(trainloader, 0):
        print('%d kkk' % i)

        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()                   # Weight의 gradient 초기화

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()                         # Gradient 계산
        optimizer.step()

        running_loss += loss.data[0]            # loss값 갱신

        if i % 128 == 127:  # print every 2000 mini-batches
            print('[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, running_loss / 128))
            running_loss = 0.0

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy 1000 test images : %d %%' % (100 * correct / total))

print('Finished Training')

torch.save(net.state_dict(), 'CNN_.pkl')




