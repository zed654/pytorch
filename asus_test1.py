import torch
from torch.autograd import Variable

dtype = torch.float
device = torch.device("cpu")
bias = 1
# device = torch.device("cuda:0")


x = Variable(torch.randn(1, 4 + bias, device=device, dtype=dtype), requires_grad=False)
y = Variable(torch.randn(1, 3, device=device, dtype=dtype), requires_grad=False)

w1 = Variable(torch.randn(4 + bias, 8 + bias, device=device, dtype=dtype), requires_grad=True)
w2 = Variable(torch.randn(8 + bias, 6 + bias, device=device, dtype=dtype), requires_grad=True)
w3 = Variable(torch.randn(6 + bias, 3, device=device, dtype=dtype), requires_grad=True)

r_l = 0.015

for i in range(500):
    #x[0][4] = 1
    f1 = torch.mm(x, w1)
    z1 = torch.clamp(f1, min=0)
    #z1[0][8] = 1
    f2 = torch.mm(z1, w2)
    z2 = torch.clamp(f2, min=0)
    #z2[0][6] = 1
    f3 = torch.mm(z2, w3)
    hypothesis = f3

    loss = (hypothesis - y).pow(2).sum()

    print(i, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= r_l * w1.grad
        w2 -= r_l * w2.grad
        w3 -= r_l * w3.grad

        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
