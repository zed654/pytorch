import torch
#from torch.autograd import Variable

dtype = torch.float
device = torch.device("cpu")
bias = 1
# device = torch.device("cuda:0")


#x = Variable(torch.randn(1, 4 + bias, device=device, dtype=dtype), requires_grad=False)
#y = Variable(torch.randn(1, 3, device=device, dtype=dtype), requires_grad=False)

x = torch.randn(1, 4 + bias,device=device, dtype=dtype, requires_grad=False)
y = torch.randn(1, 3, device=device, dtype=dtype, requires_grad=False)
'''
w1 = Variable(torch.randn(4 + bias, 8 + bias, device=device, dtype=dtype), requires_grad=True)
w2 = Variable(torch.randn(8 + bias, 6 + bias, device=device, dtype=dtype), requires_grad=True)
w3 = Variable(torch.randn(6 + bias, 3, device=device, dtype=dtype), requires_grad=True)
'''
w1 = torch.randn(4 + bias, 8 + bias, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(8 + bias, 6 + bias, device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn(6 + bias, 3, device=device, dtype=dtype, requires_grad=True)

r_l = 0.015

for i in range(500):
    #with torch.no_grad():
    #    with torch.enable_grad():
    #        w1.requires_grad = True

    #torch.set_grad_enabled(True)
    #x[0][4] = 1
    torch.set_grad_enabled(True)
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

    # 새로운 변수를 생성할 때 생성된 변수의 requires_grad가 False가 된다.
    #   만일 True가 되길원한다면, with torch.enable_grad():도 넣어주면 된다.
    #   w1 등의 웨이트를 갱신할 때는 grad를 False해주어야 한다.!!!
    '''
    with torch.no_grad():
        w1 -= r_l * w1.grad
        w2 -= r_l * w2.grad
        w3 -= r_l * w3.grad

        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
    '''
    #with torch.no_grad():
    #    with torch.enable_grad():
    #with torch.set_grad_enabled(False):
    torch.set_grad_enabled(False)
    w1 -= r_l * w1.grad
    w2 -= r_l * w2.grad
    w3 -= r_l * w3.grad

    w1.grad.zero_()
    w2.grad.zero_()
    w3.grad.zero_()


