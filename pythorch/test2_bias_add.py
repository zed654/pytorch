import torch
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime

start = datetime.now()

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

batch_size = 1
bias = 1

input_data = 3
output_data = 3

hidden_layer_n1 = 4 + bias
hidden_layer_n2 = 6 + bias

#x = Variable(torch.randn(batch_size, input_data, device=device, dtype=dtype), requires_grad=True)
x = Variable(torch.Tensor([[0.7, 0.5, bias]]), requires_grad=False)#.cuda()
#x = torch.Tensor([[0.7, 0.5]])
#y = Variable(torch.randn(batch_size, output_data, device=device, dtype=dtype), requires_grad=True)
y = Variable(torch.Tensor([[1., 2., 3.]]), requires_grad=False)#.cuda()
#y = torch.Tensor([[1., 2., 3.]])

w1 = Variable(torch.randn(input_data, hidden_layer_n1, device=device, dtype=dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer_n1, hidden_layer_n2, device=device, dtype=dtype), requires_grad=True)
w3 = Variable(torch.randn(hidden_layer_n2, output_data, device=device, dtype=dtype), requires_grad=True)

l_r = 0.015

for i in range(500):
    x[0][2] = 1     # bias = 1
    f1 = torch.mm(x, w1)
    z1 = torch.clamp(f1, min=0, max=10)
    z1 = F.relu(f1)
    z1[0][4] = 1    # bias = 1
    f2 = torch.mm(z1, w2)
    z2 = torch.clamp(f2, min=0)
    z2 = F.relu(f2)
    z2[0][6] = 1    # bias = 1
    f3 = torch.mm(z2, w3)
    hypothesis = f3

    loss = (y - hypothesis).pow(2).sum()


    loss.backward() # 얠 사용하려면

    print(i, loss.item())

    '''
    with torch.no_grad():
        w1 -= l_r * w1.grad
        w2 -= l_r * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
    '''
    with torch.no_grad():
        w1 -= l_r * w1.grad
        w2 -= l_r * w2.grad
        w3 -= l_r * w3.grad

        #optimizer.zero_grad()
        w1.grad.zero_() # update된 grad를 초기화해주는 것. optimizer.zero_grad()로 대체가능?
        w2.grad.zero_()
        w3.grad.zero_()

print(datetime.now() - start)
