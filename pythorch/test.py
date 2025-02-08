# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from datetime import datetime

start = datetime.now()

'''
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y
    #y_pred = x.mm(w1).clamp(min=0).mm(w2)
    h = x.mm(w1)            # 얘는 h = torch.mm(h, w1)과 같음
    h_relu = h.clamp(min=0) # 얘는 h_relu = torch.clamp(h, min=0, max=inf)와 같음
    y_pred = h_relu.mm(w2)  # 얘는 y_pred = torch.mm(h_relu, w2)와 같음

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
'''

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
# x = torch.randn(N, D_in, device=device, dtype=dtype)
x = Variable(torch.randn(N, D_in, device=device, dtype=dtype), requires_grad=True)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

print(datetime.now() - start)

# torch.Tensor(행, 열) # 초기화되지 않은 Tensor
# torch.rand(3, 3)  # 0~1의 랜덤한 값이 ddd배정
# torch.randn(3, 3) # 평균이 0이고 분산이 1인 normal distribution random value

# Numpy를 Tensor로
# a = np.array([1, 2, 3, 4]) 를 tensor로 바꾸려면
# b = torch.Tensor(a)

# Tensor에서 Numpy로
# a = torch.rand(3, 3)
# b = a.numpy()

# Tensor 형태 변환
# a = torch.rand(3, 3)
# a = a.view(1, 1, 3, 3)

# Tensor 합치기
# torch.cat((Tensor_A, Tensor_B), dim)
# inception, resnet에서 사용
# a = torch.randn(1, 1, 3, 3)
# b = torch.randn(1, 1, 3, 3)
# c = torch.cat((a, b), 0)

# Tensor 계산을 GPU로
# x = torch.rand(3, 3)
# y = torch.rand(3, 3)
# if torch.cuda.is_available():
#   x = x.cuda()
#   y = y.cuda()
#   sum = x+y

# torchvision               - opencv같은거
# torchvision.models        - vgg같이 유명한 모델들 들어있음
# torchvision.transforms    - 데이터 부족할 때, 이미지 변환에 사용
# torchvision.utils         - 이미지 저장 함수

# Type 변환
# x = x.type_as(torch.IntTensor())

# Tensor 기본형
# 기본형(torch.Tensor()는 float형으로, torch.FloatTensor()과 같다)

'''
start = datetime.now()

N, D = 3, 4

x = Variable(torch.randn(N, D), requires_grad=True)
y = Variable(torch.randn(N, D), requires_grad=True)
z = Variable(torch.randn(N, D), requires_grad=True)

a = x * y
b = a * z
c = torch.sum(b)

c.backward(gradient=torch.FloatTensor([1.0]))

print(x.grad)
print(y.grad)
print(z.grad)
print(datetime.now()-start)
'''

# 그 외
# Tensor a의 평균
#   a.mean()
# Tensor a의 원소합
#   a.sum()

'''
x = torch.randn(1, 10)
prev_h = torch.randn(1, 20)
W_h = torch.randn(20, 20)
W_x = torch.randn(20, 10)

i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h+h2h
next_h = next_h.tanh()

loss = next_h.sum()
'''

#print(x)
#print(prev_h)
#print(W_h)
#print(W_x)

#loss.backward()
