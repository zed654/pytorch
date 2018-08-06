import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import torch

x = torch.randn(20, 30)
plt.figure(1)
plt.plot(x.numpy())


img = imread('hymenoptera_data/train/ants/0013035.jpg')
img_tensor = torch.Tensor(img)
plt.figure(2)
plt.imshow(img)
plt.show()

for t in ['abc', 'def']:
  print(t)


class _aa:
  class Module:
    def conv1(self):
      print("I'm conv1")
  def __init__(self):
    print("_aa 생성자")
  def conv1(self):
    print("conv1")
  def gnp(self):
    print("_aa")

class aa(_aa.Module):
  def gnp(self):
    print("im aa")
    self.conv1()

class a(aa):
  a_value = 33
  def __init__(self):
    print("a 생성자")
  def gnp(self):
    print("handsome")
    return 4

class b(a):
  def __init__(self):
    print("b 생성자")

  def gnp(self):
    print(self.a_value)
    print("존나")
    x = super(a, self).gnp()
    print(x)
    return 3

class c(a):
  def __init__(self):
    print("c 생성자")

b_m = b()

# print 1
print(b_m.gnp())

# print 2
#print(b.a_value)

class A:
  def __init__(self):
    print("Class A__init__()")


class B(A):
  def __init__(self):
    print("Class B__init__()")
    A.__init__(self)

class C(A):
  def __init__(self):
    print("Class C__init__()")
    A.__init__(self)

class D(B, C):
  def __init__(self):
    print("Class D__init__()")
    B.__init__(self)
    C.__init__(self)

d = D()