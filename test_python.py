from PIL import Image

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import time
import numpy as np
import matplotlib.pylab as plt

def show_Tensor(img):
    PIL_img = ToPILImage()(img)
    PIL_img.show()

path = '/Users/CHP/Lane_detector_pytorch/sample/00000098.jpg'

PIL_input_img = Image.open(path)
# PIL_input_img.show()
w, h = PIL_input_img.size
area = (0, 800, w, h)         # (좌측위x, 좌측위y, 우측아래x, 우측아래y)
PIL_img = PIL_input_img.crop(area)
PIL_img.show()

w2, h2 = PIL_img.size
# img = Image.open(path)
# area = (400, 400, 800, 800)
# cropped_img = img.crop(area)
# cropped_img.show()

# Tensor_img = ToTensor()(PIL_img)
# show_Tensor(Tensor_img)

# mi = Image.open('/Users/CHP/Lane_detector_pytorch/sample/00000098.jpg')
# mi.show()


fdsfasf=3

# to_img = ToPILImage()
#
# # a = torch.FloatTensor(3, 128, 128)
# a = torch.randint(0, 255, (3, 128, 128))
# a = a.type_as(torch.FloatTensor())
# b = to_img(a)
# b.show()





'''
import os

test_1 = {'a' : '3'}
test_2 = {b : '3' for b in range(2)}
test_3 = {c : os.path.join('path_name', c) for c in ['what', 'how']}

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
'''