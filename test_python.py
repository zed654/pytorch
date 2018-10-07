import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

import torchvision.transforms as transforms

import time
import numpy as np

# 415 397
# 414 349
# 405 254
# 1571 302

def show_img_at_tensor(img_tensor_):
    img_PIL_ = ToPILImage()(img_tensor_)
    img_PIL_.show()

# PIL 형식으로 image 불러오기
img_path = '/Users/CHP/Lane_detector_pytorch/sample/'
img_name = '00000098.jpg'
img_PIL = Image.open(img_path + img_name)
w, h = img_PIL.size

# img_PIL_resize = img_PIL.resize((960, 604))
img_PIL_resize = img_PIL.resize((int(w/2), int(h/2)))

# PIL 형식의 이미지를 tensor 형식으로 바꾸어 저장하기
img_Tensor = ToTensor()(img_PIL_resize)

# tensor 형식의 이미지를 show하기
show_img_at_tensor(img_Tensor)
# area = (50, 50, 100, 100)#1280, 960)
# img_PIL_ROI = img_PIL.crop((414-30, h - (349+30), 414+30, h - (349-30)))
# img_PIL_ROI = img_PIL.crop((415-30, h - (397+30), 415+30, h - (397-30)))
# img_PIL_ROI = img_PIL.crop((405-30, h - (254+30), 405+30, h - (254-30)))
img_PIL_patch = img_PIL_resize.crop(((405-30)/2, (h - (254+30))/2, (405+30)/2, (h - (254-30))/2))
# img_PIL_ROI = img_PIL_resize.crop(((1571-30)/2, (h - (302+30))/2, (1571+30)/2, (h - (302-30))/2))

# 현재 1980 1208 기준 30x30임
# 그러면 990 x 604 기준 15x15
img_PIL_patch.show()

# space 3개로 점들 구분
# space 1개로 width, height 구분
txt_path = '/Users/CHP/Lane_detector_pytorch/sample/txt/'
txt_name = 'L000000098.txt'
gt_txt = open(txt_path + txt_name, 'rt')
gt_data_line = gt_txt.readline()
gt_data = gt_data_line.split("   ")
len(gt_data)            # 좌표 개수 + 1 개가 나옴.
label = gt_data[0]      # L0이 들어감.
coordinates = {c : gt_data[c+1].split(" ") for c in range(len(gt_data)-1)}   # L0의 좌표들이 들어감
# float(coordinates[0][0]) .. 첫 번째의 x좌표
# float(coordinates[0][1]) .. 첫 번째의 y좌표
# float(coordinates[1][0]) .. 두 번째의 x좌표
# float(coordinates[1][1]) .. 두 번째의 y좌표






#
# # 모든 이미지 파일의 이름을 읽고 저장하는 작업
# img_path = '/Users/CHP/Lane_detector_pytorch/sample/img'
# for root, dirs, files in os.walk(img_path):
#     for t in files:
#         # full_fname = dict(zip([tmp],[i for i in [os.path.join(root, fname)]]))
#         full_fname = os.path.join(root, t)
#         print(full_fname)
#
# # GT txt에서 txt파일을 찾기 위해 이미지 파일로부터 이름을 가져온다.
# fname = []
# for i in range(len(files)):
#     fname.append(files[i][-14:-4])
#
# # GT의 종류인 L2 ~ R2를 저장해둔 변수
# label_name = dict(zip([x for x in range(6)], [i for i in ['L2', 'L1', 'L0', 'R0', 'R1', 'R2']]))
#
# # txt_name에 이미지 이름에 따른 모든 경우의 gt파일이 추가된다.
# txt_name = []
# for t in range(len(files)):
#     for i in range(len(label_name)):
#         txt_name.append(label_name[i] + fname[t] + '.txt')
#












# GT txt에 img files의 이름이 들어가고, img files보다 gt txt들의 개수가 더 많으므로
# gt txt의 폴더를 읽어 txt파일들의 이름을 변수에 저장한 후
# 조금 수정하여 img files의 이름도 저장한 코드이다.
import os

# 해당 이름을 기준으로 txt를 실행
txt_path = '/Users/CHP/Lane_detector_pytorch/sample/txt/'
img_path = '/Users/CHP/Lane_detector_pytorch/sample/'

for root, dirs, txt_files in os.walk(txt_path):
    for t in txt_files:
        full_fname = os.path.join(root, t)
        print(full_fname)

# GT txt에서 txt파일을 찾기 위해 이미지 파일로부터 이름을 가져온다.
#   txt_name과 img_name의 순서는 일치하다. (txt_name[3] = img_name[3])
txt_name = []
for i in range(len(txt_files)):
    txt_name.append(txt_files[i])

img_name = []
for i in range(len(txt_files)):
    img_name.append(txt_files[i][-12:-4] + '.jpg')

# img_name = list(set(img_name))

# 좌표를 coordinates[num][좌표카운팅][좌표의 x점][좌표의 y점] 이 된다.
#   여기서 len(coordinates[num])을 통해 해당 점에서 좌표가 카운팅 되었는지를 확인해보고 넘어가야 오류가 안뜬다.
coordinates = []
# gt_txt_file = []
# gt_data_line = []
# gt_data = []
for i in range(len(txt_files)):
    # txt의 path를 통해 파일을 읽음
    gt_txt_file = open(txt_path + txt_name[i], 'rt')

    # 읽은 파일에서 한 줄을 읽어서 저장
    gt_data_line = gt_txt_file.readline()
    print(gt_data_line)

    # 데이터의 첫 줄
    gt_data = gt_data_line.split("   ")
    len(gt_data)            # 좌표 개수 + 1 개가 나옴.

    label = gt_data[0]      # L0이 들어감.
    print(label)

    # coordinates에 저장할 좌표변수 생성 및 txt파일로부터 읽어들여서 저장
    coordinates_tmp = []
    for c in range(len(gt_data)-1):
        coordinates_tmp.append(gt_data[c+1].split(" "))
    coordinates.append(coordinates_tmp)

# 요약
#   txt_name[num]
#   img_name[num]
#   coordinates[num][좌표카운팅][좌표의x점][좌표의y점]

# img_name과 coordinates를 이용해 GT patch를 추출해낸다.
# GT patch는 따로 저장해둬야함..?


def putpixel_area(img_, x_, y_):
    for i in range(20):
        for j in range(20):
            img_.putpixel((x_ + j , y_ + i), (128, 128, 128))

# patch save
img_PIL_patch = []
for gt_num in range(len(coordinates)):
    img_PIL = Image.open(img_path + img_name[gt_num])
    w, h = img_PIL.size # 1920 x 1208
    img_PIL_resize = img_PIL.resize((int(w/2), int(h/2)))

    for coord_num in range(len(coordinates[gt_num])):
        print(gt_num, coord_num)

        # resized된 이미지에서 patch size를 15x15로
        # x1 = int((int(coordinates[gt_num][coord_num][0]) - 15)/2)
        # y1 = int((int(coordinates[gt_num][coord_num][1]) - 15)/2 + 302)
        # x2 = int((int(coordinates[gt_num][coord_num][0]) + 15)/2)
        # y2 = int((int(coordinates[gt_num][coord_num][1]) + 15)/2 + 302)
        # patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

        # resized된 이미지에서 patch size를 30x30로
        x1 = int(int(coordinates[gt_num][coord_num][0])/2 - 15)
        y1 = int(int(coordinates[gt_num][coord_num][1])/2 - 15 + 302)
        x2 = int(int(coordinates[gt_num][coord_num][0])/2 + 15)
        y2 = int(int(coordinates[gt_num][coord_num][1])/2 + 15 + 302)
        patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

        # 원본 이미지에서 patch size를 120x120로
        # x1 = (int(coordinates[gt_num][coord_num][0]) - 60)
        # y1 = (int(coordinates[gt_num][coord_num][1]) - 60) + 604
        # x2 = (int(coordinates[gt_num][coord_num][0]) + 60)
        # y2 = (int(coordinates[gt_num][coord_num][1]) + 60) + 604
        # patch_tmp = img_PIL.crop((x1, y1, x2, y2))

        # 원본 이미지에 GT x, y좌표를 그리는 코드
        # tmp_x = int(coordinates[gt_num][coord_num][0])
        # tmp_y = int(coordinates[gt_num][coord_num][1])
        # putpixel_area(img_PIL, tmp_x, tmp_y + 604)
        # img_PIL.show()

        img_PIL_patch.append(patch_tmp)


for i in range(len(img_PIL_patch)):
    img_PIL_patch[i].show()




# txt_name = 'L000000098.txt'

# R000000098.txt 출력
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