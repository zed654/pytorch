
# torch.Tensor(행, 열) # 초기화되지 않은 Tensor
# torch.rand(3, 3)  # 0~1의 랜덤한 값이 ddd배정
# torch.randn(3, 3) # 평균이 0이고 분산이 1인 normal distribution random value

# Numpy를 Tensor로
#   a = np.array([1, 2, 3, 4]) 를 tensor로 바꾸려면
#   b = torch.Tensor(a)
#   b = a.Tensor() 도 될듯 -> 안됌안됌안됌안됌

# Tensor에서 Numpy로
#   a = torch.rand(3, 3)
#   b = a.numpy()

# Tensor 형태 변환
#   a = torch.rand(3, 3)
#   a = a.view(1, 1, 3, 3)

# Tensor 합치기 (이어붙이기)
#   torch.cat((Tensor_A, Tensor_B), dim)
#   inception, resnet에서 사용
#   a = torch.randn(1, 1, 3, 3)
#   b = torch.randn(1, 1, 3, 3)
#   c = torch.cat((a, b), 0)

# Tensor 계산을 GPU로
#   x = torch.rand(3, 3)
#   y = torch.rand(3, 3)
#   if torch.cuda.is_available():
#       x = x.cuda()
#       y = y.cuda()
#       sum = x+y

# torchvision               - opencv같은거
# torchvision.models        - vgg같이 유명한 모델들 들어있음
# torchvision.transforms    - 데이터 부족할 때, 이미지 변환에 사용
# torchvision.utils         - 이미지 저장 함수

# Type 변환
#   x = x.type_as(torch.IntTensor())

# Tensor 기본형
# 기본형(torch.Tensor()는 float형으로, torch.FloatTensor()과 같다)

# 특정 숫자 범위로 값 선언하기 (0 ~ 254값, 3 Tensor로 선언)
#   a = torch.randint(0, 255, (3, 960, 1280))

# 시간 카운트하기
#   from datetime import datetime
#   start = datetime.now()
#
# print(datetime.now()-start)


# Tensor 나누기 (chunk, split)
# a = torch.rand(10, 4)
# >>> a.size()
# >>>   torch.Size([10, 4])
# b = torch.chunk(a, 5, 0)      # 0번째 탠서를 5등분 하겠다는 의미
# >>> b[0].size()
# >>>   torch.Size([2, 4])
# >>> b[4].size()
# >>>   torch.Size([2, 4])
#
#   split도 마찬가지인데, chunk은 5등분 하는 것 이고(10이면 b[0]~b[3]), split은 5로 나누겠다는 의미(10이면 b[0]. b[1])


# 그 외
# Tensor a의 평균
#   a.mean()
# Tensor a의 원소합
#   a.sum()






#                                       #
#   Pytorch에서 이미지 read/write/modify   #
#        (Pillow 사용)                   #

# Pytorch에서 이미지를 사용하기 위해서는 PIL이미지를 사용하는것이 좋다.
# 이유는 Python에서 기본적으로 지원하는 함수이며, OpenCV에 있는 기본 기능 모두 구현이 가능하다.

# Python에서 이미지 처리는 PIL 사용
#   from PIL import Image

# PIL에서 이미지 read
#   img_PIL = Image.open('/Users/.../~.jpg')

# PIL에서 이미지 show
#   img_PIL.show()

# PIL에서 ROI 자르기 (좌측위x, 좌측위y, 우측아래x, 우측아래y)
#   img_PIL_ROI = img_PIL.crop((50, 50, 100, 100))

# PIL Image의 width, height 정보 받아오기
#   w, h = img_PIL.size

# PIL Image에서 Channel Split하기
#   r, g, b = img_PIL.split()

# PIL Image의 크기 정보 보기
#   img_PIL.size

# PIL Image 리사이즈하기
#   img_PIL_resized = img_PIL.resize((1280, 960))

# PIL Image To Tensor
#   from torchvision.transforms import ToTensor
#   to_Tensor = ToTensor()
#   img_Tensor = to_Tensor(img_PIL) or img_Tensor = ToTensor()(img_PIL)


# Tensor To PIL Image (Tensor의 dtype은 torch.FloatTensor() 이어야 한다.)
#   from torchvision.transforms import ToPILImage
#   to_PIL = ToPILImage()
#       img_Tensor = torch.randint(0, 254, (3, 960, 1280))
#       img_Tensor = img_Tensor.type_as(torch.FloatTensor())
#   img_PIL = to_PIL(img_Tensor)      or      img_PIL = ToPILImage()(img_Tensor)





# 숫자를 문자열로 바꾸기
#   num = 3
#   str = str(num)

# 문자열 앞에 0 채우기
#   num.zfill(8)
#   >>> 00000003



# 폴더내 모든 파일 읽기 (파이썬)
# import os
#
# for root, dirs, files in os.walk('/folder/path'):
#     for fname in files:
#         full_fname = os.path.join(root, fname)
#
#         print(full_fname)


# 배열을 동적으로 선언하기
#   fname = []
#   for i in range(10):
#       fname.append(i)
#


# 리스트에서 중복된 값 제거
#   list_1 = [1, 2, 3, 4, 5, 1, 2, 3]
#   list_2 = list(set(list_1))
#   print(list_2)
#   >>> [1, 2, 3, 4, 5]





# 쓰레드 사용하여 병렬연산 하기 (Pytorch)
# 해당 연산은 device가 cpu던 gpu던 무관하게 사용 가능.
# import torch
#
# class DataParallelModel(torch.nn.Model):
#     def __init__(self):
#         super(self, DataParallelModel).__init__()
#         self.block1 = torch.nn.Linear(10, 20)
#
#         # warp block2 in DataParallel
#         self.block2 = torch.nn.linear(20, 20)
#         self.block2 = torch.nn.DataParallel(self.block2)
#
#         self.block3 = torch.nn.linear(20, 20)
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return x


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

