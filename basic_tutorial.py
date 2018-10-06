
# torch.Tensor(행, 열) # 초기화되지 않은 Tensor
# torch.rand(3, 3)  # 0~1의 랜덤한 값이 ddd배정
# torch.randn(3, 3) # 평균이 0이고 분산이 1인 normal distribution random value

# Numpy를 Tensor로
# a = np.array([1, 2, 3, 4]) 를 tensor로 바꾸려면
# b = torch.Tensor(a)
# b = a.Tensor() 도 될듯 -> 안됌안됌안됌안됌

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


# 시간 카운트하기
# from datetime import datetime
# start = datetime.now()

# print(datetime.now()-start)

# 그 외
# Tensor a의 평균
#   a.mean()
# Tensor a의 원소합
#   a.sum()






#                                       #
#   Pytorch에서 이미지 read/write/modify   #
#                                       #

# Pytorch에서 이미지를 사용하기 위해서는 PIL이미지를 사용하는것이 좋다.
# 이유는 Python에서 기본적으로 지원하는 함수이며, OpenCV에 있는 기본 기능 모두 구현이 가능하다.

# Python에서 이미지 처리는 PIL 사용
#   from PIL import Image

# PIL에서 이미지 read
#   img_PIL = Image.open('/Users/.../~.jpg')

# PIL에서 이미지 show
#   img_PIL.show()

# PIL Image To Tensor
#   from torchvision.transforms import ToTensor
#   to_Tensor = ToTensor()
#   img_Tensor = to_Tensor(img_PIL) or img_Tensor = ToTensor()(img_PIL)


# Tensor To PIL Image
#   from torchvision.transforms import ToPILImage
#   to_PIL = ToPILImage()
#   img_PIL = to_PIL(img_Tensor)      or      img_PIL = ToPILImage()(img_Tensor)


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

