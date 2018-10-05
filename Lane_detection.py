import torch
# import numpy as np
import time

# import torchvision

from torchvision.transforms import ToPILImage
from IPython.display import Image



import numpy as np
import matplotlib.pylab as plt

# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()


def show(img):
         npimg = img.numpy()
         plt.imshow(np.transpose(npimg, (2,0,1)), interpolation='nearest')
         plt.show()

to_img = ToPILImage()


# numpy input img를 pytorch img로
input_img = plt.imread('/Users/CHP/Lane_detector_pytorch/sample/00000098.jpg')
input_img_reshape = np.transpose(input_img, (2, 1, 0))      # 2, 1, 0 은 채널, 가로, 세로 순서를 바꿔주는 것
input_img_torch = torch.from_numpy(input_img_reshape)
# or
path = '/Users/CHP/Lane_detector_pytorch/sample/00000098.jpg'
def read_img(path):
    input_img_tmp = plt.imread(path)
    input_img_reshape_tmp = np.transpose(input_img_tmp, (2, 1, 0))  # 2, 1, 0 은 채널, 가로, 세로 순서를 바꿔주는 것
    input_img_torch_tmp = torch.from_numpy(input_img_reshape_tmp)
    return input_img_torch_tmp


    

# pytorch img를 numpy img로
input_img2 = torch.randint(128, 129, (3, 1920, 1208))     # 0 ~ 254
input_img2_numpy = input_img2.numpy()
input_img2_numpy_reshape = np.transpose(input_img2_numpy,(2, 1, 0))
input_img2_numpy_reshape_dtype = input_img2_numpy_reshape.astype('uint8')
plt.imshow(input_img2_numpy_reshape_dtype)
# or
def show_img(img):
    img_tmp = img.numpy()
    img_reshape_tmp = np.transpose(img_tmp, (2, 1, 0))
    img_reshape_dtype_tmp = img_reshape_tmp.astype('uint8')
    plt.imshow(img_reshape_dtype_tmp)




# input_img_torch = torch.from_numpy(np.transpose(input_img, (2, 0, 1)), interpolation='nearest')

plt.imshow()
# display tensor
a = torch.randint(128, 255, (3, 64, 64)).normal_()
show(a)

    # torch.Tensor(3, 64, 64).normal_()
to_img(a)

#display imagefile
Image('/Users/CHP/Lane_detector_pytorch/sample/00000098.jpg')

# img = Image.open('/home/CHP/Lane_detector_pytorch/sample/00000098.jpg').convert('RGB')
# to_pil = torchvision.transforms.ToPILImage()
# img = to_pil(img)

"입력 이미지는 4:3기준 512*384(4:3 * 128)로 리사이즈하여 사용"
"GT는 좌표지점 기준으로 15 * 15 예상 (홀수여야함)"

"region proposal 1은 세로 Grid 기준, 7pixel(=15/2)씩 이동. 가로가 512면 한 줄에 약 73개"
"region proposal 1 ... 세로가 20줄이면 1,460개의 Region이 제안된다."

"region proposal 2 세로 Grid기준, 최하단의 grid는 7pixel(=15/2)씩 이동하고 Lane Search Flag에 1을 줌"
"region proposal 2 ... 첫 줄의 Lane Search Flag가 1인 부분의 상단부분만 (동일 x기준, k픽셀의 +,- 범위에서. k는 아직 미정)"
"region proposal 2 ... 얘는 대략 [<차선 개수> * 3] 개의 Region이 제안될 것으로 예측 (약 12개씩)"
"region proposal 2 ... 고로 20줄이면, 첫 줄에 73개. 두 번째 줄 부터 12개씩 1 * 73 + 19 * 12 = 301개의 region이 제안됨 "
"region proposal 2 ... 장점으로는 속도가 빠르나, 단점으로는 첫 줄에서 찾지 못하면 뒤로도 못찾음"
"region proposal 2 ... 하지만 해볼만한 이유로는 차선 패턴이 단순하여 정확도, 재현율이 거의 100%일 것 같음"

" Train method 1"
"CNN을 통한 Classifier 학습은 region proposal된 부분만 시켜도 될 듯 함."
"False에 대한 15 * 15 이미지는 필요 없음 (학습 방식을 보면 당연한거임)"

" Train method 2"
"이미지 전체에 GT영역 넣어서 연산"

" Train method 3"
"GT영역만 잡아서 CNN없이 FC로 --> 근대 얜 별로일듯. 2d 특징이 죽을 것 같음."

mini_batch = 30
channel = 3
width = 15#256#256     #1280
height = 15#480#256    #960    # 1920 / 4 = 480
outputs_class = 2  #480


# inputs = torch.randn(mini_batch, channel, width, height)
inputs = torch.randint(0, 255, (mini_batch, channel, width, height))    # 0~255범위의 값을 입력한다.
print(inputs)

# outputs = torch.randn(mini_batch, outputs_class)        # GT image
# outputs = torch.Tensor([[1., 0.], [1., 0.], [1., 0.]])
outputs = torch.randint(0, 2, (mini_batch, outputs_class))
for t in range(mini_batch):
    outputs[t][0] = 1.
    outputs[t][1] = 0.

# outputs = torch.Tensor([[0.5, 0.5],[0.8, 0.8],[1., 1.]])    # mini_batch = 3, output class = 2 임

# outputs = torch.Tensor([[700, 300], [201, 203], [220, 110]])
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class KJY_MODEL(torch.nn.Module):
    def __init__(self):
        super(KJY_MODEL, self).__init__()

        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # m = nn.BatchNorm2d(100, affine=False)

        self.feature = torch.nn.Sequential(
            # 입력 채널 개수
            torch.nn.BatchNorm2d(3),        # BN은 배치사이즈가 1보다 커야함.

            # CNN Layer 1
            torch.nn.Conv2d(3, 81, kernel_size=3, stride=1, padding=1),     # output = 9 x 15 x 15
            torch.nn.BatchNorm2d(81),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),         # output = 9 x 8 x 8

            # CNN Layer 2
            torch.nn.Conv2d(81, 274, kernel_size=3, stride=1, padding=1),   # output = 27 x 8 x 8
            torch.nn.BatchNorm2d(274),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)          # 274 * 5 * 5 = 6,850
        )

        self.classifier = torch.nn.Sequential(
            # CNN단의 최종 데이터에 대해 BN 실시
            #       Conv2의 출력을 Linear로 변환한 값을 넣음. channel x width x height
            torch.nn.BatchNorm1d(274*5*5),

            # FC Linear calc 1단계
            torch.nn.Linear(274*5*5, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(inplace=True),

            # FC Linear calc 2단계
            #       1000 -> 1000의 이유는 차원 증가를 위해서. (MLP쓰는 이유에서 착안)
            torch.nn.Linear(1000, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(inplace=True),

            # 최종 Classification단계
            #       최종 1000개의 Feature에서 최종 클래스인 2 (차선이다, 아니다)로 구분
            torch.nn.Linear(1000, 2),
            torch.nn.BatchNorm1d(2),        # softmax단인데 필요함? --> 없는게 학습 더 잘 될 듯.
            torch.nn.Softmax()
        )

    def forward(self, input_):
        feature_out_ = self.feature(input_)
        fc_inputs_ = feature_out_.view(feature_out_.size(0), 274 * 5 * 5)   # Conv2 -> Fully Connected
                                                                            # feature_out_.size(0) 는 batch_size
        hypothesis_ = self.classifier(fc_inputs_)
        print(hypothesis_.size())
        return hypothesis_



    # def bacward(self):



# 아래는 Training 단계
net = KJY_MODEL()

learning_rate = 1.5e-1
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fun = torch.nn.MSELoss()

for t in range(500):
    start_clock = time.time()

    hypothesis = net.forward(inputs)

    loss = loss_fun(hypothesis, outputs)

    finish_clock = time.time()
    fps = 1. / (finish_clock - start_clock)
    print(t, loss.item(), 'fps = %f' % (fps))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # if (t % 10) == 0:
    #     torch.save(net, 'save_net.pt')
