import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import time
import os
import numpy as np
import matplotlib.pylab as plt



# GT txt에 img files의 이름이 들어가고, img files보다 gt txt들의 개수가 더 많으므로
# gt txt의 폴더를 읽어 txt파일들의 이름을 변수에 저장한 후
# 조금 수정하여 img files의 이름도 저장한 코드이다.

# 해당 이름을 기준으로 txt를 실행
txt_path = '/Users/CHP/Lane_detector_pytorch/sample/txt/'
img_path = '/Users/CHP/Lane_detector_pytorch/sample/'
# txt_path = '/Users/CHP/Lane_detector_pytorch/gt_sw/txt/'
# img_path = '/Users/CHP/Lane_detector_pytorch/gt_Sw/'

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
    len(gt_data)  # 좌표 개수 + 1 개가 나옴.

    label = gt_data[0]  # L0이 들어감.
    print(label)

    # coordinates에 저장할 좌표변수 생성 및 txt파일로부터 읽어들여서 저장
    coordinates_tmp = []
    for c in range(len(gt_data) - 1):
        coordinates_tmp.append(gt_data[c + 1].split(" "))
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
            img_.putpixel((x_ + j, y_ + i), (128, 128, 128))


# patch save
img_PIL_patch = []
for gt_num in range(len(coordinates)):
    img_PIL = Image.open(img_path + img_name[gt_num])
    w, h = img_PIL.size  # 1920 x 1208
    img_PIL_resize = img_PIL.resize((int(w / 2), int(h / 2)))

    for coord_num in range(len(coordinates[gt_num])):
        print('The number of Txt files = %d / %d %d %s \t\t The number of Patch Image = %d' % (gt_num, len(coordinates), coord_num, txt_name[gt_num], len(img_PIL_patch)))

        # resized된 이미지에서 patch size를 15x15로
        x1 = int((int(coordinates[gt_num][coord_num][0]) - 15)/2)
        y1 = int((int(coordinates[gt_num][coord_num][1]) - 15)/2 + 302)
        x2 = int((int(coordinates[gt_num][coord_num][0]) + 15)/2)
        y2 = int((int(coordinates[gt_num][coord_num][1]) + 15)/2 + 302)

        # patch size가 (15, 15)로 반듯하게 잘리지 않으면, 강제로 (15, 15)로 리사이즈해준다.
        if patch_tmp.size != (15, 15):
            patch_tmp = patch_tmp.resize((15, 15))

        patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

        # resized된 이미지에서 patch size를 30x30로
        # x1 = int(int(coordinates[gt_num][coord_num][0]) / 2 - 15)
        # y1 = int(int(coordinates[gt_num][coord_num][1]) / 2 - 15 + 302)
        # x2 = int(int(coordinates[gt_num][coord_num][0]) / 2 + 15)
        # y2 = int(int(coordinates[gt_num][coord_num][1]) / 2 + 15 + 302)
        # patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

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

# patch 이미지 view
# for i in range(len(img_PIL_patch)):
#     img_PIL_patch[i].show()


# # batch_size에 맞춰서 PIL을 tensor형식으로 바꾸기
# batch_size = 10
# result = ToTensor()(img_PIL_patch[0])
# result = result.view(1,3,15,15)
#
# for i in range(len(img_PIL_patch)):
#     if (i%batch_size == 0) & (i != 0):
#         for j in range(batch_size):
#             print(i, j, i-j-1)
#             if i-j-1 != 0 : # 처음 result에 들어가있어서.
#                 k = ToTensor()(img_PIL_patch[i - j - 1])
#                 k = k.view(1,3,15,15)
#                 result = torch.cat((result, k), 0)


# 10 batch size로 inputs 데이터를 묶은 것.
batch_size = 1
patch_ea = len(img_PIL_patch)
for t in range(100):
    # Input patch image load and reform to mini batch format
    l = int(len(img_PIL_patch) / batch_size)
    n = (t % l)+1
    m = batch_size * n
    inputs = ToTensor()(img_PIL_patch[m-1])
    inputs = inputs.view(1, 3, 15, 15)
    for i in range(batch_size-1):
        inputs_tmp = ToTensor()(img_PIL_patch[m-i-2])
        inputs_tmp = inputs_tmp.view(1, 3, 15, 15)
        inputs = torch.cat((inputs, inputs_tmp), 0)


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

batch_size = 2
channel = 3
width = 15#256#256     #1280
height = 15#480#256    #960    # 1920 / 4 = 480
outputs_class = 2  #480

# cuda가 있으면 GPU연산을 하고, 없으면 CPU연산
device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")

# inputs = torch.randn(mini_batch, channel, width, height)
# inputs = torch.randint(0, 255, (mini_batch, channel, width, height))    # 0~255범위의 값을 입력한다.
# print(inputs)

# outputs = torch.randn(mini_batch, outputs_class)        # GT image
# outputs = torch.Tensor([[1., 0.], [1., 0.], [1., 0.]])




# 아래의 주석이 풀린 outputs과 같은 결과임. 더 간단해서 바꿧음
# outputs = torch.randint(0, 2, (mini_batch, outputs_class))
# for t in range(mini_batch):
#     outputs[t][0] = 1.
#     outputs[t][1] = 0.

outputs = torch.Tensor(batch_size, outputs_class)
outputs[:, 0] = 1.
for i in range(outputs_class-1):
    outputs[:, i+1] = 0.




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

            # # CNN Layer 1
            # torch.nn.Conv2d(3, 81, kernel_size=3, stride=1, padding=1),     # output = 9 x 15 x 15
            # torch.nn.BatchNorm2d(81),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),         # output = 9 x 8 x 8
            #
            # # CNN Layer 2
            # torch.nn.Conv2d(81, 274, kernel_size=3, stride=1, padding=1),   # output = 27 x 8 x 8
            # torch.nn.BatchNorm2d(274),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)          # 274 * 5 * 5 = 6,850

            # CNN Layer 1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8

            # CNN Layer 1
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8

            # CNN Layer 1
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8

            # # CNN Layer 2
            # torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output = 27 x 8 x 8
            # torch.nn.BatchNorm2d(256),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 274 * 5 * 5 = 6,850
        )

        self.classifier = torch.nn.Sequential(
            # CNN단의 최종 데이터에 대해 BN 실시
            #       Conv2의 출력을 Linear로 변환한 값을 넣음. channel x width x height
            torch.nn.BatchNorm1d(256*8*8),

            # FC Linear calc 1단계
            torch.nn.Linear(256*8*8, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),

            # FC Linear calc 2단계
            #       1000 -> 1000의 이유는 차원 증가를 위해서. (MLP쓰는 이유에서 착안)
            torch.nn.Linear(4096, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),

            # 최종 Classification단계
            #       최종 1000개의 Feature에서 최종 클래스인 2 (차선이다, 아니다)로 구분
            torch.nn.Linear(4096, outputs_class),
            # torch.nn.BatchNorm1d(outputs_class),        # softmax단인데 필요함? --> 없는게 학습 더 잘 될 듯.
            torch.nn.Softmax()
        )

    def forward(self, input_):
        feature_out_ = self.feature(input_)
        fc_inputs_ = feature_out_.view(feature_out_.size(0), 256 * 8 * 8)   # Conv2 -> Fully Connected
                                                                            # feature_out_.size(0) 는 batch_size
        hypothesis_ = self.classifier(fc_inputs_)
        print(hypothesis_.size())
        return hypothesis_



    # def bacward(self):



# 아래는 Training 단계
net = KJY_MODEL().to(device)

# net.load_state_dict(torch.load('save_Lane_net_20181007.pt'))
net.load_state_dict(torch.load('save_Lane_net.pt'))


loss_fun = torch.nn.MSELoss()

# while(1):

for t in range(500):

    if batch_size != 1 :
        print('The value of patch_size have to 1')

    # 학습한 데이터 입력
    l = int(len(img_PIL_patch) / batch_size)
    n = (t % l)+1
    m = batch_size * n
    inputs = ToTensor()(img_PIL_patch[m-1])
    inputs = inputs.view(1, 3, 15, 15)
    for i in range(batch_size-1):
        inputs_tmp = ToTensor()(img_PIL_patch[m-i-2])
        inputs_tmp = inputs_tmp.view(1, 3, 15, 15)
        inputs = torch.cat((inputs, inputs_tmp), 0)

    # 랜덤 데이터 입력
    # inputs = torch.randint(0, 255, (batch_size, channel, width, height))  # 0~255범위의 값을 입력한다.

    start_clock = time.time()

    hypothesis = net.forward(inputs).to(device)

    # hypothesis 결과를 True, False로 출력하는 코드
    # for k in range(batch_size):
    #     if hypothesis[k][0] > hypothesis[k][1] :
    #         print(True)
    #     else:
    #         print(False)
    # print('\n')

    # hypothesis 결과를 value로 출력하는 코드
    print(hypothesis)



    # print('%.2f' % hypothesis)
    # # 아래 과정에서 loss 계산, backpropagation
    # loss = loss_fun(hypothesis, outputs).to(device)
    #
    # finish_clock = time.time()
    # fps = 1. / (finish_clock - start_clock)
    # print(loss.item(), 'fps = %f' % (fps))
