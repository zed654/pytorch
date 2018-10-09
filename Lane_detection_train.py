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

# 이미지에 점 그리는 함수. 20x20의 사이즈로 그려진다.
def putpixel_area(img_, x_, y_):
    for i in range(20):
        for j in range(20):
            img_.putpixel((x_ + j, y_ + i), (128, 128, 128))

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
        patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

        # patch size가 (15, 15)로 반듯하게 잘리지 않으면, 강제로 (15, 15)로 리사이즈해준다.
        if patch_tmp.size != (15, 15):
            patch_tmp = patch_tmp.resize((15, 15))

        img_PIL_patch.append(patch_tmp)


device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")

batch_size = 2
channel = 3
width = 15#256#256     #1280
height = 15#480#256    #960    # 1920 / 4 = 480
outputs_class = 2  #480

inputs = ToTensor()(img_PIL_patch[0])
inputs = inputs.view(1,3,15,15)

outputs = torch.Tensor(batch_size, outputs_class)
outputs[:, 0] = 1.
for i in range(outputs_class-1):
    outputs[:, i+1] = 0.

print(outputs)

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
            torch.nn.Linear(4096, outputs_class)
            # torch.nn.BatchNorm1d(outputs_class),        # softmax단인데 필요함? --> 없는게 학습 더 잘 될 듯.
            # torch.nn.Softmax()
        )

    def forward(self, input_):
        feature_out_ = self.feature(input_)
        fc_inputs_ = feature_out_.view(feature_out_.size(0), 256 * 8 * 8)   # Conv2 -> Fully Connected
                                                                            # feature_out_.size(0) 는 batch_size
        hypothesis_ = self.classifier(fc_inputs_)
        # print(hypothesis_.size())
        return hypothesis_



    # def bacward(self):



# 아래는 Training 단계
net = KJY_MODEL().to(device)

learning_rate = 1.5e-3#1.5e-1
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fun = torch.nn.MSELoss()

# loss graph 출력용
loss_graph_x = []
loss_graph_y = []


patch_size_error_count = 0
for t in range(1000):
    # Input patch image load and reform to mini batch format
    l = int(len(img_PIL_patch) / batch_size)
    if batch_size > len(img_PIL_patch) :
        print('batch_size is larger than img patchs')
    n = (t % l)+1
    m = batch_size * n
    inputs = ToTensor()(img_PIL_patch[m-1])
    inputs = inputs.view(1, 3, 15, 15)
    for i in range(batch_size-1):
        # print(img_PIL_patch[m-i-2])
        inputs_tmp = ToTensor()(img_PIL_patch[m-i-2])
        # print(inputs_tmp.size())
        inputs_tmp = inputs_tmp.view(1, 3, 15, 15)
        inputs = torch.cat((inputs, inputs_tmp), 0)

    # 랜덤 데이터 입력
    # inputs = torch.randint(0, 255, (batch_size, channel, width, height))  # 0~255범위의 값을 입력한다.

    start_clock = time.time()

    # 아래 과정에서 inference
    hypothesis = net.forward(inputs).to(device)

    # 아래 과정에서 loss 계산
    loss = loss_fun(hypothesis, outputs).to(device)

    finish_clock = time.time()
    fps = 1. / (finish_clock - start_clock)
    print(t, loss.item(), 'fps = %f' % (fps))


    # loss graph 출력
    loss_graph_x.append(t)
    loss_graph_y.append(loss.item())
    plt.plot(loss_graph_x, loss_graph_y)


    # 아래 과정에서 back-propagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (t == 999) | (loss.item() < 1.5e-3):
    # if(loss.item() < 1.5e-3):
        torch.save(net.state_dict(), 'save_Lane_net.pt')
        # break

plt.show()