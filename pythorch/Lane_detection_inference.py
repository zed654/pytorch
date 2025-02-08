import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import time
import os
import numpy as np

from LD_Model import KJY_MODEL

# import matplotlib.pylab as plt

for x in range(1000):
    # img_path = '/Users/CHP/Lane_detector_pytorch/sample/'
    # img_name = '00000006.jpg'
    img_path = '/Users/CHP/Lane_detector_pytorch/gt_sw/'
    img_name_tmp = 3777
    img_name_tmp2 = img_name_tmp + x
    img_name = str(img_name_tmp2).zfill(8) + '.jpg'
    # img_name = '00007777.jpg'
    # img_name = '00003777.jpg'

    coordinates = []
    for i in range(132):
        for j in range(20):
            coordinates.append([i * 7 + 21, 302 + (j * 10 + 180) / 2])


    def putpixel_area(img_, x_, y_):
        for i in range(3):
            for j in range(3):
                img_.putpixel((int(x_ + j), int(y_ + i)), (254, 0, 0))


    img_PIL = []
    img_PIL_resize = []
    img_PIL_patch = []
    patch_tmp = []
    img_PIL = Image.open(img_path + img_name)
    x = []
    y = []
    # img_PIL.show()
    w, h = img_PIL.size
    img_PIL_resize = img_PIL.resize((int(w / 2), int(h / 2)))
    for j in range(len(coordinates)):
        x.append(int(coordinates[j][0]))
        y.append(int(coordinates[j][1]))
        x1 = int(x[j] - 15 / 2)
        y1 = int(y[j] - 15 / 2)
        x2 = int(x[j] + 15 / 2)
        y2 = int(y[j] + 15 / 2)
        patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

        if patch_tmp.size != (15, 15):
            patch_tmp = patch_tmp.resize((15, 15))

        #putpixel_area(img_PIL_resize, x[j], y[j])
        img_PIL_patch.append(patch_tmp)
    #img_PIL_resize.show()

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
    width = 15
    height = 15
    outputs_class = 2  # 480

    # cuda가 있으면 GPU연산을 하고, 없으면 CPU연산
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 아래는 Training 단계
    net = KJY_MODEL().to(device)

    # net.load_state_dict(torch.load('save_Lane_net_20181007.pt'))
    # net.load_state_dict(torch.load('save_Lane_net.pt'))
    # net.load_state_dict(torch.load('save_Lane_net_new.pt'))
    net.load_state_dict(torch.load('save_Lane_net_new_1024.pt'))

    loss_fun = torch.nn.MSELoss()

    # loss graph 출력용
    loss_graph_x = []
    loss_graph_y = []

    inputs = []
    outputs = []
    # for t in range(len(img_PIL_patch)):
    for t in range(len(coordinates)):
    # for t in range(100):
        if (batch_size % 2 != 0):
            print('batch size % 2 != 0')

        # batch_size_half = int(batch_size/2)
        # outputs = torch.Tensor(batch_size, outputs_class)

        # Input patch image load and reform to mini batch format
        l = int(len(img_PIL_patch) / batch_size)
        if batch_size > len(img_PIL_patch):
            print('batch_size is larger than img patchs')
        n = (t % l) + 1
        m = batch_size * n

        inputs = ToTensor()(img_PIL_patch[m - 1])
        inputs = inputs.view(1, 3, 15, 15)
        for i in range(batch_size - 1):
            print(m - 1, m - i - 2)
            # print(x[m - 1], y[m - 1], x[m - i - 2], y[m - i - 2])
            # print(img_PIL_patch[m-i-2])
            inputs_tmp = ToTensor()(img_PIL_patch[m - i - 2])
            # print(inputs_tmp.size())
            inputs_tmp = inputs_tmp.view(1, 3, 15, 15)
            inputs = torch.cat((inputs, inputs_tmp), 0)

        start_clock = time.time()

        # 아래 과정에서 inference
        hypothesis = net.forward(inputs)  # .to(device)

        finish_clock = time.time()
        fps = 1. / (finish_clock - start_clock)
        print(hypothesis)
        print(t, batch_size, len(img_PIL_patch), 'fps = %f' % (fps))

        if (hypothesis[0][0] > 0.99):
            putpixel_area(img_PIL_resize, x[m - 1], y[m - 1])

        if (hypothesis[1][0] > 0.99):
            putpixel_area(img_PIL_resize, x[m - 2], y[m - 2])

    img_PIL_resize.show()
    img_PIL_resize.save('/Users/CHP/Lane_detector_pytorch/saved_img/'+img_name)