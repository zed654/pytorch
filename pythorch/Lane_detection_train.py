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

from LD_Model import KJY_MODEL

# GT txt에 img files의 이름이 들어가고, img files보다 gt txt들의 개수가 더 많으므로
# gt txt의 폴더를 읽어 txt파일들의 이름을 변수에 저장한 후
# 조금 수정하여 img files의 이름도 저장한 코드이다.

# 이미지에 점 그리는 함수. 20x20의 사이즈로 그려진다.
def putpixel_area(img_, x_, y_):
    for i in range(20):
        for j in range(20):
            img_.putpixel((x_ + j, y_ + i), (128, 128, 128))


########################################################################################################
########################################################################################################
##########################      정답 이미지 불러오기 ########################################################
########################################################################################################
########################################################################################################

# 해당 이름을 기준으로 txt를 실행
# txt_path = '/Users/CHP/Lane_detector_pytorch/sample/txt/'
# img_path = '/Users/CHP/Lane_detector_pytorch/sample/'
txt_path = '/Users/CHP/Lane_detector_pytorch/gt_sw/txt/'
img_path = '/Users/CHP/Lane_detector_pytorch/gt_Sw/'

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












########################################################################################################
########################################################################################################
##########################      오답 이미지 불러오기 ########################################################
########################################################################################################
########################################################################################################

# 해당 이름을 기준으로 txt를 실행
txt_path2 = '/Users/CHP/Lane_detector_pytorch/gt_negative/txt/'
img_path2 = '/Users/CHP/Lane_detector_pytorch/gt_negative/'
# txt_path2 = '/Users/CHP/Lane_detector_pytorch/gt_sw/txt/'
# img_path2 = '/Users/CHP/Lane_detector_pytorch/gt_Sw/'

for root, dirs, txt_files in os.walk(txt_path2):
    for t in txt_files:
        full_fname = os.path.join(root, t)
        print(full_fname)

# GT txt에서 txt파일을 찾기 위해 이미지 파일로부터 이름을 가져온다.
#   txt_name과 img_name의 순서는 일치하다. (txt_name[3] = img_name[3])
txt_name2 = []
for i in range(len(txt_files)):
    txt_name2.append(txt_files[i])

img_name2 = []
for i in range(len(txt_files)):
    img_name2.append(txt_files[i][-12:-4] + '.jpg')

# img_name = list(set(img_name))

# 좌표를 coordinates[num][좌표카운팅][좌표의 x점][좌표의 y점] 이 된다.
#   여기서 len(coordinates[num])을 통해 해당 점에서 좌표가 카운팅 되었는지를 확인해보고 넘어가야 오류가 안뜬다.
coordinates2 = []
# gt_txt_file = []
# gt_data_line = []
# gt_data = []
for i in range(len(txt_files)):
    # txt의 path를 통해 파일을 읽음
    gt_txt_file = open(txt_path2 + txt_name2[i], 'rt')

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
    coordinates2.append(coordinates_tmp)

# 요약
#   txt_name[num]
#   img_name[num]
#   coordinates[num][좌표카운팅][좌표의x점][좌표의y점]

# patch save
img_PIL_patch2 = []
for gt_num in range(len(coordinates2)):
    img_PIL = Image.open(img_path2 + img_name2[gt_num])
    w, h = img_PIL.size  # 1920 x 1208
    img_PIL_resize = img_PIL.resize((int(w / 2), int(h / 2)))

    for coord_num in range(len(coordinates2[gt_num])):
        print('The number of Txt files = %d / %d %d %s \t\t The number of Patch Image = %d' % (gt_num, len(coordinates2), coord_num, txt_name2[gt_num], len(img_PIL_patch2)))

        # resized된 이미지에서 patch size를 15x15로
        x1 = int((int(coordinates2[gt_num][coord_num][0]) - 15)/2)
        y1 = int((int(coordinates2[gt_num][coord_num][1]) - 15)/2 + 302)
        x2 = int((int(coordinates2[gt_num][coord_num][0]) + 15)/2)
        y2 = int((int(coordinates2[gt_num][coord_num][1]) + 15)/2 + 302)
        patch_tmp = img_PIL_resize.crop((x1, y1, x2, y2))

        # patch size가 (15, 15)로 반듯하게 잘리지 않으면, 강제로 (15, 15)로 리사이즈해준다.
        if patch_tmp.size != (15, 15):
            patch_tmp = patch_tmp.resize((15, 15))

        img_PIL_patch2.append(patch_tmp)





############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")

batch_size = 100
channel = 3
width = 15#256#256     #1280
height = 15#480#256    #960    # 1920 / 4 = 480
outputs_class = 2  #480

# 아래는 Training 단계
net = KJY_MODEL().to(device)

# net.load_state_dict(torch.load('save_Lane_net_new_1024.pt'))

learning_rate = 1.5e-4#1.5e-1
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fun = torch.nn.MSELoss()

# loss graph 출력용
loss_graph_x = []
loss_graph_y = []

inputs = []
outputs = []
for t in range(100):
# t = 0
# while(1):
#     t = t + 1

    if(batch_size % 2 != 0):
        print('batch size % 2 != 0')

    batch_size_half = int(batch_size/2)
    outputs = torch.Tensor(batch_size, outputs_class)

    # Input patch image load and reform to mini batch format
    l = int(len(img_PIL_patch) / batch_size)
    if batch_size > len(img_PIL_patch) :
        print('batch_size is larger than img patchs')
    n = (t % l)+1
    m = batch_size * n

    inputs = ToTensor()(img_PIL_patch[m-1])
    inputs = inputs.view(1, 3, 15, 15)
    for i in range(batch_size_half-1):
        # print(img_PIL_patch[m-i-2])
        inputs_tmp = ToTensor()(img_PIL_patch[m-i-2])
        # print(inputs_tmp.size())
        inputs_tmp = inputs_tmp.view(1, 3, 15, 15)
        inputs = torch.cat((inputs, inputs_tmp), 0)

    for i in range(batch_size_half):
        outputs[i][0] = 1.
        outputs[i][1] = 0.






    batch_size_half = int(batch_size/2)
    # outputs = torch.Tensor(batch_size, outputs_class)

    # Input patch image load and reform to mini batch format
    l = int(len(img_PIL_patch2) / batch_size)
    if batch_size > len(img_PIL_patch2) :
        print('batch_size is larger than img patchs')
    n = (t % l)+1
    m = batch_size * n

    # inputs = ToTensor()(img_PIL_patch2[m-1])
    inputs_tmp2 = ToTensor()(img_PIL_patch2[m-1])
    inputs_tmp2 = inputs_tmp2.view(1, 3, 15, 15)
    inputs = torch.cat((inputs, inputs_tmp2), 0)
    #inputs = inputs.view(1, 3, 15, 15)
    for i in range(batch_size_half-1):
        # print(img_PIL_patch[m-i-2])
        inputs_tmp = ToTensor()(img_PIL_patch2[m-i-2])
        # print(inputs_tmp.size())
        inputs_tmp = inputs_tmp.view(1, 3, 15, 15)
        inputs = torch.cat((inputs, inputs_tmp), 0)

    for i in range(batch_size_half):
        outputs[i+batch_size_half][0] = 0.
        outputs[i+batch_size_half][1] = 1.




    # inputs_tmp = torch.randint(0, 255, (batch_size_half, channel, width, height)) / 255.
    # inputs = torch.cat((inputs, inputs_tmp), 0)
    #
    # for i in range(batch_size_half):
    #     outputs[batch_size_half + i][0] = 0.
    #     outputs[batch_size_half + i][1] = 1.
    #
    # inputs = inputs.to(device)
    # outputs = outputs.to(device)








    start_clock = time.time()

    # 아래 과정에서 inference
    hypothesis = net.forward(inputs)#.to(device)

    # 아래 과정에서 loss 계산
    loss = loss_fun(hypothesis, outputs)#.to(device)

    finish_clock = time.time()
    fps = 1. / (finish_clock - start_clock)
    print(outputs[0], hypothesis[0])
    print(outputs[batch_size - 1], hypothesis[batch_size - 1])

    print(t, batch_size, len(img_PIL_patch), loss.item(), 'fps = %f' % (fps))


    # # loss graph 출력
    loss_graph_x.append(t)
    loss_graph_y.append(loss.item())
    plt.plot(loss_graph_x, loss_graph_y)


    # 아래 과정에서 back-propagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # if (t == 999) | (loss.item() < 1.5e-3):
    # if(loss.item() < 1.5e-3) & (t % 100 == 0):
    # if(loss.item() < 1.5e-3):
    if(t%10 == 0) & (loss.item() < 1.5e-3):
        torch.save(net.state_dict(), 'save_Lane_net_new_1024.pt')
        print('save loss')
        # break

plt.show()
