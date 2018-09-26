import torch
import time

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

mini_batch = 3
channel = 3
width = 256#256     #1280
height = 960#256    #960
outputs_class = 34 * 960

inputs = torch.randn(mini_batch, channel, width, height)
outputs = torch.randn(mini_batch, outputs_class)        # GT image
# outputs = torch.Tensor([[0.5, 0.5],[0.8, 0.8],[1., 1.]])    # mini_batch = 3, output class = 2 임

# outputs = torch.Tensor([[700, 300], [201, 203], [220, 110]])
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class KJY_MODEL(torch.nn.Module):
    def __init__(self):
        super(KJY_MODEL, self).__init__()

        self.BN1 = torch.nn.BatchNorm2d(3)

        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # m = nn.BatchNorm2d(100, affine=False)

        self.feature = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 6, kernel_size=2, stride=1, padding=1),            # torch.nn.BatchNorm2d,
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Conv2d(6, 9, kernel_size=2, stride=1, padding=1),            # torch.nn.BatchNorm2d,
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Conv2d(9, 12, kernel_size=2, stride=1, padding=1),            # torch.nn.BatchNorm2d,
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(12 * 34 * 122),     # 34 * 34
            # torch.nn.Dropout(),
            torch.nn.Linear(12 * 34 * 122, 4096),    # 4096
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(),
            # torch.nn.Linear(4096, 4096),
            # torch.nn.BatchNorm1d(4096),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, outputs_class),
            torch.nn.BatchNorm1d(outputs_class)
        )

    def forward(self, input_):
        feature_out_ = self.feature(input_)
        fc_inputs_ = feature_out_.view(feature_out_.size(0), 12 * 34 * 122)
        hypothesis_ = self.classifier(fc_inputs_)
        print(hypothesis_.size())
        return hypothesis_



    # def bacward(self):

net = KJY_MODEL()

learning_rate = 1.5e-2
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
