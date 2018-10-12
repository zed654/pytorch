import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor


class KJY_MODEL(torch.nn.Module):
    def __init__(self):
        super(KJY_MODEL, self).__init__()

        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # m = nn.BatchNorm2d(100, affine=False)

        self.feature1 = torch.nn.Sequential(
            # 입력 채널 개수
            torch.nn.BatchNorm2d(3),        # BN은 배치사이즈가 1보다 커야함.

            # CNN Layer 1
            torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=True)
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
            #
            # # CNN Layer 1
            # torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            # torch.nn.BatchNorm2d(6),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
        )

        self.feature2 = torch.nn.Sequential(
            # 입력 채널 개수
            # torch.nn.BatchNorm2d(3),        # BN은 배치사이즈가 1보다 커야함.
            #
            # # CNN Layer 1
            # torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            # torch.nn.BatchNorm2d(6),
            # torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8

            # CNN Layer 1
            torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
        )

        self.classifier = torch.nn.Sequential(
            # CNN단의 최종 데이터에 대해 BN 실시
            #       Conv2의 출력을 Linear로 변환한 값을 넣음. channel x width x height
            torch.nn.BatchNorm1d(6*8*8),

            # FC Linear calc 1단계
            torch.nn.Linear(6*8*8, 192),
            torch.nn.BatchNorm1d(192),
            torch.nn.ReLU(inplace=True),

            # FC Linear calc 2단계
            #       1000 -> 1000의 이유는 차원 증가를 위해서. (MLP쓰는 이유에서 착안)
            # torch.nn.Linear(1024, 1024),
            # torch.nn.BatchNorm1d(1024),
            # torch.nn.ReLU(inplace=True),

            # 최종 Classification단계
            #       최종 1000개의 Feature에서 최종 클래스인 2 (차선이다, 아니다)로 구분
            torch.nn.Linear(192, 2),
            torch.nn.Sigmoid()
            # torch.nn.BatchNorm1d(outputs_class),        # softmax단인데 필요함? --> 없는게 학습 더 잘 될 듯.
            # torch.nn.Softmax()
        )

    def forward(self, input_):
        feature_out_1 = self.feature1(input_)
        feature_out_2 = self.feature2(feature_out_1)
        #fc_inputs_1 = feature_out_1.view(feature_out_1.size(0), 6 * 15 * 15)
        fc_inputs_2 = feature_out_2.view(feature_out_2.size(0), 6 * 8 * 8)#256 * 8 * 8)   # Conv2 -> Fully Connected
                                                                            # feature_out_.size(0) 는 batch_size
        # # feature img 출력하기
        # feature_1_view = torch.chunk(feature_out_1, feature_out_1.size(0), 0)
        # feature_2_view = torch.chunk(feature_out_2, feature_out_2.size(0), 0)
        # for i in range(len(feature_1_view)):
        #     # len(feature_1_view)는 배치사이즈 개수만큼 있음.
        #     feature_1_img = feature_1_view[i].view(feature_out_1.size(1), feature_out_1.size(2), feature_out_1.size(3))
        #     feature_2_img = feature_2_view[i].view(feature_out_2.size(1), feature_out_2.size(2), feature_out_2.size(3))
        #     # Tensor를 PIL로 바꾸어 비주얼라이즈
        #     feature_1_PIL = ToPILImage()(feature_1_img)
        #     feature_2_PIL = ToPILImage()(feature_2_img)
        #     # feature show
        #     feature_1_PIL = feature_1_PIL.resize((60, 60))
        #     feature_2_PIL = feature_2_PIL.resize((30, 30))
        #     feature_1_PIL.show()
        #     feature_2_PIL.show()

        #fc_inputs_ = torch.cat((fc_inputs_1, fc_inputs_2), 1)
        fc_inputs_ = fc_inputs_2
        hypothesis_ = self.classifier(fc_inputs_)
        # print(hypothesis_.size())
        return hypothesis_

    # def bacward(self):


# class KJY_MODEL(torch.nn.Module):
#     def __init__(self):
#         super(KJY_MODEL, self).__init__()
#
#         # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         # m = nn.BatchNorm2d(100, affine=False)
#
#         self.feature1 = torch.nn.Sequential(
#             # 입력 채널 개수
#             torch.nn.BatchNorm2d(3),        # BN은 배치사이즈가 1보다 커야함.
#
#             # CNN Layer 1
#             torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
#             torch.nn.BatchNorm2d(6),
#             torch.nn.ReLU(inplace=True)
#             # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
#             #
#             # # CNN Layer 1
#             # torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
#             # torch.nn.BatchNorm2d(6),
#             # torch.nn.ReLU(inplace=True),
#             # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
#         )
#
#         self.feature2 = torch.nn.Sequential(
#             # 입력 채널 개수
#             # torch.nn.BatchNorm2d(3),        # BN은 배치사이즈가 1보다 커야함.
#             #
#             # # CNN Layer 1
#             # torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
#             # torch.nn.BatchNorm2d(6),
#             # torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
#
#             # CNN Layer 1
#             torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # output = 9 x 15 x 15
#             torch.nn.BatchNorm2d(6),
#             torch.nn.ReLU(inplace=True),
#             # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output = 9 x 8 x 8
#         )
#
#         self.classifier = torch.nn.Sequential(
#             # CNN단의 최종 데이터에 대해 BN 실시
#             #       Conv2의 출력을 Linear로 변환한 값을 넣음. channel x width x height
#             torch.nn.BatchNorm1d(6*8*8+6*15*15),
#
#             # FC Linear calc 1단계
#             torch.nn.Linear(6*8*8+6*15*15, 192),
#             torch.nn.BatchNorm1d(192),
#             torch.nn.ReLU(inplace=True),
#
#             # FC Linear calc 2단계
#             #       1000 -> 1000의 이유는 차원 증가를 위해서. (MLP쓰는 이유에서 착안)
#             # torch.nn.Linear(1024, 1024),
#             # torch.nn.BatchNorm1d(1024),
#             # torch.nn.ReLU(inplace=True),
#
#             # 최종 Classification단계
#             #       최종 1000개의 Feature에서 최종 클래스인 2 (차선이다, 아니다)로 구분
#             torch.nn.Linear(192, 2),
#             torch.nn.Sigmoid()
#             # torch.nn.BatchNorm1d(outputs_class),        # softmax단인데 필요함? --> 없는게 학습 더 잘 될 듯.
#             # torch.nn.Softmax()
#         )
#
#     def forward(self, input_):
#         feature_out_1 = self.feature1(input_)
#         feature_out_2 = self.feature2(feature_out_1)
#         fc_inputs_1 = feature_out_1.view(feature_out_1.size(0), 6 * 15 * 15)
#         fc_inputs_2 = feature_out_2.view(feature_out_2.size(0), 6 * 8 * 8)#256 * 8 * 8)   # Conv2 -> Fully Connected
#                                                                             # feature_out_.size(0) 는 batch_size
#         # # feature img 출력하기
#         # feature_1_view = torch.chunk(feature_out_1, feature_out_1.size(0), 0)
#         # feature_2_view = torch.chunk(feature_out_2, feature_out_2.size(0), 0)
#         # for i in range(len(feature_1_view)):
#         #     # len(feature_1_view)는 배치사이즈 개수만큼 있음.
#         #     feature_1_img = feature_1_view[i].view(feature_out_1.size(1), feature_out_1.size(2), feature_out_1.size(3))
#         #     feature_2_img = feature_2_view[i].view(feature_out_2.size(1), feature_out_2.size(2), feature_out_2.size(3))
#         #     # Tensor를 PIL로 바꾸어 비주얼라이즈
#         #     feature_1_PIL = ToPILImage()(feature_1_img)
#         #     feature_2_PIL = ToPILImage()(feature_2_img)
#         #     # feature show
#         #     feature_1_PIL = feature_1_PIL.resize((60, 60))
#         #     feature_2_PIL = feature_2_PIL.resize((30, 30))
#         #     feature_1_PIL.show()
#         #     feature_2_PIL.show()
#
#         fc_inputs_ = torch.cat((fc_inputs_1, fc_inputs_2), 1)
#         hypothesis_ = self.classifier(fc_inputs_)
#         # print(hypothesis_.size())
#         return hypothesis_
#
#     # def bacward(self):