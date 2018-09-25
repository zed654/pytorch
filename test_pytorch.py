import torch
mini_batch = 3
channel = 3
width = 256     #1280
height = 256    #960
outputs_class = 2

inputs = torch.randn(mini_batch, channel, width, height)
outputs = torch.randn(mini_batch, outputs_class)
outputs = torch.Tensor([[0.5, 0.5],[0.8, 0.8],[1., 1.]])    # mini_batch = 3, output class = 2 임

# outputs = torch.Tensor([[700, 300], [201, 203], [220, 110]])
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class CHP_MODEL(torch.nn.Module):
    def __init__(self):
        super(CHP_MODEL, self).__init__()

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
            torch.nn.BatchNorm1d(12 * 34 * 34),
            # torch.nn.Dropout(),
            torch.nn.Linear(12 * 34 * 34, 4096),
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
        fc_inputs_ = feature_out_.view(feature_out_.size(0), 12 * 34 * 34)
        hypothesis_ = self.classifier(fc_inputs_)
        print(hypothesis_.size())
        return hypothesis_



    # def bacward(self):

net = CHP_MODEL()

learning_rate = 1.5e-2
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fun = torch.nn.MSELoss()

for t in range(500):
    hypothesis = net.forward(inputs)

    loss = loss_fun(hypothesis, outputs)
    print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (t % 10) == 0:
        torch.save(net, 'save_net.pt')
