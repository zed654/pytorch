import torch

N, D_in, H1, H2, D_out = 1, 10, 8, 8, 5

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

class CHP_Net(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(CHP_Net, self).__init__()             # 얜 함수로 불러온 것 이기 때문에 __init__(self)가 아님.
        self.Linear1 = torch.nn.Linear(D_in, H1)
        self.Linear2 = torch.nn.Linear(H1, H2)
        self.Linear3 = torch.nn.Linear(H2, D_out)
        self.loss_fn = torch.nn.MSELoss(size_average=False, reduce=True)

        self.model = torch.nn.Sequential(
            self.Linear1,
            torch.nn.ReLU(),
            self.Linear2,
            torch.nn.ReLU(),
            self.Linear3
        )
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        self.y_pred = self.model(x)

    def backward(self, y, iter):
        self.loss = self.loss_fn(self.y_pred, y)
        print(iter, self.loss.item())
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


net = CHP_Net(D_in, H1, H2, D_out)

for t in range(500):
    net.forward(x)
    net.backward(y, t)