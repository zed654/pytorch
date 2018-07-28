import torch

N, D_in, H, D_out = 4, 4, 100, 2

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

class CHP_model(torch.nn.Module):           # torch.nn.Module은 클래스이다.
    def __init__(self, D_in, H, D_out):     # 생성자
        super(CHP_model, self).__init__()   # torch.nn.Module의 생성자.
        self.Linear1 = torch.nn.Linear(D_in, H)     # self.Linear1은 torch.nn.Module클래스에 변수 Linear1을 추가하는 것 이다.
        self.Linear2 = torch.nn.Linear(H, D_out)    # self.Linear2은 torch.nn.Module클래스에 변수 Linear2을 추가하는 것 이다.

    def forward(self, x):
        """
        forward는 y_pred를 계산하기 위해서 사용된다.

        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        model = torch.nn.Sequential(            # y_pred를 계산하기위한 방법들을 이곳에서 똑같이 구현할 수 있다.
            self.Linear1,                       # 추가된 점은 클래스화되어 있기 때문에 클래스의 변수를 그대로 사용할 수 있다.
                                                # __init__에서 self.변수로 변수를 생성하였기 때문에 이처럼 사용가능하다.
            #torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            self.Linear2
            #torch.nn.Linear(H, D_out)
        )
        y_pred = model(x)                       # torch.nn.Sequential() 문법을 그대로 사용할 수 있다.

        return y_pred

model = CHP_model(D_in, H, D_out)               # 해당 모델에 대한 클래스의 객체를 초기변수를 입력받으며 생성한다.

loss_fn = torch.nn.MSELoss(size_average=False, reduce=True)         # 왠지 얘도 클래스안에 넣을 수 있을것같다.

learning_rate = 1e-4                                                # 왠지 얘도 클래스안에 넣을 수 있을것같다.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 왠지 얘도 클래스안에 넣을 수 있을것같다.

for t in range(500):
    y_pred = model.forward(x)   # y_pred 계산을 위해서 model객체(CHP_model()에서 받은)의 forward를 가져온다. 다만 입력인 x도 함께 넣어준다.
                                # 클래스가 없는 상태에서 model = torch.nn.Sequential()를 이용해 만들었다면, y_pred = model(x)가 올바른 문법이다.
    loss = loss_fn(y_pred, y)

    print(t, loss.item())

    optimizer.zero_grad()       # 얘는 .backward() 전에 명령해줘야함.
    loss.backward()

    optimizer.step()
    #with torch.no_grad():
    #    for param in model.parameters():
    #        param -= param.grad * learning_rate