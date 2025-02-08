# -*- coding: utf-8 -*-
########################################################################
# A full and working MNIST example is located here
# https://github.com/pytorch/examples/tree/master/mnist
#
# Example 2: Recurrent Net
# ------------------------
#
# Next, let’s look at building recurrent nets with PyTorch.
#
# Since the state of the network is held in the graph and not in the
# layers, you can simply create an nn.Linear and reuse it over and over
# again for the recurrence.
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output


rnn = RNN(50, 20, 10)

########################################################################
#
# A more complete Language Modeling example using LSTMs and Penn Tree-bank
# is located
# `here <https://github.com/pytorch/examples/tree/master/word\_language\_model>`_
#
# PyTorch by default has seamless CuDNN integration for ConvNets and
# Recurrent Nets

loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

loss = 0
for t in range(TIMESTEPS):
    # yes! you can reuse the same network several times,
    # sum up the losses, and call backward!
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
    print(t, loss.item())
loss.backward()