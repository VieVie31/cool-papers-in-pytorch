"""
Highway Network on MNIST with PyTorch
https://arxiv.org/pdf/1505.00387.pdf

The goal is only to optimize very deep network (50, 100 layers)
(not studying the generalization properties)
with gated skip-connections because we saw that very deep plain
network are super hard to optimise and often do not even converge
on training data....
Take a look to the logs com' at the end for comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch as T
import torch.nn as nn

from torch.nn.modules import *

from tqdm import tqdm, trange
from torchvision import datasets, transforms


T.set_default_tensor_type('torch.FloatTensor')

batch_size = 32
nb_epochs  = 1
nb_digits  = 10

train_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 
test_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 

class GatedLayer(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(), deactivate_gates=False, bias_init=-2.):
        super(GatedLayer, self).__init__()
        self.deactivate_gates = deactivate_gates
        if not self.deactivate_gates:
            self.gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
            #bias initialization
            self.gate[0].bias = nn.Parameter(T.autograd.Variable((abs(self.gate[0].bias) * 0 + 1) * bias_init)) #random or fixed...
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            activation
        )

    def forward(self, x):
        if self.deactivate_gates:
            return self.layer(x)
        T = self.gate(x)
        H = self.layer(x)
        return T * H + (1 - T) * x

class CHighway(nn.Module):
    def __init__(self, depth, hidden_size=20, deactivate_gates=False, bias_init=-2., activation=nn.ReLU()):
        super(CHighway, self).__init__()
        tmp = [nn.Linear(784, hidden_size)] +\
            [
                GatedLayer(hidden_size, deactivate_gates=deactivate_gates, bias_init=bias_init, activation=activation) 
                for i in range(depth)
            ] +\
            [nn.Linear(hidden_size, 10), nn.Softmax()]
        self.network = nn.Sequential(*tmp)

    def forward(self, x):
        return self.network(x)


def train(celias_model, loss_function, optimizer, repeat=1):
    batchs_history = []
    for i in range(repeat):
        for x, y in tqdm(train_loader):
            if x.shape[0] != batch_size:
                continue
            optimizer.zero_grad()
            yhat = celias_model(x.view([batch_size, 784]))
            loss = loss_function(yhat, y) 
            loss.backward()
            optimizer.step()
            batchs_history.append(loss.item())
    return batchs_history

def accuracy(model):
    accuracy = []
    for x, y in tqdm(test_loader):
        if x.shape[0] != batch_size:
            continue
        yhat = model(x.view([batch_size, 784]))
        accuracy.append((yhat.argmax(1) == y).float().mean().item())
    return np.mean(accuracy)


def avg_filtering(signal, window_size):
    signal = np.array(signal)
    return [signal[i:i + window_size].mean() for i in range(len(signal) - window_size)]

loss_function = CrossEntropyLoss()
for depth in [1, 3, 5, 10, 20, 50, 100]:
    tqdm.write("depth: {}".format(depth), end=' ')

    #training highway
    celias_model = CHighway(depth, hidden_size=20, bias_init=-4., activation=nn.Tanh())
    optimizer = torch.optim.Adam(celias_model.parameters())#, lr=0.001, momentum=.8)

    highway_loss = train(celias_model, loss_function, optimizer, repeat=1)#2 if depth >= 10 else 1)

    tqdm.write("h {}".format(accuracy(celias_model)), end=' ')

    #training classic
    celias_model = CHighway(depth, hidden_size=71, deactivate_gates=True)
    optimizer = torch.optim.Adam(celias_model.parameters())

    classic_loss = train(celias_model, loss_function, optimizer, repeat=1)#2 if depth >= 10 else 1)

    tqdm.write("c {}".format(accuracy(celias_model)), end=' ')

    plt.title("Training Loss (noise attenuation with avg filter) - depth = {}".format(depth))
    plt.plot(avg_filtering(highway_loss, 10), label='highway')
    plt.plot(avg_filtering(classic_loss, 10), label='classic')
    plt.legend()
    #plt.show()
    plt.savefig("_depth_{}.png".format(depth))
    plt.cla() #reset figure

    tqdm.write(".")

## logs
# highway vs non highway
#depth: 1 h 0.9344951923076923 c 0.9270833333333334 .
#depth: 3 h 0.9386017628205128 c 0.9219751602564102 .
#depth: 5 h 0.9303886217948718 c 0.8683894230769231 .
#depth: 10 h 0.9372996794871795 c 0.5400641025641025 .
#depth: 20 h 0.9332932692307693 c 0.33673878205128205 .
#depth: 50 h 0.9367988782051282 c 0.11328125 .
#depth: 100 h 0.9242788461538461 c 0.11358173076923077 .
#==> the gated skip connection help to ignore useless transfrmation layers...

