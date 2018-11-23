"""
Re-implemeting "Local Binary Convolutional Neural Networks" (CVPR '17)
https://arxiv.org/pdf/1608.06049.pdf
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
nb_epochs  = 30
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

class LBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, percent=.2):
        super(LBConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=False)
        bernoulli_distribution = T.ones((out_channels, in_channels, 3, 3)) * percent
        weights = T.bernoulli(bernoulli_distribution) + T.bernoulli(bernoulli_distribution) * -1
        self.conv.weight = T.nn.Parameter(weights, requires_grad=False)
        self.learnable_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        return self.learnable_conv(self.conv(x))


class SmallLBCNN(nn.Module):
    def __init__(self):
        super(SmallLBCNN, self).__init__()
        self.features = nn.Sequential(
            LBConv2d(1, 15),
            nn.ReLU(),
            nn.MaxPool2d(2),
            LBConv2d(15, 48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.clf = nn.Sequential(
            nn.Linear(1200, 10)
        )
    
    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        return self.clf(out)



model = SmallLBCNN()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
loss_function = nn.CrossEntropyLoss()


model.train()
for x, y in tqdm(train_loader):
    optimizer.zero_grad()
    yhat = model(x.view([x.shape[0], 1, 28, 28]))
    loss = loss_function(yhat, y)
    loss.backward()
    optimizer.step()



accuracy = []
for x, y in test_loader:
    if x.shape[0] != batch_size:
        continue
    yhat = model(x.view([batch_size, 1, 28, 28]))
    accuracy.append((yhat.argmax(1) == y).float().mean().item())
print(np.mean(accuracy)) #0.9833733974358975

