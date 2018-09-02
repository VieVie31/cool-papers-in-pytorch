"""
 Idea from : Distilling the Knowledge in a Neural Network
 https://arxiv.org/pdf/1503.02531.pdf
"""
import os
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary
from torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision.transforms as transforms

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

class FCNetwork(nn.Module):
    def __init__(self, hidden_size=1200, T=1, d=.5):
        super(FCNetwork, self).__init__()
        self.T = T
        self.clf = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(d),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 10)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.clf(x)
        out = out / self.T
        return self.softmax(out)

def train(model, optimizer, loss_function, nb_epochs=30):
    for i in trange(nb_epochs):
        big_network.train()
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([batch_size, -1]), Variable(target)
            out = model(x)
            loss = loss_function(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(sum(loss_history) / len(loss_history), end='\t')
        big_network.eval()
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([batch_size, -1]), Variable(target)
            out = model(x)
            loss = loss_function(out, target)
            loss_history.append(loss.item())
        print(sum(loss_history) / len(loss_history))

def accuracy(model):
    model.eval()
    t = []
    for x, target in train_loader:
        x, target = Variable(x).view([batch_size, -1]), Variable(target)
        out = model(x)
        t += [sum(out.argmax(1) == target).item() / batch_size]
    return np.array(t).mean()


big_network = FCNetwork(800)
optimizer = optim.Adam(big_network.parameters())
loss_function = nn.CrossEntropyLoss()
train(big_network, optimizer, loss_function, 30) #re-iter until convergence
print("teacher accuracy : ", accuracy(big_network)) #0.9911666666666666

small_network = FCNetwork(30, d=0.1)
optimizer = optim.Adam(small_network.parameters())
loss_function = nn.CrossEntropyLoss()
train(small_network, optimizer, loss_function, 30) #re-iter until convergence
print("small net accuracy : ", accuracy(small_network)) #0.96785

def distillation_loss_function(model_pred, teach_pred, target, T, alpha=0.9):
    return nn.KLDivLoss()(model_pred, teach_pred) * (T * T * alpha) +\
        nn.CrossEntropyLoss()(model_pred, target) * (1 - alpha)


def distill(student, teacher, T, optimizer, nb_epochs=30, alpha=0.9):
    student.T = T
    teacher.T = T
    student.train()
    teacher.eval()
    for i in trange(nb_epochs):
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([batch_size, -1]), Variable(target)
            student_pred = student(x)
            teacher_pred = teacher(x).detach() #to not requires grad
            loss = distillation_loss_function(student_pred, teacher_pred, target, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(sum(loss_history) / float(len(loss_history)))
    student.T = 1.
    teacher.T = 1.

small_network_d = FCNetwork(30, d=.01)
optimizer = optim.Adam(small_network_d.parameters())
distill(small_network_d, big_network, 20, optimizer, 30, 1.) #re-iter until convergence
print("small net as student accuracy : ", accuracy(small_network_d)) #0.9909666666666667




