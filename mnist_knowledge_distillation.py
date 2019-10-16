""" Implementation of : "Distilling the Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf

Authors :
* Olivier Risser-Maroix
* Nicolas Bizzozzéro
"""
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import trange
from torch.autograd import Variable


class FCNetwork(nn.Module):
    """ FC Network with 1 hidden layer, ReLU as activation and batchnorm + dropout.
    Includes a `distillation_temperature` parameter to weights the confidency the network has in its output.
    """

    def __init__(self, dim_input, dim_hidden, dim_output, dropout=0.1, distillation_temperature=1.0):
        super(FCNetwork, self).__init__()
        self.distillation_temperature = distillation_temperature
        self.clf = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden),
            nn.Linear(dim_hidden, dim_output)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.clf(x)
        out /= self.distillation_temperature
        return self.softmax(out)


def main(path_dir_data, batch_size, nb_epochs, dropout, dim_hidden_teacher, dim_hidden_student,
         distillation_temperature, alpha, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if not os.path.exists(path_dir_data):
        os.mkdir(path_dir_data)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(root=path_dir_data, train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(root=path_dir_data, train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=False
    )

    dim_input = 28 * 28  # MNIST picture's dim
    dim_output = 10  # MNIST n° of classes

    # Learn the teacher alone
    model_teacher = FCNetwork(dim_input=dim_input, dim_hidden=dim_hidden_teacher, dim_output=dim_output,
                              dropout=dropout)
    optimizer = optim.Adam(model_teacher.parameters())
    criterion = nn.CrossEntropyLoss()
    train(model=model_teacher, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
          test_loader=test_loader, nb_epochs=nb_epochs)
    print("Accuracy model_teacher                    :", accuracy(model=model_teacher, test_loader=test_loader))

    # Learn the small network alone for comparisons
    model_student = FCNetwork(dim_input=dim_input, dim_hidden=dim_hidden_student, dim_output=dim_output,
                              dropout=dropout)
    optimizer = optim.Adam(model_student.parameters())
    criterion = nn.CrossEntropyLoss()
    train(model=model_student, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
          test_loader=test_loader, nb_epochs=nb_epochs)
    print("Accuracy model_student alone              :", accuracy(model=model_student, test_loader=test_loader))

    # Learn the same small network by distillation
    model_student_d = FCNetwork(dim_input=dim_input, dim_hidden=dim_hidden_student, dim_output=dim_output,
                                dropout=dropout)
    optimizer = optim.Adam(model_student_d.parameters())
    train_distillation(model_student=model_student_d, model_teacher=model_teacher, optimizer=optimizer,
                       train_loader=train_loader, nb_epochs=nb_epochs,
                       distillation_temperature=distillation_temperature, alpha=alpha)
    print("Accuracy model_student with model_teacher :", accuracy(model=model_student_d, test_loader=test_loader))


def train(*, model, optimizer, criterion, train_loader, test_loader, nb_epochs=32):
    for _ in trange(nb_epochs):
        model.train()
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([x.shape[0], -1]), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(sum(loss_history) / len(loss_history), end='\t')

        model.eval()
        loss_history = []
        for x, target in test_loader:
            x, target = Variable(x).view([x.shape[0], -1]), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss_history.append(loss.item())
        print(sum(loss_history) / len(loss_history))


def accuracy(model, test_loader):
    model.eval()
    t = []
    for x, target in test_loader:
        x, target = Variable(x).view([x.shape[0], -1]), Variable(target)
        out = model(x)
        t += [sum(out.argmax(1) == target).item() / x.shape[0]]
    return np.array(t).mean()


def distillation_loss_function(model_pred, teach_pred, target, distillation_temperature, alpha=0.9):
    return nn.KLDivLoss()(model_pred, teach_pred) * (distillation_temperature * distillation_temperature * alpha) +\
        nn.CrossEntropyLoss()(model_pred, target) * (1 - alpha)


def train_distillation(*, model_student, model_teacher, optimizer, train_loader, nb_epochs=32,
                       distillation_temperature=20, alpha=0.9):
    model_student.distillation_temperature = distillation_temperature
    model_teacher.distillation_temperature = distillation_temperature
    model_student.train()
    model_teacher.eval()
    for i in trange(nb_epochs):
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([x.shape[0], -1]), Variable(target)
            student_pred = model_student(x)
            teacher_pred = model_teacher(x).detach()
            loss = distillation_loss_function(student_pred, teacher_pred, target, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(sum(loss_history) / float(len(loss_history)))
    model_student.distillation_temperature = 1.0
    model_teacher.distillation_temperature = 1.0


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", '--path-dir-data', type=str, default="data",
                        help='Path to the root directory containing the MNIST dataset')
    parser.add_argument('-b', "--batch-size", type=int, default=32,
                        help='Size of a dataset batch')
    parser.add_argument('-e', "--epochs", type=int, default=32,
                        help='Number of epochs for each training')
    parser.add_argument('-d', "--dropout", type=float, default=0.1,
                        help='Dropout probability during training')
    parser.add_argument("--dim-hidden-teacher", type=int, default=512,
                        help='Dimensionality of the hidden layer for the teacher network')
    parser.add_argument("--dim-hidden-student", type=int, default=32,
                        help='Dimensionality of the hidden layer for the students networks')
    parser.add_argument("-t", "--distillation-temperature", type=float, default=20.0,
                        help='Weights the confidency the network has in its output.')
    parser.add_argument("-a", "--alpha", type=float, default=0.9,
                        help='Regularisation parameter for the distillation loss.')
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help='Random seed to fix. No seed is fixed if none is provided')
    args = parser.parse_args()

    main(
        path_dir_data=args.path_dir_data,
        batch_size=args.batch_size,
        nb_epochs=args.epochs,
        dropout=args.dropout,
        dim_hidden_teacher=args.dim_hidden_teacher,
        dim_hidden_student=args.dim_hidden_student,
        distillation_temperature=args.distillation_temperature,
        alpha=args.alpha,
        seed=args.seed
    )
