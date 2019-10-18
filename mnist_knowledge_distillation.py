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

from tqdm import tqdm, trange
from torch.autograd import Variable


_FORMAT_PROGRESS_BAR = r"|{bar}|{n_fmt}/{total_fmt} epoch [{elapsed}<{remaining}]{postfix}"


class FCNetwork(nn.Module):
    """ FC Network with 1 hidden layer, ReLU as activation and batchnorm + dropout.
    Includes a `distillation_temperature` parameter to weights the confidency the network has in its output.
    """

    def __init__(self, dim_input, dim_hidden, dim_output, dropout=0.0, distillation_temperature=1.0):
        super(FCNetwork, self).__init__()
        self.distillation_temperature = distillation_temperature
        self.clf = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.clf(x)
        out /= self.distillation_temperature
        return self.softmax(out)


def main(path_dir_data, batch_size, nb_epochs, dropout, dim_hidden_teacher, dim_hidden_student,
         distillation_temperature, alpha, cuda, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

    if not os.path.exists(path_dir_data):
        os.mkdir(path_dir_data)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(root=path_dir_data, train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(root=path_dir_data, train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    dim_input = 28 * 28  # MNIST picture's dim
    dim_output = 10  # MNIST number of classes

    # Learn the teacher alone
    model_teacher = FCNetwork(dim_input=dim_input, dim_hidden=dim_hidden_teacher, dim_output=dim_output,
                              dropout=dropout).to(device)
    optimizer = optim.Adam(model_teacher.parameters())
    criterion = nn.CrossEntropyLoss()
    print("Training teacher :")
    train(model=model_teacher, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
          test_loader=test_loader, nb_epochs=nb_epochs, device=device)

    # Learn the small network alone for comparisons
    model_student = FCNetwork(dim_input=dim_input, dim_hidden=dim_hidden_student, dim_output=dim_output).to(device)
    optimizer = optim.Adam(model_student.parameters())
    criterion = nn.CrossEntropyLoss()
    print("\nTraining student alone :")
    train(model=model_student, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
          test_loader=test_loader, nb_epochs=nb_epochs, device=device)

    # Learn the same small network by distillation
    model_student_d = FCNetwork(dim_input=dim_input, dim_hidden=dim_hidden_student, dim_output=dim_output).to(device)
    optimizer = optim.Adam(model_student_d.parameters())
    print("\nTraining student with teacher :")
    train_distillation(model_student=model_student_d, model_teacher=model_teacher, optimizer=optimizer,
                       train_loader=train_loader, test_loader=test_loader, nb_epochs=nb_epochs,
                       distillation_temperature=distillation_temperature, alpha=alpha, device=device)


def train(*, model, optimizer, criterion, train_loader, test_loader, device, nb_epochs=10):
    with tqdm(total=nb_epochs, bar_format=_FORMAT_PROGRESS_BAR) as progress_bar:
        for _ in range(nb_epochs):
            model.train()
            loss_history = []
            for x, target in train_loader:
                x = x.view([x.shape[0], -1]).to(device=device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)
                out = model(x)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.detach().cpu().item())
            train_loss = sum(loss_history) / len(loss_history)

            model.eval()
            loss_history = []
            accuracy_history = []
            for x, target in test_loader:
                x = x.view([x.shape[0], -1]).to(device=device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)
                out = model(x)
                loss = criterion(out, target)
                loss_history.append(loss.detach().cpu().item())
                accuracy_history.append(sum(out.argmax(1) == target).detach().cpu().item() / x.shape[0])
            test_loss = sum(loss_history) / len(loss_history)
            test_accuracy = sum(accuracy_history) / len(accuracy_history)

            # Update the progress bar
            progress_bar.update()
            progress_bar.set_postfix({
                "train_loss": "{0:.6f}".format(train_loss),
                "test_loss": "{0:.6f}".format(test_loss),
                "test_accuracy": "{0:.6f}".format(test_accuracy)
            })


def train_distillation(*, model_student, model_teacher, optimizer, train_loader, test_loader, device, nb_epochs=10,
                       distillation_temperature=20.0, alpha=0.7):
    model_student.distillation_temperature = distillation_temperature
    model_teacher.distillation_temperature = distillation_temperature
    model_teacher.eval()
    with tqdm(total=nb_epochs, bar_format=_FORMAT_PROGRESS_BAR) as progress_bar:
        for _ in range(nb_epochs):
            model_student.train()
            loss_history = []
            for x, target in train_loader:
                x = x.view([x.shape[0], -1]).to(device=device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)
                student_pred = model_student(x)
                teacher_pred = model_teacher(x).detach()
                loss = criterion_distillation(student_pred, teacher_pred, target, alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.detach().cpu().item())
            train_loss = sum(loss_history) / len(loss_history)

            model_student.eval()
            loss_history = []
            accuracy_history = []
            for x, target in test_loader:
                x = x.view([x.shape[0], -1]).to(device=device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)
                out = model_student(x)
                loss = nn.CrossEntropyLoss()(out, target)
                loss_history.append(loss.detach().cpu().item())
                accuracy_history.append(sum(out.argmax(1) == target).detach().cpu().item() / x.shape[0])
            test_loss = sum(loss_history) / len(loss_history)
            test_accuracy = sum(accuracy_history) / len(accuracy_history)

            # Update the progress bar
            progress_bar.update()
            progress_bar.set_postfix({
                "train_loss": "{0:.6f}".format(train_loss),
                "test_loss": "{0:.6f}".format(test_loss),
                "test_accuracy": "{0:.6f}".format(test_accuracy)
            })
    model_student.distillation_temperature = 1.0
    model_teacher.distillation_temperature = 1.0


def criterion_distillation(model_pred, teach_pred, target, distillation_temperature, alpha=0.7):
    return nn.KLDivLoss(reduction='batchmean')(model_pred, teach_pred) * (distillation_temperature ** 2 * 2. * alpha) \
        + nn.CrossEntropyLoss()(model_pred, target) * (1 - alpha)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", '--path-dir-data', type=str, default="data",
                        help='Path to the root directory containing the MNIST dataset')
    parser.add_argument('-b', "--batch-size", type=int, default=128,
                        help='Size of a dataset batch')
    parser.add_argument('-e', "--epoch", type=int, default=10,
                        help='Number of epochs for each training')
    parser.add_argument('-d', "--dropout", type=float, default=0.8,
                        help='Dropout probability during training')
    parser.add_argument("--dim-hidden-teacher", type=int, default=1200,
                        help='Dimensionality of the hidden layer for the teacher network')
    parser.add_argument("--dim-hidden-student", type=int, default=800,
                        help='Dimensionality of the hidden layer for the students networks')
    parser.add_argument("-t", "--distillation-temperature", type=float, default=20.0,
                        help='Weights the confidency the network has in its output.')
    parser.add_argument("-a", "--alpha", type=float, default=0.7,
                        help='Regularisation parameter for the distillation loss.')
    parser.add_argument("-c", "--cuda", action='store_true',
                        help='Train networks with cuda (if available)')
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help='Random seed to fix. No seed is fixed if none is provided')
    args = parser.parse_args()

    main(
        path_dir_data=args.path_dir_data,
        batch_size=args.batch_size,
        nb_epochs=args.epoch,
        dropout=args.dropout,
        dim_hidden_teacher=args.dim_hidden_teacher,
        dim_hidden_student=args.dim_hidden_student,
        distillation_temperature=args.distillation_temperature,
        alpha=args.alpha,
        cuda=args.cuda,
        seed=args.seed
    )
