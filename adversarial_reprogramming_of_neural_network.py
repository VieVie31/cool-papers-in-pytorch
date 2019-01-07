"""
Re-implementation of "Adversarial Reprogramming of Neural Networks" (ICLR'19)
https://arxiv.org/pdf/1806.11146.pdf
"""
import numpy as np

import torch 
import torch as T
import torch.nn as nn

import torchvision

from tqdm import tqdm
from torchvision import datasets, transforms


def x_to_X(x, X_size, channel_out=3):
    """
    This function places a batch of small image x in the center 
    of a bigger one of size X_size with zero padding.

    :param x: batch x, [batch_size, channels, im_size, im_size]
    :param X_size: the size of the new image 
    :param channel_out: the number of the channel

    :type x: torch.Tensor
    :type X_size: int 
    :type channel_out: int

    :return: x centred in X_size zerroed image 
    :rtype: torch.tensor
    """
    X = T.zeros((x.shape[0], channel_out, X_size, X_size))

    start_x = X_size // 2 - x.shape[2] // 2
    end_x = start_x + x.shape[2] 
    start_y = X_size // 2 - x.shape[3] // 2
    end_y = start_y + x.shape[3]

    x = x.expand(x.shape[0], channel_out, x.shape[2], x.shape[3])
    X[:, :, start_x:end_x, start_y:end_y] = x

    return X

def get_mask(patch_size, X_size, channel_out, batch_size=1):
    """
    This function return the mask for an img of size patch_size 
    which is in the center of a bigger on with size X_size

    :param patch_size: the size of patch that we want to put in the center
    :param X_size: the new size of the img
    :param channel_out: nb channels
    :param batch_size: nb times that the mask will be replicated

    :type patch_size: int
    :type X_size: int
    :type channel_out: int
    :type batch_size: int
    
    :return: binary mask
    :rtype: torch.Tensor 
    """
    ones = T.ones((batch_size, channel_out, patch_size, patch_size))
    return x_to_X(ones, X_size, channel_out)

def get_mnist(batch_size):
    """
    This function retruns the train and test loader of mnist 
    dataset for a given batch_size

    :param batch_size: size of the batch for data loader
    
    :type batch_size: int

    :return: train and test loader
    :rtype: tuple[torch.utils.data.DataLoader]
    """
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
    return train_loader, test_loader

class ProgrammingNetwork(nn.Module):
    """
    This class is the module that contains the network
    that will be uilized and the associated programm 
    that will be learned to hijak the first one
    """

    def __init__(self, pretained_model, input_size, patch_size, channel_out=3):
        """
        Constructor

        :param pretrained_model: the model to hitjak
        :param input_size: the img's size excepected by pretrained_model
        :param patch_size: the size of the small target domain img
        :param channel_out: nb channel
        
        :type pretrained_model: modul
        :type input_size: int
        :type patch_size: int
        :type channel_out: int
        """
        super().__init__()
        self.model = pretained_model
        self.p = T.autograd.Variable(T.randn((channel_out, input_size, input_size)).to(device), requires_grad=True)
        self.input_size = input_size
        self.mask = get_mask(patch_size, input_size, channel_out, batch_size=1)[0].to(device)
        self.mask.requires_grad = False

    def forward(self, x):
        #P = tanh (W + M)
        P = nn.Tanh()((1 - self.mask) * self.p)
        #Xadv = hf (˜x; W) = X˜ + P
        x_adv = x_to_X(x, self.input_size, self.p.shape[0]).to(device) + P
        return self.model(x_adv)

device = "cuda:0"

batch_size = 16
train_loader, test_loader = get_mnist(batch_size)

#pretrained_model = torchvision.models.resnet101(pretrained=True).eval()
pretrained_model = torchvision.models.squeezenet1_0(pretrained=True).eval()

input_size = 224
patch_size = 28

model = ProgrammingNetwork(pretrained_model, input_size, patch_size)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = T.optim.Adam([model.p])


nb_epochs = 20
loss_history = []
for epoch in range(nb_epochs): 
    print("epoch : ", epoch)
    for i, (x, y) in enumerate(tqdm(train_loader)):
        y_hat = model(x.to(device))
        optimizer.zero_grad()
        loss = loss_function(y_hat, y.to(device))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if not i % 50: #save each 50 batches
            #T.save(model.state_dict(), "./models/squeezenet1_0_mnist.pth")
            np.save("./models/squeezenet1_0_mnist_program_{}_{}".format(epoch, i // 50), model.p.detach().to("cpu").numpy())
            np.save("loss_history", loss_history)

    #np.save("loss_history", loss_history)

    #compute test accuracy
    test_accuracy = []
    for i, (x, y) in enumerate(tqdm(test_loader)):
        y_hat = model(x.to(device))
        (y_hat.argmax(1).to('cpu') == y).float()
        test_accuracy.extend((y_hat.argmax(1).to('cpu') == y).float().numpy())

    print("test accuracy : ", np.array(test_accuracy).mean())

