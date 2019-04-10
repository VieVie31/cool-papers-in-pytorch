"""
Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection
https://arxiv.org/pdf/1904.02639.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch as T
import torch.nn as nn

from torch.nn.modules import *

from tqdm import tqdm, trange
from torchvision import datasets, transforms

from sklearn.metrics import f1_score, accuracy_score


T.set_default_tensor_type('torch.FloatTensor')

batch_size = 32
nb_epochs  = 1
nb_digits  = 10


train_normals = [
    img for img, lbl in datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
    ) if lbl == 9
]
train_normals = torch.utils.data.TensorDataset(
    torch.tensor([v.numpy() for v in train_normals])
)
train_normals_loader = T.utils.data.DataLoader(
    train_normals, 
    batch_size=batch_size, 
    shuffle=True
) 


train_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True
)
 
test_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=False
) 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  16, 1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cnn(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,  1, 3, ),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.cnn(x) #[B, 1, 26, 26]

class Memory(nn.Module):
    def __init__(self, dimention, capacity=100, lbd=.02):
        super(Memory, self).__init__()
        self.cap = capacity
        self.dim = dimention
        self.lbd = lbd
        self.mem = T.rand((capacity, dimention), requires_grad=True)
        self.cos_sim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(1)

    def forward(self, z):
        #z should be : [BATCH, dimention]
        z = z.unsqueeze(1)
        #compute w with attention
        w = self.softmax(self.cos_sim(
            z.permute(0, 2, 1),
            self.mem.expand(z.shape[0], self.cap, self.dim).permute(0, 2, 1)
        ))
        #hard-shrinking of w
        t = w - self.lbd
        w_hat = (T.max(t, T.zeros(w.shape)) * w) / (abs(t) + 1e-15)
        print("average number of 0ed adresses", ((w_hat == 0).sum(1)).float().mean())
        w_hat = (w_hat + 1e-15) / (w_hat + 1e-15).sum(1).reshape(-1, 1) #adding epsilon because of infinity graidnt => nan
        #compute the w_hat enery by request
        adressing_enery = (-w_hat * T.log(w_hat + 1e-3)).sum(0)
        #get z_hat from memory with the computer soft adresseses w_hat
        z_hat = w_hat.mm(self.mem)
        return z_hat, adressing_enery

# Build the proposed model
class MemAE(nn.Module):
    def __init__(self, dimension=2304, capacity=100, lbd=.002):
        super(MemAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.memory  = Memory(dimention=dimension, capacity=capacity, lbd=lbd)
    
    def forward(self, x):
        # Compute z and flatten it
        z = self.encoder(x)
        encoded_input_shape = z.shape
        z = z.reshape(z.shape[0], -1)
        # Get the new z_hat latent representation and the energy required for retriving it
        z_hat, adressing_enery = self.memory(z)
        # Decode the new latent representation
        out = self.decoder(z_hat.reshape(encoded_input_shape))
        return out, adressing_enery

    def parameters(self):
        for p in self.encoder.parameters():
            yield p
        for p in self.decoder.parameters():
            yield p
        yield self.memory.mem
        return

# Train a classic ConvAE for future comparison
classic_AE = nn.Sequential(Encoder(), Decoder())

optimizer = torch.optim.Adam(classic_AE.parameters())
loss_function = nn.BCELoss()

classic_AE.train()
for (x,) in tqdm(train_normals_loader):
    y = x[:, :, 1:-1, 1:-1]
    optimizer.zero_grad()
    yhat = classic_AE(x.view([x.shape[0], 1, 28, 28]))
    loss = loss_function(yhat, y)
    loss.backward()
    optimizer.step()


# Train the proposed anomaly detection autoencoder
anomdec_memae = MemAE(lbd=.01)

optimizer = torch.optim.Adam(anomdec_memae.parameters())
loss_function = nn.BCELoss()

anomdec_memae.train()
for i in range(2):
    for (x,) in tqdm(train_normals_loader):
        y = x[:, :, 1:-1, 1:-1]
        optimizer.zero_grad()
        yhat, energy = anomdec_memae(x.view([x.shape[0], 1, 28, 28]))
        loss = loss_function(yhat, y) + (.002 * energy).mean()
        loss.backward()
        optimizer.step()
        #slowly augment the sparse regulariation for addressing
        anomdec_memae.memory.lbd = min(anomdec_memae.memory.lbd + 1e-5, 0.01005)
        print(loss.item(), energy.mean().item())

# Try to classify 9 or not 9 after learning only on 9 on the test set after fining the optimal threshold a posteriori

# Print the classical reconstruction error with normal AE (at 1.5 std)
classic_recontruction = []
labels = []
for xx, yy in tqdm(test_loader):
    classic_recontruction.extend(
        ((classic_AE(xx) - xx[:, :, 1:-1, 1:-1]) ** 2).sum(1).sum(1).sum(1).detach().numpy()
    )
    labels.extend(yy.numpy())

print(
    "classical mean training reconstruction error on normal : ", 
    np.array(classic_recontruction)[np.array(labels) == 9].mean()
)
print(
    "classical mean training reconstruction error on abnormal : ", 
    np.array(classic_recontruction)[np.array(labels) != 9].mean()
)

naive_th = np.array(classic_recontruction)[np.array(labels) == 9].mean() + 1.5 * np.array(classic_recontruction)[np.array(labels) == 9].std()

print("classical AE f1 :",       f1_score(np.array(labels) == 9, classic_recontruction < naive_th))
print("classical AE acc:", accuracy_score(np.array(labels) == 9, classic_recontruction < naive_th))
#classical AE f1 : 0.1899810019
#classical AE acc: 0.1899

# Compare with the new method
memae_recontruction = []
labels = []
for xx, yy in tqdm(test_loader):
    memae_recontruction.extend(
        ((anomdec_memae(xx)[0] - xx[:, :, 1:-1, 1:-1]) ** 2).sum(1).sum(1).sum(1).detach().numpy()
    )
    labels.extend(yy.numpy())

print(
    "anomdec_memae mean training reconstruction error on normal : ", 
    np.array(memae_recontruction)[np.array(labels) == 9].mean()
)
print(
    "anomdec_memae mean training reconstruction error on abnormal : ", 
    np.array(memae_recontruction)[np.array(labels) != 9].mean()
)

naive_th = np.array(memae_recontruction)[np.array(labels) == 9].mean() + 1.5 * np.array(memae_recontruction)[np.array(labels) == 9].std()

print("memory AE f1 :",       f1_score(np.array(labels) == 9, memae_recontruction < naive_th))
print("memory AE acc:", accuracy_score(np.array(labels) == 9, memae_recontruction < naive_th))
#memory AE f1 : 0.455628495016
#memory AE acc: 0.7761



