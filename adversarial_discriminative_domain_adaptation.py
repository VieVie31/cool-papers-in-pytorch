"""
Reimplementation of 'Adversarial Discriminative Domain Adaptation' (CVPR `17)
http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf

Doing domain adaptation from USPS (source) to MNIST (target) with adversarial approch.

Step 0 : Pre-training on USPS...
 - MNIST acc : 0.689181170886076
 - USPS  acc : 0.9217218120892843
Step 1 : Adversarial adaptation for MNIST...
 - MNIST acc : 0.7762419871794872 (max obtained after several run)
"""
import os
import gzip
import pickle
import urllib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch as T
import torch.nn as nn
import torch.utils.data as data


from tqdm import tqdm, trange
from torchvision import datasets, transforms

T.manual_seed(3)

print("""'Adversarial Discriminative Domain Adaptation' (CVPR `17)
    Source : USPS
    Target : MNIST
""")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 48, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.Tanh(),
        )

        self.flatten = Flatten()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.clf(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Linear(128, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2)
        )
    
    def forward(self, x):
        return self.d(x)

batch_size = 128

source_encoder = Encoder()
target_encoder = Encoder()
classifier = Classifier()
discriminator = Discriminator()


class USPS(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    #FEW ADAPTATIONS DONE ON THIS CODE... NOT ORIGINAL ONE...

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN_PyTorch/master/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        #self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])[0]
        # label = torch.FloatTensor([label.item()])
        return T.tensor(img).permute(2, 0, 1).t().transpose(1, 2).float(), T.tensor(label).long()

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

mnist_train_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
)

mnist_test_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 

usps_train_loader = T.utils.data.DataLoader(USPS(
    './ressources', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
)

usps_test_loader = T.utils.data.DataLoader(USPS(
    './ressources', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 


model = nn.Sequential(
    source_encoder,
    classifier
)
optimizer = T.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()


print("Step 0 : Pre-training on USPS...")

train_history = []
for i in trange(20): #pretrain USPS for 20 epochs
    batch_loss = []
    for x, y in usps_train_loader:
        optimizer.zero_grad()
        yhat = model(x.view([x.shape[0], 1, 28, 28]))
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    train_history.extend(batch_loss)


accuracy = []
for x, y in mnist_test_loader:
    yhat = model(x)
    accuracy.append((yhat.argmax(1) == y).float().mean().item())
print(" - MNIST acc : ", np.mean(accuracy)) 


accuracy = []
for x, y in usps_test_loader:
    yhat = model(x)
    accuracy.append((yhat.argmax(1) == y).float().mean().item())
print(" - USPS acc : ", np.mean(accuracy)) 


print("Step 1 : Adversarial adaptation for MNIST...")

discriminator = Discriminator()

source_encoder.eval()
target_encoder.load_state_dict(source_encoder.state_dict())

for p in source_encoder.parameters():
    p.requires_grad = False

d_optimizer = T.optim.Adam(discriminator.parameters(),  lr=.0001, betas=(.5, .999))
g_optimizer = T.optim.Adam(target_encoder.parameters(), lr=.0001, betas=(.5, .999))

d_train_history = []
g_train_history = []
for i in trange(10): #doing domain adaptation for 10 epochs
    for i, ((x_mnist, y_mnist), (x_usps, y_usps)) in enumerate(zip(mnist_train_loader, usps_train_loader)):
        mnist_features = target_encoder(x_mnist.view([x_mnist.shape[0], 1, 28, 28]))
        usps_features  = source_encoder(x_usps.view([ x_usps.shape[0],  1, 28, 28]))
        features       = T.cat((usps_features, mnist_features), dim=0)

        mnist_labels   = T.ones(mnist_features.shape[0]).long()
        usps_labels    = T.zeros(usps_features.shape[0]).long()
        labels         = T.cat((usps_labels, mnist_labels), dim=0)

        #train discriminator
        d_optimizer.zero_grad()
        yhat = discriminator(features.detach())
        d_loss = nn.CrossEntropyLoss()(yhat, T.autograd.Variable(labels))
        d_loss.backward()
        d_optimizer.step()

        #train generator
        g_optimizer.zero_grad()
        target_encoder.zero_grad()

        yhat = discriminator(mnist_features)
        g_loss = nn.CrossEntropyLoss()(yhat, T.autograd.Variable(mnist_labels - 1))
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_train_history.append(d_loss.item())
        g_train_history.append(g_loss.item())


    model = nn.Sequential(
        target_encoder,
        classifier
    )
    accuracy = []
    for x, y in mnist_test_loader:
        yhat = model(x)
        accuracy.append((yhat.argmax(1) == y).float().mean().item())
    print(" - MNIST acc : ", np.mean(accuracy)) 


plt.plot(d_train_history)
plt.plot(g_train_history)
plt.show()


model = nn.Sequential(
    target_encoder,
    classifier
)
model.eval()

accuracy = []
for x, y in mnist_test_loader:
    if x.shape[0] != batch_size:
        continue
    yhat = model(x)
    accuracy.append((yhat.argmax(1) == y).float().mean().item())
print(" - MNIST acc : ", np.mean(accuracy)) 





best_run_log = """
'Adversarial Discriminative Domain Adaptation' (CVPR `17)
    Source : USPS
    Target : MNIST

Step 0 : Pre-training on USPS...
100%|███████████████████████████████████████████| 20/20 [04:04<00:00, 12.21s/it]
 - MNIST acc :  0.689181170886076
 - USPS acc :  0.9217218120892843
Step 1 : Adversarial adaptation for MNIST...
  0%|                                                    | 0/10 [00:00<?, ?it/s] - MNIST acc :  0.7122231012658228
 10%|████▍                                       | 1/10 [00:28<04:17, 28.58s/it] - MNIST acc :  0.6709849683544303
 20%|████████▊                                   | 2/10 [00:58<03:53, 29.21s/it] - MNIST acc :  0.682060917721519
 30%|█████████████▏                              | 3/10 [01:33<03:38, 31.25s/it] - MNIST acc :  0.7244857594936709
 40%|█████████████████▌                          | 4/10 [02:02<03:04, 30.74s/it] - MNIST acc :  0.7454509493670886
 50%|██████████████████████                      | 5/10 [02:32<02:32, 30.50s/it] - MNIST acc :  0.7404074367088608
 60%|██████████████████████████▍                 | 6/10 [03:03<02:02, 30.59s/it] - MNIST acc :  0.7669106012658228
 70%|██████████████████████████████▊             | 7/10 [03:34<01:31, 30.60s/it] - MNIST acc :  0.7682950949367089
 80%|███████████████████████████████████▏        | 8/10 [04:04<01:01, 30.57s/it] - MNIST acc :  0.7712618670886076
 90%|███████████████████████████████████████▌    | 9/10 [04:36<00:30, 30.73s/it] - MNIST acc :  0.7752175632911392
100%|███████████████████████████████████████████| 10/10 [05:06<00:00, 30.65s/it]
2018-12-20 17:34:35.451 Python[21283:34501747] ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to (null)
 - MNIST acc :  0.7762419871794872
 """

 