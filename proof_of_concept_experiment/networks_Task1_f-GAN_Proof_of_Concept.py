import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
class DCGANGenerator(nn.Module):
    def __init__(self, nrand, select_dataset):
        super(DCGANGenerator, self).__init__()
        self.lin1 = nn.Linear(nrand, 4 * 4 * 512)
        init.xavier_uniform_(self.lin1.weight, gain=0.1)
        self.lin1bn = nn.BatchNorm1d(4 * 4 * 512)
        self.dc1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(256)
        self.dc2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(128)
        self.dc3a = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc3abn = nn.BatchNorm2d(64)
        if select_dataset == "mnist":
            self.dc3b = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        elif select_dataset == "mnist2" or select_dataset == "cifar10":
            self.dc3b = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        else:
            raise ValueError("Unknown dataset.")
    def forward(self, z):
        h = F.relu(self.lin1bn(self.lin1(z)))
        h = torch.reshape(h, (-1, 512, 4, 4))
        h = F.relu(self.dc1bn(self.dc1(h)))
        h = F.relu(self.dc2bn(self.dc2(h)))
        h = F.relu(self.dc3abn(self.dc3a(h)))
        x = self.dc3b(h)
        return x
class DCGANDiscriminator(nn.Module):
    def __init__(self, select_dataset):
        super(DCGANDiscriminator, self).__init__()
        if select_dataset == "mnist":
            self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        elif select_dataset == "mnist2" or select_dataset == "cifar10":
            self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        else:
            raise ValueError("Unknown dataset.")
        self.conv1bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv2bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv3bn = nn.BatchNorm2d(256)
        self.lin1 = nn.Linear(4 * 4 * 256, 512)
        self.lin1bn = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 1)
    def forward(self, x):
        h = F.elu(self.conv1bn(self.conv1(x)))
        h = F.elu(self.conv2bn(self.conv2(h)))
        h = F.elu(self.conv3bn(self.conv3(h)))
        h = torch.reshape(h, (-1, 4 * 4 * 256))
        h = F.elu(self.lin1bn(self.lin1(h)))
        v = self.lin2(h)
        return v
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
