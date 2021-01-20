from __future__ import print_function
from datasets_Task1_fGAN_Proof_of_Concept import *
from networks_Task1_fGAN_Proof_of_Concept import *
from losses_Task1_fGAN_Proof_of_Concept import *
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
seed_value = 2
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nrand = 100
#select_dataset = "select_dataset"
select_dataset = "mnist"
#select_dataset = "mnist2"
#select_dataset = "cifar10"
gen = DCGANGenerator(nrand, select_dataset)
disc = DCGANDiscriminator(select_dataset)
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
fgan = FGANLearningObjective(gen, disc, "pearson", gamma=10.0)
fgan = fgan.to(device)
batchsize = 64
optimizer_gen = optim.Adam(fgan.gen.parameters(), lr=1.0e-3)
optimizer_disc = optim.Adam(fgan.disc.parameters(), lr=1.0e-3)
data_forTrainloader = choose_dataset(select_dataset)
from torch.utils.data import Subset
def get_target_label_idx(labels, targets):
  return np.argwhere(np.isin(labels, targets)).flatten().tolist()
train_idx_normal = get_target_label_idx(data_forTrainloader.targets, [1, 2, 3, 4, 5, 6, 7, 8, 9])
data_forTrainloader = Subset(data_forTrainloader, train_idx_normal)
print(len(data_forTrainloader))
trainloader = torch.utils.data.DataLoader(data_forTrainloader, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)
writer = SummaryWriter(log_dir="runs/CIFAR10", comment="f-GAN-Pearson")
nepochs = 500
niter = 0
for epoch in range(nepochs):
    zmodel = Variable(torch.rand((batchsize,nrand), device=device))
    xmodel = fgan.gen(zmodel)
    xmodelimg = vutils.make_grid(xmodel, normalize=True, scale_each=True)
    writer.add_image('Generated', xmodelimg, global_step=niter)
    for i, data in enumerate(trainloader, 0):
        niter += 1
        imgs, labels = data
        fgan.zero_grad()
        xreal = Variable(imgs.to(device), requires_grad=True)
        zmodel = Variable(torch.rand((batchsize,nrand), device=device))
        loss_gen, loss_disc = fgan(xreal, zmodel)
        writer.add_scalar('obj/disc', loss_disc, niter)
        writer.add_scalar('obj/gen', loss_gen, niter)
        if i == 0:
            print("epoch %d  iter %d  obj(D) %.4f  obj(G) %.4f" % (epoch, niter, loss_disc, loss_gen))
        fgan.gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizer_gen.step()
        fgan.disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
writer.export_scalars_to_json("./allscalars.json")
writer.close()
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
