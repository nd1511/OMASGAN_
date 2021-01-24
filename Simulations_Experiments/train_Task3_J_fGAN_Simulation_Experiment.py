from __future__ import print_function
from datasets_Task1_fGAN_Simulation_Experiment import *
from networks_Task1_fGAN_Simulation_Experiment import *
#from losses_Task1_fGAN_Simulation_Experiment import *
#from losses_Task3_fGAN_Simulation_Experiment import *
from losses_Task3_J_fGAN_Simulation_Experiment import *
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Use the leave-one-out (LOO) evaluation methodology.
# The LOO evaluation methodology is setting K classes of a dataset with (K + 1)
# classes as the normal class and the leave-out class as the abnormal class.
#abnormal_class_LOO = abnormal_class_LOO
abnormal_class_LOO = 0
#abnormal_class_LOO = 1
#abnormal_class_LOO = 2
# Select the learning rate.
# Double the learning rate if you double the batch size.
#lr_select_disc = lr_select_disc
lr_select_disc = 1.0e-3
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
#nrand = 128
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
# Choose and select the batch size.
# Double the learning rate if you double the batch size.
#batchsize = batchsize
batchsize = 64
optimizer_disc = optim.Adam(fgan.disc.parameters(), lr=lr_select_disc)
data_forTrainloader = choose_dataset(select_dataset)
from torch.utils.data import Subset
def get_target_label_idx(labels, targets):
  return np.argwhere(np.isin(labels, targets)).flatten().tolist()
train_idx_normal = get_target_label_idx(data_forTrainloader.targets, np.delete(np.array(list(range(0, 10))), abnormal_class_LOO))
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, [1, 2, 3, 4, 5, 6, 7, 8, 9])
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, [0, 2, 3, 4, 5, 6, 7, 8, 9])
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, [0, 1, 3, 4, 5, 6, 7, 8, 9])
# We use the leave-one-out (LOO) evaluation methodology.
# The LOO methodology is setting K classes of a dataset with (K + 1) classes
# as the normal class and the leave-out class as the abnormal class.
data_forTrainloader = Subset(data_forTrainloader, train_idx_normal)
print(len(data_forTrainloader))
trainloader = torch.utils.data.DataLoader(data_forTrainloader, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)
writer = SummaryWriter(log_dir="runModel/MNIST", comment="f-GAN-Pearson")
nepochs = 500
niter = 0
#checkpoint = torch.load('./.pt')
#fgan.disc.load_state_dict(checkpoint['disc_state_dict'])
#optimizer_disc.load_state_dict(checkpoint['disc_opt_state_dict'])
fgan.gen.eval()
fgan.disc.train()
for param in fgan.gen.parameters():
    param.requires_grad = False
# xmodel is G'(z) and xreal2 is B(z)
# 2 choices: Load xmodel and xreal2 as models or as samples
# Either load xmodel and xreal2 as models or as samples
# We choose to load xmodel and xreal2 as models
# xmodel is G'(z)
# Load xmodel: Load G'(z)
fgan2 = FGANLearningObjective(gen, disc, "pearson", gamma=10.0)
fgan2 = fgan2.to(device)
# Load Task 3 because xmodel is G'(z)
#checkpoint = torch.load('./.pt')
checkpoint = torch.load('./Task3_fGAN_Simulation_Experiment.pt')
#fgan2.gen.load_state_dict(checkpoint['gen_model_state_dict'])
fgan2.gen.load_state_dict(checkpoint['gen_state_dict'])
fgan2.gen.eval()
fgan2.disc.eval()
for param in fgan2.gen.parameters():
    param.requires_grad = False
for param in fgan2.disc.parameters():
    param.requires_grad = False
# xreal2 is B(z)
# Load xreal2: Load B(z)
fgan3 = FGANLearningObjective(gen, disc, "pearson", gamma=10.0)
fgan3 = fgan3.to(device)
# Load Task 2 because xreal2 is B(z)
#checkpoint = torch.load('./.pt')
checkpoint = torch.load('./Task2_fGAN_Simulation_Experiment.pt')
#fgan2.gen.load_state_dict(checkpoint['gen_model_state_dict'])
fgan2.gen.load_state_dict(checkpoint['gen_state_dict'])
fgan3.gen.eval()
fgan3.disc.eval()
for param in fgan3.gen.parameters():
    param.requires_grad = False
for param in fgan3.disc.parameters():
    param.requires_grad = False
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
        # xmodel is G'(z) and xreal2 is B(z)
        # 2 choices: Load xmodel and xreal2 as models or as samples
        # Either load xmodel and xreal2 as models or as samples
        #loss_disc = fgan(xreal, xmodel, xreal2)
        # xmodel is G'(z) and xreal2 is B(z)
        # 2 choices: Load xmodel and xreal2 as models or as samples
        # We choose to load xmodel and xreal2 as models
        loss_disc = fgan(xreal, fgan2.gen(zmodel), fgan3.gen(zmodel))
        writer.add_scalar('obj/disc', loss_disc, niter)
        if i == 0:
            print("epoch %d  iter %d  obj(D) %.4f" % (epoch, niter, loss_disc))
        fgan.disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
        # Save the trained model J.
        if epoch >= 100:
            if epoch % 50 == 0:
                # torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
                #            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
                torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
                            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task3_J_fGAN_Simulation_Experiment.pt')
        # torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
        #            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task3_J_fGAN_Simulation_Experiment.pt')
        # if epoch >= 20:
        #     if epoch % 10 == 0:
        #         # torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
        #         #            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
        #         torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
        #                     'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task3_J_fGAN_Simulation_Experiment.pt')
# torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
#            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task3_J_fGAN_Simulation_Experiment.pt')
writer.export_scalars_to_json("./allscalars.json")
writer.close()
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
