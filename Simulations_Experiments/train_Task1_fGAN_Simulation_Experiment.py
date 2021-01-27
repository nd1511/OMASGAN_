from __future__ import print_function
from datasets_Task1_fGAN_Simulation_Experiment import *
from networks_Task1_fGAN_Simulation_Experiment import *
from losses_Task1_fGAN_Simulation_Experiment import *
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
# Additional acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs)
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--abnormal_class', required=True, type=int, default=0, help='Select the abnormal class.')
parser.add_argument('--abnormal_class', type=int, default=0, help='Select the abnormal class.')
opt = parser.parse_args()
print(opt.abnormal_class)
# Select and set the learning rate.
# Double the learning rate if you double the batch size.
#lr_select = lr_select
lr_select = 1.0e-3
#lr_select = 1.0e-4
lr_select_gen = lr_select
lr_select_disc = lr_select
# Use the leave-one-out (LOO) evaluation methodology.
# The LOO evaluation methodology is setting K classes of a dataset with (K + 1)
# classes as the normal class and the leave-out class as the abnormal class.
# The LOO methodology leads to a multimodal distribution with disconnected components for the normal class.
#abnormal_class_LOO = abnormal_class_LOO
abnormal_class_LOO = opt.abnormal_class
#abnormal_class_LOO = 0
#abnormal_class_LOO = 1
#abnormal_class_LOO = 2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
#random.seed(2019)
#np.random.seed(2019)
#seed_value = seed_value
seed_value = 2
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
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
# Choose and set the batch size.
# Double the learning rate if you double the batch size.
batchsize = 64
optimizer_gen = optim.Adam(fgan.gen.parameters(), lr=lr_select_gen)
optimizer_disc = optim.Adam(fgan.disc.parameters(), lr=lr_select_disc)
# To boost the AD performance, a scheduler to decrease the learning rate is recommended, i.e. “scheduler
# = optim.lr_scheduler.MultiStepLR(optimizer_gen” and “scheduler.step()”. Also, AE-based pretraining
# using a dictionary for the NN decoder parameters, e.g. “ae_net_dict = {k: v for k, v in”, is recommended.
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
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
def visualize(epoch, model, itr, real_imgs):
    model.eval()
    makedirs(os.path.join('experim', 'images'))
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs
    nvals = 256
    with torch.no_grad():
        fake_imgs = model(Variable(torch.rand((batchsize, nrand), device=device)))
        fake_imgs = fake_imgs.view(-1, 1, 32, 32)
        imgs = torch.cat([_real_imgs, fake_imgs], 0)
        filename = os.path.join('experim', 'images', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        print(filename)
        save_image(imgs.cpu().float(), filename, nrow=16, padding=2)
    model.train()
writer = SummaryWriter(log_dir="runs/MNIST", comment="f-GAN-Pearson")
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
            print("epoch %d  iter %d  loss(G) %.4f  loss(D) %.4f" % (epoch, niter, loss_gen, loss_disc))
        fgan.gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizer_gen.step()
        fgan.disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
    # Save the images and use visualization.
    if epoch % 50 == 0:
        visualize(epoch, fgan.gen, i, xreal)
    # Save the two models G and D.
    if epoch >= 100:
        if epoch % 50 == 0:
            #torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
            #            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
            torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
                        'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task1_fGAN_Simulation_Experiment.pt')
    #torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
    #            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task1_fGAN_Simulation_Experiment.pt')
    # if epoch >= 20:
    #     if epoch % 10 == 0:
    #         # torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
    #         #            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
    #         torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
    #                     'gen_opt_state_dict': optimizer_gen.state_dict(),
    #                     'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task1_fGAN_Simulation_Experiment.pt')
#torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
#            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
torch.save({'gen_state_dict': fgan.gen.state_dict(), 'disc_state_dict': fgan.disc.state_dict(),
            'gen_opt_state_dict': optimizer_gen.state_dict(), 'disc_opt_state_dict': optimizer_disc.state_dict()}, './Task1_fGAN_Simulation_Experiment.pt')
writer.export_scalars_to_json("./allscalars.json")
writer.close()
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
        checkpoint = torch.load('./Task1_fGAN_Simulation_Experiment.pt')
        fgan.gen.load_state_dict(checkpoint['gen_state_dict'])
        fgan.disc.load_state_dict(checkpoint['disc_state_dict'])
        visualize(epoch, fgan.gen, i, xreal)
        break
    break
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
# Additional acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs)
