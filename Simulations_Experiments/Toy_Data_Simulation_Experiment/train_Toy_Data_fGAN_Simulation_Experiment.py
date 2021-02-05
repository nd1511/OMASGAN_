from __future__ import print_function
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
lr_select = lr_select
#lr_select = 0.525e-6
lr_select_gen = lr_select
lr_select_disc = lr_select
mu_select = mu_select
ni_select = ni_select
import math
import torch
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
df = 40.0
tdist = scipy.stats.t(df, loc=1.5, scale=0.15)
tdist2 = scipy.stats.t(df, loc=1.5, scale=0.15)
ntrain = 5000
#ntrain = 10000
Xtrain = tdist.rvs(ntrain)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
class ConjugateDualFunction:
    def __init__(self, divergence_name):
        self.divergence_name = divergence_name
    def T(self, v):
        if self.divergence_name == "kl":
            return v
        elif self.divergence_name == "klrev":
            return -F.exp(v)
        # According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
        # After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
        elif self.divergence_name == "pearson":
            return v
        elif self.divergence_name == "neyman":
            return 1.0 - F.exp(v)
        elif self.divergence_name == "hellinger":
            return 1.0 - F.exp(v)
        elif self.divergence_name == "jensen":
            return math.log(2.0) - F.softplus(-v)
        elif self.divergence_name == "gan":
            return -F.softplus(-v)
        else:
            raise ValueError("Unknown f-divergence.")
    def fstarT(self, v):
        if self.divergence_name == "kl":
            return torch.exp(v - 1.0)
        elif self.divergence_name == "klrev":
            return -1.0 - v
        # According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
        # After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
        elif self.divergence_name == "pearson":
            return 0.25*v*v + v
        elif self.divergence_name == "neyman":
            return 2.0 - 2.0*F.exp(0.5*v)
        elif self.divergence_name == "hellinger":
            return F.exp(-v) - 1.0
        elif self.divergence_name == "jensen":
            return F.softplus(v) - math.log(2.0)
        elif self.divergence_name == "gan":
            return F.softplus(v)
        else:
            raise ValueError("Unknown f-divergence.")
class Generator(nn.Module):
    def __init__(self, nhidden):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(2, nhidden)
        self.lin2 = nn.Linear(nhidden, nhidden)
        self.lin3 = nn.Linear(nhidden, nhidden)
        #self.lin4 = nn.Linear(nhidden, nhidden)
        self.lin4 = nn.Linear(nhidden, 2)
    def forward(self, z):
        h = F.relu(self.lin1(z))
        h = F.relu(self.lin2(h))
        h = F.relu(self.lin3(h))
        x = self.lin4(h)
        x += z
        return x
class Discriminator(nn.Module):
    def __init__(self, nhidden):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(2, nhidden)
        self.lin2 = nn.Linear(nhidden, nhidden)
        self.lin3 = nn.Linear(nhidden, nhidden)
        self.lin4 = nn.Linear(nhidden, 1)
    def forward(self, x):
        h = F.relu(self.lin1(x))
        h = F.relu(self.lin2(h))
        h = F.relu(self.lin3(h))
        v = self.lin4(h)
        return v
class FGANLearningObjective(nn.Module):
    def __init__(self, gen, disc, divergence_name="pearson", gamma=10.0):
        super(FGANLearningObjective, self).__init__()
        self.gen = gen
        self.disc = disc
        self.conj = ConjugateDualFunction(divergence_name)
        self.gammahalf = 0.5 * gamma
    def forward(self, xreal, zmodel, mu, ni):
        vreal = self.disc(xreal)
        Treal = self.conj.T(vreal)
        xmodel = self.gen(zmodel)
        vmodel = self.disc(xmodel)
        fstar_Tmodel = self.conj.fstarT(vmodel)
        D1 = torch.norm(xreal[None, :].expand(xmodel.shape[0], -1, -1) - xmodel[:, None], dim=-1)**2
        D2 = torch.norm(zmodel[None, :].expand(zmodel.shape[0], -1, -1) - zmodel[:, None], dim=-1)**2 / (1e-17 + torch.norm(xmodel[None, :].expand(xmodel.shape[0], -1, -1) - xmodel[:, None], dim=-1)**2)
        # The first term in the loss function is a strictly decreasing function of a distribution metric.
        # Create dynamics by pushing the generated samples OoD: Likelihood-free boundary of data distribution
        # We use -m(B, G), where m(B,G) is -fstar_Tmodel.mean().
        loss_gen = fstar_Tmodel.mean() + mu * torch.min(D1, dim=1)[0].mean() + ni * torch.mean(D2, dim=1)[0].mean()
        loss_disc = fstar_Tmodel.mean() - Treal.mean()
        #loss_gen = fstar_Tmodel.mean() + distance + dispersion
        #loss_disc = fstar_Tmodel.mean() - Treal.mean()
        if self.gammahalf > 0.0:
            batchsize = xreal.size(0)
            grad_pd = torch.autograd.grad(Treal.sum(), xreal, create_graph=True, only_inputs=True)[0]
            grad_pd_norm2 = grad_pd.pow(2)
            grad_pd_norm2 = grad_pd_norm2.view(batchsize, -1).sum(1)
            gradient_penalty = self.gammahalf * grad_pd_norm2.mean()
            loss_disc += gradient_penalty
        return loss_gen, loss_disc
# The first two terms of the proposed objective cost function, i.e. $- m( B, G)$ and $d( B, G )$,
# are convex as functions of the underlying probability measures. Because of the neural architectures
# used, the optimization problem is non-convex. Regarding the choice of loss function for the boundary
# Task, we create dynamics by pushing the generated samples OoD. Taking into account results from
# optimization theory and optimization results for non-linear architectures, we introduce a regularized
# cost function in the optimization instead of attempting to solve a much harder constrained optimization.
gen = Generator(64)
disc = Discriminator(64)
gen2 = Generator(64)
disc2 = Discriminator(64)
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
fgan = FGANLearningObjective(gen, disc, "pearson", gamma=1.0)
fgan2 = FGANLearningObjective(gen2, disc2, "pearson", gamma=1.0)
fgan = fgan.to(device)
fgan2 = fgan2.to(device)
batchsize = 64
#batchsize = 128
optimizer_gen = optim.Adam(fgan.gen.parameters(), lr=lr_select_gen)
optimizer_disc = optim.Adam(fgan.disc.parameters(), lr=lr_select_disc)
# Using a scheduler, i.e. “scheduler = optim.lr_scheduler.MultiStepLR(optimizer_gen” and “scheduler.step()” is recommended.
niter = 50000
#niter = 100000
xreal = Variable(torch.Tensor(np.reshape(tdist.rvs(4096), (4096,1))), requires_grad=True)
xreal = torch.cat((xreal, Variable(torch.Tensor(np.reshape(tdist.rvs(4096), (4096,1))), requires_grad=True)), dim=1)
#checkpoint = torch.load('./.pt')
#fgan2.gen.load_state_dict(checkpoint['gen_model_state_dict'])
#fgan2.gen.eval()
#fgan2.disc.eval()
#for param in fgan2.gen.parameters():
#    param.requires_grad = False
#for param in fgan2.disc.parameters():
#    param.requires_grad = False
xreal = torch.cat((torch.cat((Variable(torch.Tensor(np.reshape(tdist.rvs(4096 // 2),
                                                               (4096 // 2, 1))), requires_grad=True),
                              Variable(torch.Tensor(np.reshape(tdist.rvs(4096 // 2),
                                                               (4096 // 2, 1))), requires_grad=True)), dim=1),
                   torch.cat((Variable(torch.Tensor(np.reshape(tdist2.rvs(4096 // 2),
                                                               (4096 // 2, 1))), requires_grad=True),
                              Variable(torch.Tensor(np.reshape(tdist2.rvs(4096 // 2),
                                                               (4096 // 2, 1))), requires_grad=True)), dim=1)), dim=0)
xreal = Variable(xreal, requires_grad=True)
xreal = xreal.to(device)
#checkpoint = torch.load('./.pt')
#fgan.gen.load_state_dict(checkpoint['gen_state_dict'])
#optimizer_gen.load_state_dict(checkpoint['gen_opt_state_dict'])
fgan.gen.train()
fgan.disc.train()
for i in range(niter):
    fgan.zero_grad()
    zmodel = Variable(torch.rand((batchsize, 2), device=device))
    loss_gen, loss_disc = fgan(xreal, zmodel, mu_select, ni_select)
    if i >= 10000:
       if i % 1000:
           torch.save({'gen_state_dict': fgan.gen.state_dict(),
                       'disc_state_dict': fgan.disc.state_dict(),
                       'gen_opt_state_dict': optimizer_gen.state_dict(),
                       'disc_opt_state_dict': optimizer_disc.state_dict()},
                      './.pt')
    if i % 1000 == 1:
        print("iter %d  obj(G) %.4f" % (i, loss_gen))
        # Example 1:
        # iter 1  obj(G) 19.8317
        # iter 1001  obj(G) 12.7071
        # iter 2001  obj(G) 10.1078
        # iter 3001  obj(G) 8.3397
        # iter 4001  obj(G) 6.4563
        # iter 5001  obj(G) 5.0766
        # iter 6001  obj(G) 4.4117
        # iter 7001  obj(G) 3.7541
        # iter 8001  obj(G) 3.5934
        # Example 2:
        # iter 1  obj(G) 3874.3281
        # iter 1001  obj(G) 847.9266
        # iter 2001  obj(G) 491.9786
        # iter 3001  obj(G) 404.2814
        # iter 4001  obj(G) 302.7887
        # iter 5001  obj(G) 239.9177
        # iter 6001  obj(G) 213.9305
        # iter 7001  obj(G) 192.0002
        # iter 8001  obj(G) 146.1595
        #print("iter %d  obj(D) %.4f  obj(G) %.4f" % (i, loss_disc, loss_gen))
        n = 4096
        allData = xreal.cpu().detach()
        XmoModel = fgan.gen(Variable(torch.rand((batchsize, 2), device=device), requires_grad=False)).detach().cpu().numpy()
        plt.figure()
        plt.plot(allData[:, 0].numpy(), allData[:, 1].numpy(), 'bo', label='Real data')
        plt.plot(XmoModel[:,0], XmoModel[:,1], 'gx', label='Boundary data')
        plt.legend()
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.savefig('./' + str(int(i)) + '.png', dpi=500)
        plt.close()
    fgan.gen.zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizer_gen.step()
    fgan.disc.zero_grad()
    loss_disc.backward()
    optimizer_disc.step()
torch.save({'gen_state_dict': fgan.gen.state_dict(),
                       'disc_state_dict': fgan.disc.state_dict(),
                       'gen_opt_state_dict': optimizer_gen.state_dict(),
                       'disc_opt_state_dict': optimizer_disc.state_dict()}, './.pt')
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
# Create dynamics by pushing the generated samples OoD: Likelihood-free boundary of data distribution
# # Candidate models
# # Different architectures
# # Model architectures
# # Hyper-parameters: Width and depth
# # Hyper-parameter: Model architecture
# # (A) Feed-forward network
# class Generator(nn.Module):
#     def __init__(self, nhidden):
#         super(Generator, self).__init__()
#         self.lin1 = nn.Linear(2, nhidden)
#         self.lin2 = nn.Linear(nhidden, 2)
#     def forward(self, z):
#         h = F.relu(self.lin1(z))
#         x = self.lin2(h)
#         return x
# # (B) Residual network
# class Generator(nn.Module):
#     def __init__(self, nhidden):
#         super(Generator, self).__init__()
#         self.lin1 = nn.Linear(2, nhidden)
#         self.lin2 = nn.Linear(nhidden, 2)
#     def forward(self, z):
#         h = F.relu(self.lin1(z))
#         x = self.lin2(h)
#         x2 = x + z
#         return x2
# # (C) Feed-forward with Batch Normalization
# class Generator(nn.Module):
#     def __init__(self, nhidden):
#         super(Generator, self).__init__()
#         self.lin1 = nn.Linear(2, nhidden)
#         self.lin1bn = nn.BatchNorm1d(nhidden)
#         self.lin2 = nn.Linear(nhidden, 2)
#     def forward(self, z):
#         h = F.relu(self.lin1bn(self.lin1(z)))
#         x = self.lin2(h)
#         return x
# # (D) Residual with Batch Normalization
# class Generator(nn.Module):
#     def __init__(self, nhidden):
#         super(Generator, self).__init__()
#         self.lin1 = nn.Linear(2, nhidden)
#         self.lin1bn = nn.BatchNorm1d(nhidden)
#         self.lin2 = nn.Linear(nhidden, 2)
#     def forward(self, z):
#         h = F.relu(self.lin1bn(self.lin1(z)))
#         x = self.lin2(h)
#         x2 = x + z
#         return x2
# # (E) Feed-forward with weight initialization and Batch Normalization
# class Generator(nn.Module):
#     def __init__(self, nhidden):
#         super(Generator, self).__init__()
#         self.lin1 = nn.Linear(2, nhidden)
#         init.xavier_uniform_(self.lin1.weight, gain=0.1)
#         self.lin1bn = nn.BatchNorm1d(nhidden)
#         self.lin2 = nn.Linear(nhidden, 2)
#     def forward(self, z):
#         h = F.relu(self.lin1bn(self.lin1(z)))
#         x = self.lin2(h)
#         return x
# # (F) Residual with weight initialization and Batch Normalization
# class Generator(nn.Module):
#     def __init__(self, nhidden):
#         super(Generator, self).__init__()
#         self.lin1 = nn.Linear(2, nhidden)
#         init.xavier_uniform_(self.lin1.weight, gain=0.1)
#         self.lin1bn = nn.BatchNorm1d(nhidden)
#         self.lin2 = nn.Linear(nhidden, 2)
#     def forward(self, z):
#         h = F.relu(self.lin1bn(self.lin1(z)))
#         x = self.lin2(h)
#         x2 = x + z
#         return x2
# # (G)
# #F.relu(
# #F.leaky_relu(
# #F.tanh(
# #F.elu(
# We achieve better boundary formation results than (https://arxiv.org/pdf/1711.09325.pdf) in Figure 3.
# The use of torch.nn.DataParallel(model) is recommended along with the
# use of torch.save(model.module.state_dict(), "./.pt") instead of
# torch.save(model.state_dict(), "./.pt"). Also, saving the best model is
# recommended by using "best_loss = float('inf')" and "if loss.item()<best_loss:
# best_loss=loss.item(); torch.save(model.module.state_dict(), "./.pt")".
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
