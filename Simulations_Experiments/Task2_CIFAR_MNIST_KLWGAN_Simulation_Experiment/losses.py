from __future__ import print_function
# Acknowledgements: Thanks to the repositories: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
import torch
import torch.nn.functional as F
#mu_select = mu_select
#mu_select = 0.2
#ni_select = ni_select
#ni_select = 0.3
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2
def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss
def loss_dcgan_dis_new(dis_fake, dis_real, dis_real_fake):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    L_real_fake = torch.mean(F.softplus(dis_real_fake))
    return L1, L2, L_real_fake
def loss_hinge_dis(dis_fake, dis_real):
    weighted = F.relu(1. - dis_real)
    loss_real = torch.mean(weighted)
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss
def get_kl_ratio(v):
    v_norm = torch.logsumexp(v[:, 0], dim=0) - torch.log(torch.tensor(v.size(0)).float())
    return torch.exp(v - v_norm)
def loss_kl_dis(dis_fake, dis_real, dis_fake2, dis_real2, temp=1.0):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    #loss_disc = torch.mean(F.relu(1. + dis_fake)) + torch.mean(F.relu(1. - dis_fake2))
    #loss_disc = torch.mean(F.relu(1. + dis_fake)) + torch.mean(F.relu(1. - dis_real)) + torch.mean(F.relu(1. - dis_fake2))
    loss_disc = torch.mean(F.relu(1. + dis_fake)) + torch.mean(F.relu(1. - dis_real))
    return loss_disc
def loss_kl_gen(dis_fake, dis_fake2, dis_real, dis_real2, xreal, zmodel, xmodel, temp=1.0, mu=20, ni=50):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    second_term_loss = torch.min(torch.norm(xreal.view(-1, 3 * 32 * 32)[None, :].expand(xmodel.shape[0], -1, -1) - xmodel.view(-1, 3 * 32 * 32)[:, None], dim=-1), dim=1)[0].mean()
    third_term_loss = torch.mean(torch.norm(zmodel[None, :].expand(zmodel.shape[0], -1, -1) - zmodel[:, None], dim=-1) / (1e-17 + torch.norm(xmodel.view(-1, 3 * 32 * 32)[None, :].expand(xmodel.shape[0], -1, -1) - xmodel.view(-1, 3 * 32 * 32)[:, None], dim=-1)), dim=1)[0].mean()
    loss_gen = torch.mean(dis_fake) + mu * second_term_loss2 + ni * third_term_loss
    #loss_gen = first_term_loss + hyperparameter1*second_term_loss + hyperparameter2*third_term_loss
    # The first term in the loss function is a strictly decreasing function of a distribution metric.
    # Create dynamics by pushing the generated samples OoD: Likelihood-free boundary of data distribution
    return loss_gen
# The first two terms of the proposed objective cost function, i.e. $- m( B, G)$ and $d( B, G )$,
# are convex as functions of the underlying probability measures. Because of the neural architectures
# used, the optimization problem is non-convex. Regarding the choice of loss function for the boundary
# Task, we create dynamics by pushing the generated samples OoD. Taking into account results from
# optimization theory and optimization results for non-linear architectures, we introduce a regularized
# cost function in the optimization instead of attempting to solve a much harder constrained optimization.
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2
def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss
def loss_dcgan_dis_new(dis_fake, dis_real, dis_real_fake):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    L_real_fake = torch.mean(F.softplus(dis_real_fake))
    return L1, L2, L_real_fake
def loss_hinge_dis(dis_fake, dis_real):
    weighted = F.relu(1. - dis_real)
    loss_real = torch.mean(weighted)
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss
def get_kl_ratio(v):
    v_norm = torch.logsumexp(v[:, 0], dim=0) - torch.log(torch.tensor(v.size(0)).float())
    return torch.exp(v - v_norm)
def loss_kl_dis(dis_fake, dis_real, temp=1.0):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
def loss_kl_gen(dis_fake, xreal, zmodel, xmodel, temp=1.0, mu=20, ni=50):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    loss = torch.mean(dis_fake)
    second_term_loss = torch.min(torch.norm(xreal.view(-1, 3 * 32 * 32)[None, :].expand(xmodel.shape[0], -1, -1) - xmodel.view(-1, 3 * 32 * 32)[:, None], dim=-1), dim=1)[0].mean()
    third_term_loss = torch.mean(torch.norm(zmodel[None, :].expand(zmodel.shape[0], -1, -1) - zmodel[:, None], dim=-1) / (1e-17 + torch.norm(xmodel.view(-1, 3 * 32 * 32)[None, :].expand(xmodel.shape[0], -1, -1) - xmodel.view(-1, 3 * 32 * 32)[:, None], dim=-1)), dim=1)[0].mean()
    #loss_gen = first_term_loss + hyperparameter1*second_term_loss + hyperparameter2*third_term_loss
    loss_gen = loss + mu * second_term_loss + ni * third_term_loss
    # The first term in the loss function is a strictly decreasing function of a distribution metric.
    # We create dynamics by pushing the generated samples OoD: Likelihood-free boundary of data distribution
    return loss_gen
# Pearson Chi-Squared: According to Table 4 of the f-GAN paper, we use the
# Pearson Chi-Squared f-divergence distribution metric and we note that after Pearson
# Chi-Squared, the next best are KL and then Jensen-Shannon (Nowozin et al., 2016).
def loss_chi_dis(dis_fake, dis_real):
    dis_fake = torch.clamp(dis_fake, -1.0, 1.0)
    dis_real = torch.clamp(dis_real, -1.0, 1.0)
    loss_real = torch.mean(- dis_real)
    dis_fake_mean = torch.mean(dis_fake)
    loss_fake = torch.mean(dis_fake * (dis_fake - dis_fake_mean + 2)) / 2.0
    return loss_real, loss_fake
def loss_chi_gen(dis_fake):
    dis_fake = torch.clamp(dis_fake, -1.0, 1.0)
    dis_fake_mean = torch.mean(dis_fake)
    loss_fake = -torch.mean(dis_fake * (dis_fake - dis_fake_mean + 2)) / 2.0
    return loss_fake
def loss_dv_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    dis_fake_norm = torch.exp(dis_fake).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1. + dis_fake)) + torch.mean(dis_fake_ratio * torch.log(dis_fake_ratio))
    return loss_real, loss_fake
def loss_dv_gen(dis_fake):
    dis_fake_norm = torch.exp(dis_fake).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake) - torch.mean(dis_fake_ratio * torch.log(dis_fake_ratio))
    return loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
# Acknowledgements: Thanks to the repositories: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
