import torch
import torch.nn.functional as F
# Acknowledgements: Thanks to the repositories: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
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
def loss_hinge_analysis(dis_real):
    weighted = F.relu(1. - dis_real)
    loss_real = weighted
    return loss_real
def loss_hinge_dis_new(dis_fake, dis_real, dis_real_fake):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    loss_real_fake = torch.mean(F.relu(1. + dis_real_fake))
    return loss_real, loss_fake, loss_real_fake
def loss_hinge_dis_new_fake(dis_fake, dis_real, dis_real_fake, dis_fake_fake):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    loss_real_fake = torch.mean(F.relu(1. + dis_real_fake))
    loss_fake_fake = torch.mean(F.relu(1. + dis_fake_fake))
    return loss_real, loss_fake, loss_real_fake, loss_fake_fake
def get_kl_ratio(v):
    v_norm = torch.logsumexp(v[:, 0], dim=0) - torch.log(torch.tensor(v.size(0)).float())
    return torch.exp(v - v_norm)
def loss_kl_dis(dis_fake, dis_real, dis_fake2, dis_real2, temp=1.0, zeta=0.6):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    dis_fake_m2 = dis_fake2 / temp
    dis_fake_ratio2 = get_kl_ratio(dis_fake_m2)
    dis_fake2 = dis_fake2 * dis_fake_ratio2
    loss_disc = zeta * torch.mean(F.relu(1. + dis_real)) + (1-zeta) * torch.mean(F.relu(1. + dis_fake)) + torch.mean(F.relu(1. - dis_fake2))
    return loss_disc
def loss_kl_gen(dis_fake, dis_fake2, dis_real, dis_real2, temp=1.0):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    loss_gen = -torch.mean(dis_fake)
    return loss_gen
# def loss_kl_dis(dis_fake, dis_real):
#     loss_real = torch.mean(F.relu(1. - dis_real))
#     with torch.no_grad():
#         dis_fake_m = dis_fake - dis_fake.mean()
#         dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
#         dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
#         dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
#     dis_fake = dis_fake * dis_fake_ratio
#     loss_fake = torch.mean(F.relu(1. + dis_fake))
#     return loss_real, loss_fake
# def loss_kl_gen(dis_fake):
#     with torch.no_grad():
#         dis_fake_m = dis_fake - dis_fake.mean()
#         dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
#         dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
#         dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
#     dis_fake = dis_fake * dis_fake_ratio
#     loss = -torch.mean(dis_fake)
#     return loss
def loss_kl_dis_new(dis_fake, dis_real, dis_real_fake):
    loss_real = torch.mean(F.relu(1. - dis_real))
    with torch.no_grad():
        dis_fake_m = dis_fake - dis_fake.mean()
        dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
        dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
        dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    with torch.no_grad():
        dis_fake_m = dis_real_fake - dis_real_fake.mean()
        dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
        dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
        dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_real_fake = dis_real_fake * dis_fake_ratio
    loss_real_fake = torch.mean(F.relu(1. + dis_real_fake))
    return loss_real, loss_fake, loss_real_fake
def loss_kl_grad_dis(dis_fake, dis_real):
    dis_fake_m = dis_fake - dis_fake.mean()
    dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
    dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
def loss_kl_grad_gen(dis_fake):
    dis_fake_m = dis_fake - dis_fake.mean()
    dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
    dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss
def loss_f_kl_dis(dis_fake, dis_real):
    import ipdb
    ipdb.set_trace()
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    loss_fake = torch.mean(torch.exp(dis_fake - 1.0))
    return loss_real, loss_fake
def loss_f_kl_gen(dis_fake):
    import ipdb
    ipdb.set_trace()
    loss = -torch.mean(torch.exp(dis_fake - 1.0))
    return loss
def loss_dv_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    loss_fake = -torch.logsumexp(dis_fake) / dis_fake.size(0)
    return loss_real, loss_fake
def loss_dv_gen(dis_fake):
    loss = torch.logsumexp(dis_fake) / dis_fake.size(0)
    return loss
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
