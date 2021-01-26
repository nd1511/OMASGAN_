import torch
import torch.nn.functional as F
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
    dis_fake_m2 = dis_fake2 / temp
    dis_fake_ratio2 = get_kl_ratio(dis_fake_m2)
    dis_fake2 = dis_fake2 * dis_fake_ratio2
    loss_disc = torch.mean(F.relu(1. + dis_fake)) + 0.7 * torch.mean(F.relu(1. + dis_fake2)) + 0.7 * torch.mean(F.relu(1. - dis_real))
    return loss_disc
def loss_kl_gen(dis_fake, dis_fake2, dis_real, dis_real2, temp=1.0):
    dis_fake_m = dis_fake / temp
    dis_fake_ratio = get_kl_ratio(dis_fake_m)
    dis_fake = dis_fake * dis_fake_ratio
    loss_gen = -torch.mean(dis_fake)
    return loss_gen
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
