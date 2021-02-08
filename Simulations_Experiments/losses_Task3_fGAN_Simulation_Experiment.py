# For all architectures: For f-GAN-based OMASGAN and for KLWGAN-based OMASGAN
import torch
import torch.nn as nn
import torch.nn.functional as F
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
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
            return 0.25 * v * v + v
        elif self.divergence_name == "neyman":
            return 2.0 - 2.0 * F.exp(0.5 * v)
        elif self.divergence_name == "hellinger":
            return F.exp(-v) - 1.0
        elif self.divergence_name == "jensen":
            return F.softplus(v) - math.log(2.0)
        elif self.divergence_name == "gan":
            return F.softplus(v)
        else:
            raise ValueError("Unknown f-divergence.")
# Original GAN:
# Optimization: min_G max_D E_x log(D(x)) + E_z log(1 - D(G(z)))
# L_G = E_z log(1 - D(G(z)))
# L_D = -E_x log(D(x)) - E_z log(1 - D(G(z)))
# vreal = self.disc(xreal)
# Treal = self.conj.T(vreal)
# xmodel = self.gen(zmodel)
# vmodel = self.disc(xmodel)
# fstar_Tmodel = self.conj.fstarT(vmodel)
# loss_gen = -fstar_Tmodel.mean()
# loss_disc = fstar_Tmodel.mean() - Treal.mean()
# In order of appearance, we use the random variables x, G(z), B(z), and G’(z) where x~p_x(x) and G(z)~p_g(x).
class FGANLearningObjective(nn.Module):
    def __init__(self, gen, disc, divergence_name="pearson", gamma=10.0):
        super(FGANLearningObjective, self).__init__()
        self.gen = gen
        self.disc = disc
        self.conj = ConjugateDualFunction(divergence_name)
        self.gammahalf = 0.5 * gamma
    #def forward(self, xreal, zmodel, xreal2, xreal3, alpha=0.3, beta=1.0, gamma=0.7, delta=0.4, iota=0.1):
    def forward(self, xreal, zmodel, xreal2, xreal3, alpha=0.3, beta=1.0, gamma=0.7, delta=0.4):
        xmodel = self.gen(zmodel)
        # xmodel is G'(z)
        vmodel = self.disc(xmodel)
        fstar_Tmodel = self.conj.fstarT(vmodel)
        # xreal2 is B(z)
        vmodel2 = self.disc(xreal2)
        fstar_Tmodel2 = self.conj.fstarT(vmodel2)
        # xreal3 is G(z)
        vreal2 = self.disc(xreal3)
        Treal2 = self.conj.T(vreal2)
        # xreal is x
        vreal62 = self.disc(xreal)
        Treal62 = self.conj.T(vreal62)
        #D = torch.norm(zmodel[None, :].expand(zmodel.shape[0], -1, -1) - zmodel[:, None], dim=-1) / (1e-17 + torch.norm(xmodel[None, :].expand(xmodel.shape[0], -1, -1) - xmodel[:, None], dim=-1))
        #loss_gen = -beta * fstar_Tmodel.mean() - gamma * fstar_Tmodel2.mean() + delta * Treal2.mean() + alpha * Treal62.mean() + iota * torch.mean(D, dim=1)[0].mean()
        # alpha+delta=0.7, beta=1, and gamma=0.7
        loss_gen = -beta * fstar_Tmodel.mean() - gamma * fstar_Tmodel2.mean() + delta * Treal2.mean() + alpha * Treal62.mean()
        loss_disc = beta * fstar_Tmodel.mean() + gamma * fstar_Tmodel2.mean() - delta * Treal2.mean() - alpha * Treal62.mean()
        # Gradient penalty
        if self.gammahalf > 0.0:
            batchsize = xreal.size(0)
            grad_pd = torch.autograd.grad(Treal62.sum(), xreal, create_graph=True, only_inputs=True)[0]
            grad_pd_norm2 = grad_pd.pow(2)
            grad_pd_norm2 = grad_pd_norm2.view(batchsize, -1).sum(1)
            gradient_penalty = self.gammahalf * grad_pd_norm2.mean()
            loss_disc += gradient_penalty
        return loss_gen, loss_disc
# Optimization of original GAN: min_G max_D E_x log(D(x)) + E_z log(1 - D(G(z)))
# L_G = E_z log(1 - D(G(z))) and L_D = -E_x log(D(x)) - E_z log(1 - D(G(z)))
# loss_gen = -fstar_Tmodel.mean() # Generator loss (minimize)
# loss_disc = fstar_Tmodel.mean() - Treal.mean() # Discriminator loss (minimize)
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
