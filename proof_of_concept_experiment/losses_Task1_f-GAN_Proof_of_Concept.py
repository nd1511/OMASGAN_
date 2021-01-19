import torch
import torch.nn as nn
import torch.nn.functional as F
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
            raise ValueError("This is an unknown f-divergence.")
class FGANLearningObjective(nn.Module):
    def __init__(self, gen, disc, divergence_name="gan", gamma=0.01):
        super(FGANLearningObjective, self).__init__()
        self.gen = gen
        self.disc = disc
        self.conj = ConjugateDualFunction(divergence_name)
        self.gammahalf = 0.5*gamma
    def forward(self, xreal, zmodel):
        vreal = self.disc(xreal)
        Treal = self.conj.T(vreal)
        xmodel = self.gen(zmodel)
        vmodel = self.disc(xmodel)
        fstar_Tmodel = self.conj.fstarT(vmodel)
        loss_gen = -fstar_Tmodel.mean()
        loss_disc = fstar_Tmodel.mean() - Treal.mean()
        # Gradient penalty
        if self.gammahalf > 0.0:
            batchsize = xreal.size(0)
            grad_pd = torch.autograd.grad(Treal.sum(), xreal,
                create_graph=True, only_inputs=True)[0]
            grad_pd_norm2 = grad_pd.pow(2)
            grad_pd_norm2 = grad_pd_norm2.view(batchsize, -1).sum(1)
            gradient_penalty = self.gammahalf * grad_pd_norm2.mean()
            loss_disc += gradient_penalty
        return loss_gen, loss_disc
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
