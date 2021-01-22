from __future__ import print_function
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
# Additional acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs)
from datasets_Task1_fGAN_Simulation_Experiment import *
from networks_Task1_fGAN_Simulation_Experiment import *
from losses_Task2_fGAN_Simulation_Experiment import *
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# We use the leave-one-out (LOO) evaluation methodology.
# The LOO  methodology is setting K classes of a dataset with (K + 1) classes
# as the normal class and the leave-out class as the abnormal class.
#abnormal_class_LOO = abnormal_class_LOO
abnormal_class_LOO = 0
#abnormal_class_LOO = 1
lr_select = lr_select
#lr_select = 1.0e-3
lr_select_gen = lr_select
lr_select_disc = lr_select
mu_select = mu_select
#mu_select = 0.2
ni_select = ni_select
#ni_select = 0.3
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
from torchvision.utils import save_image
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
optimizer_gen = optim.Adam(fgan.gen.parameters(), lr=lr_select_gen)
optimizer_disc = optim.Adam(fgan.disc.parameters(), lr=lr_select_disc)
data_forTrainloader = choose_dataset(select_dataset)
from torch.utils.data import Subset
def get_target_label_idx(labels, targets):
  return np.argwhere(np.isin(labels, targets)).flatten().tolist()
train_idx_normal = get_target_label_idx(data_forTrainloader.targets, np.delete(np.array(list(range(0, 10))), abnormal_class_LOO))
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, np.delete(np.array(list(range(0, 10))), 0))
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, np.delete(np.array(list(range(0, 10))), 1))
# Example 1:
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, np.delete(np.array(list(range(0, 10))), 0))
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, [1, 2, 3, 4, 5, 6, 7, 8, 9])
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, list(range(1, 10)))
# Example 2:
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, np.delete(np.array(list(range(0, 10))), 1))
#train_idx_normal = get_target_label_idx(data_forTrainloader.targets, [0, 2, 3, 4, 5, 6, 7, 8, 9])
# Use the leave-one-out (LOO) evaluation methodology.
# The LOO evaluation methodology is setting K classes of a dataset with (K + 1)
# classes as the normal class and the leave-out class as the abnormal class.
data_forTrainloader = Subset(data_forTrainloader, train_idx_normal)
print(len(data_forTrainloader))
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
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
            raise ValueError("Unknown f-divergence name.")
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
            raise ValueError("Unknown f-divergence name.")
ngpu = 1
nz = 128
nc = 1
nrand = nz
ngf = 64
ndf = 64
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class Generator(nn.Module):
    def __init__(self, nrand):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(nrand, 4*4*512)
        init.xavier_uniform_(self.lin1.weight, gain=0.1)
        self.lin1bn = nn.BatchNorm1d(4*4*512)
        self.dc1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dc1bn = nn.BatchNorm2d(256)
        self.dc2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc2bn = nn.BatchNorm2d(128)
        self.dc3a = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc3abn = nn.BatchNorm2d(64)
        self.dc3b = nn.Conv2d(64, 1, 3, stride=1, padding=1)
    def forward(self, z):
        h = F.relu(self.lin1bn(self.lin1(z)))
        h = torch.reshape(h, (-1, 512, 4, 4))
        h = F.relu(self.dc1bn(self.dc1(h)))
        h = F.relu(self.dc2bn(self.dc2(h)))
        h = F.relu(self.dc3abn(self.dc3a(h)))
        x = self.dc3b(h)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
gen = Generator(nrand).to(device)
gen2 = Generator(nrand).to(device)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
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
disc = Discriminator().to(device)
disc2 = Discriminator().to(device)
class FGANLearningObjective(nn.Module):
    def __init__(self, gen, disc, divergence_name="pearson", gamma=0.01):
        super(FGANLearningObjective, self).__init__()
        self.gen = gen
        self.disc = disc
        self.conj = ConjugateDualFunction(divergence_name)
        self.gammahalf = 0.5 * gamma
    def forward(self, xreal, zmodel):
        vreal = self.disc(xreal)
        Treal = self.conj.T(vreal)
        xmodel = self.gen(zmodel)
        vmodel = self.disc(xmodel)
        fstar_Tmodel = self.conj.fstarT(vmodel)
        second_term_loss = torch.min(torch.norm(
                xreal.view(-1, 32 * 32)[None, :].expand(xmodel.shape[0], -1, -1) - xmodel.view(-1, 32 * 32)[:, None],
                dim=-1), dim=1)[0].mean()
        third_term_loss = torch.mean(torch.norm(zmodel[None, :].expand(zmodel.shape[0], -1, -1) - zmodel[:, None], dim=-1) / (1e-17 + torch.norm(
                xmodel.view(-1, 32*32)[None, :].expand(xmodel.shape[0], -1, -1) - xmodel.view(-1, 32*32)[:, None], dim=-1)), dim=1)[0].mean()
        loss_gen = fstar_Tmodel.mean() + third_term_loss + second_term_loss
        loss_disc = fstar_Tmodel.mean() - Treal.mean()
        if self.gammahalf > 0.0:
            batchsize = xreal.size(0)
            grad_pd = torch.autograd.grad(Treal.sum(), xreal, create_graph=True, only_inputs=True)[0]
            grad_pd_norm2 = grad_pd.pow(2)
            grad_pd_norm2 = grad_pd_norm2.view(batchsize, -1).sum(1)
            gradient_penalty = self.gammahalf * grad_pd_norm2.mean()
            loss_disc += gradient_penalty
        return loss_gen, loss_disc, fstar_Tmodel.mean(), second_term_loss, third_term_loss
# According to Table 4 of the f-GAN paper, we use Pearson Chi-Squared.
# After Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.
fgan = FGANLearningObjective(gen, disc, "pearson", gamma=10.0)
fgan2 = FGANLearningObjective(gen2, disc2, "pearson", gamma=10.0)
fgan = fgan.to(device)
fgan2 = fgan2.to(device)
batchsize = 64
optimizer_gen = optim.Adam(fgan.gen.parameters(), lr=1.0e-3)
optimizer_disc = optim.Adam(fgan.disc.parameters(), lr=1.0e-3)
trainloader = torch.utils.data.DataLoader(data_forTrainloader, batch_size=len(data_forTrainloader), shuffle=True)
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
def visualize(epoch, model, itr, real_imgs):
    model.eval()
    makedirs(os.path.join('exper', 'imgs'))
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs
    nvals = 256
    with torch.no_grad():
        fake_imgs = model(Variable(torch.rand((batchsize, nrand), device=device)))
        fake_imgs = fake_imgs.view(-1, 1, 32, 32)
        imgs = torch.cat([_real_imgs, fake_imgs], 0)
        filename = os.path.join('exper', 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        print(filename)
        save_image(imgs.cpu().float(), filename, nrow=16, padding=2)
    model.train()
writer = SummaryWriter(log_dir="runs/CIFAR10", comment="f-GAN-Pearson")
niter = 0
nepochs = 1000
# In Task 2, the boundary model is trained to perform sample generation on the boundary
# of the data distribution by starting from within the data distribution (Task 1).
checkpoint = torch.load('./.pt')
fgan2.gen.load_state_dict(checkpoint['gen_state_dict'])
fgan2.disc.load_state_dict(checkpoint['disc_state_dict'])
fgan2.gen.eval()
fgan2.disc.eval()
fgan2.eval()
# In Task 2, the boundary model is trained to perform sample generation on the
# boundary of the data distribution by starting from within the data distribution.
checkpoint = torch.load('./.pt')
fgan.gen.load_state_dict(checkpoint['gen_state_dict'])
fgan.disc.load_state_dict(checkpoint['disc_state_dict'])
fgan.gen.train()
fgan.disc.train()
fgan.train()
for epoch in range(nepochs):
    zmodel = Variable(torch.rand((batchsize, nrand), device=device))
    xmodel = fgan.gen(zmodel)
    xmodelimg = vutils.make_grid(xmodel, normalize=True, scale_each=True)
    writer.add_image('Generated', xmodelimg, global_step=niter)
    for i, data in enumerate(trainloader, 0):
        niter += 1
        imgs, _ = data
        fgan.zero_grad()
        zmodel = Variable(torch.rand((batchsize, nrand), device=device))
        if i == 0 and epoch == 0:
            xreal = Variable(fgan2.gen(Variable(torch.rand((imgs.shape[0], nrand), device=device))), requires_grad=True)
        loss_gen, loss_disc, fiTe, seTe, thTe = fgan(xreal, zmodel)
        writer.add_scalar('obj/disc', loss_disc, niter)
        writer.add_scalar('obj/gen', loss_gen, niter)
        if i == 0:
            print("%.4f, %.4f" % (loss_disc, loss_gen))
        fgan.gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizer_gen.step()
        fgan.disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
    if epoch % 10 == 0:
        visualize(epoch, fgan.gen, i, xreal)
    if epoch >= 100:
        if epoch % 100 == 0:
            torch.save({'gen_state_dict': fgan.gen.state_dict(),
                        'disc_state_dict': fgan.disc.state_dict(),
                        'gen_opt_state_dict': optimizer_gen.state_dict(),
                        'disc_opt_state_dict': optimizer_disc.state_dict()},
                       './/.pt')
torch.save({'gen_state_dict': fgan.gen.state_dict(),
                        'disc_state_dict': fgan.disc.state_dict(),
                        'gen_opt_state_dict': optimizer_gen.state_dict(),
                        'disc_opt_state_dict': optimizer_disc.state_dict()},
                       './/.pt')
writer.export_scalars_to_json("./allscalars.json")
writer.close()
# Example:
# 26.7841, 66.1683
# 20.7029, 60.3265
# 17.0790, 55.5102
# 11.6779, 50.5878
# 8.7261, 48.6052
# (...)
# -169.2656, 17.2085
# -177.1498, 15.1311
# -182.0917, 14.9632
# -184.8488, 12.2080

# Example:
# 29.9916, 56.3684
# exper/imgs7/e000_i000000.png
# 29.0536, 55.9183
# 21.6877, 48.6218
# 17.3165, 44.3356
# 18.9488, 43.5583
# 13.6081, 40.3400
# 12.2382, 38.0850

# Example:
# 32.5157, 55.8173
# exper/imgs7/e000_i000000.png
# 27.6867, 53.2506
# 27.7261, 53.3988
# 20.2781, 45.6117
# 15.4402, 40.5108
# 15.5646, 42.1485
# 12.6256, 41.2110
# 11.6283, 40.8528
# 10.9556, 39.1046

# Example:
# 28.8066, 56.1958
# exper/imgs7/e000_i000000.png
# 27.5902, 54.9074
# 23.8903, 51.2608
# 24.2685, 48.2836
# 18.9200, 44.9000
# 14.1086, 41.4744
# 13.8300, 41.1561
# 11.4531, 37.4275

# Example 0:
# 4.9908, 43.1165
# exper/imgs7/e000_i000000.png
# 4.5096, 42.5819
# 4.2306, 41.3680
# 4.2305, 41.1702
# 4.0586, 40.4735
# 3.4283, 37.4796

# Example 1:
# ssh://ndioneli@localhost:9000/home/ndioneli/miniconda3/bin/python3 -u /remote/rds/users/ndioneli/pycharm_project_233/BMLOO2_All.py
# /home/ndioneli/miniconda3/lib/python3.7/site-packages/torchvision/datasets/mnist.py:43: UserWarning: train_labels has been renamed targets
#   warnings.warn("train_labels has been renamed targets")
# cuda:0
# 6.4958, 45.8226
# exper/imgs7/e000_i000000.png
# 4.7186, 42.9613
# 4.8987, 42.6020
# 3.7021, 40.3955
# 3.4563, 40.2787
# 2.7054, 39.5376
# 2.9118, 40.5526
# 1.4779, 37.7500
# 2.1383, 39.5404
# 1.6033, 38.9066
# 0.9880, 38.4334
# exper/imgs7/e010_i000000.png
# 0.6265, 38.7089
# 1.0062, 38.8536
# 0.5869, 37.2278

# Example 2:
# ssh://ndioneli@localhost:9000/home/ndioneli/miniconda3/bin/python3 -u /remote/rds/users/ndioneli/pycharm_project_233/BMLOO2_All.py
# /home/ndioneli/miniconda3/lib/python3.7/site-packages/torchvision/datasets/mnist.py:43: UserWarning: train_labels has been renamed targets
#   warnings.warn("train_labels has been renamed targets")
# cuda:0
# 6.4066, 47.2562
# exper/imgs7/e000_i000000.png
# 4.6828, 43.7559
# 5.0689, 41.8758
# 3.3451, 41.9767
# 3.0634, 40.9313
# 2.8388, 40.4378
# 2.3470, 39.4283
# 1.3560, 38.6752
# 1.6963, 37.7582
# 0.9331, 38.7327
# 0.8326, 38.2803
# exper/imgs7/e010_i000000.png
# 0.8202, 38.5915
# 0.7315, 39.0685
# 0.7854, 38.6682
# 0.8372, 37.0155
# 0.7421, 40.2134
# 0.8016, 38.4790
# 0.4646, 38.5223
# 0.5654, 37.4845
# 0.5443, 37.3409
# 0.4212, 38.2105
# exper/imgs7/e020_i000000.png
# 0.4301, 38.2498
# 0.2450, 37.1797
# 0.3922, 36.6731
# 0.3671, 35.6314
# 0.3631, 36.3516
# 0.8329, 33.7773
# 0.6585, 36.3104
# 0.7235, 33.7991
# 0.6634, 34.9263
# 1.1327, 33.5715
# exper/imgs7/e030_i000000.png
# 0.4749, 35.3665
# 0.5732, 34.8143
# 0.3835, 35.3011
# 0.6525, 35.6588
# 0.3631, 36.5546
# 0.3634, 32.3212
# 0.1606, 35.8843
# 0.1630, 33.4293
# 0.2298, 36.0646
# 0.1663, 34.7547
# exper/imgs7/e040_i000000.png
# 0.1097, 33.5802
# 0.1336, 33.3643
# 0.2867, 32.9005
# 0.1678, 33.6845
# 0.2023, 32.0235
# 0.1654, 32.9960
# 0.1597, 34.0442
# 0.1242, 34.7588
# 0.1058, 34.3207
# 0.2394, 34.4645
# exper/imgs7/e050_i000000.png
# 0.1372, 32.6773
# 0.1268, 33.0378
# 0.0896, 34.8408
# 0.1172, 31.9905
# 0.1180, 31.9080
# 0.1047, 32.4412
# 0.0960, 32.8195
# 0.1157, 34.0473
# 0.1102, 32.0267
# 0.0812, 32.6068
# exper/imgs7/e060_i000000.png
# 0.1609, 30.6809
# 0.1357, 31.8510
# 0.0788, 31.5010
# 0.0836, 32.7047
# 0.0759, 30.6186
# 0.1456, 32.6338
# 0.1227, 32.4355
# 0.1330, 32.1550
# 0.1070, 33.0135
# 0.0601, 31.5108
# exper/imgs7/e070_i000000.png
# 0.0865, 32.4333
# 0.0704, 31.2856
# 0.0689, 33.7744
# 0.0373, 31.3084
# 0.0499, 32.3002
# 0.0708, 31.8045
# 0.0493, 31.3536
# 0.0401, 31.9555
# 0.1040, 33.1984
# 0.0531, 33.0112
# exper/imgs7/e080_i000000.png
# 0.0482, 31.8470
# 0.0554, 29.7912
# 0.0297, 31.6717
# 0.2225, 31.7006
# 0.0539, 31.7478
# 0.0358, 31.3508
# 0.0718, 30.5018
# 0.0483, 31.0210
# 0.0661, 30.4469
# 0.0870, 31.2012
# exper/imgs7/e090_i000000.png
# 0.0823, 30.9695
# 0.0397, 29.4345
# 0.0851, 32.1604
# 0.0370, 30.3497
# 0.0670, 30.7299
# 0.0451, 31.6258
# 0.0364, 30.0254
# 0.0537, 29.0524
# 0.0318, 31.8457
# 0.0403, 31.0077
# exper/imgs7/e100_i000000.png
# 0.0317, 31.5600
# 0.0352, 30.4931
# 0.0357, 32.0764
# 0.0556, 31.0678
# 0.0298, 30.6098
# 0.0329, 32.1026
# 0.0295, 31.5026
# 0.0204, 31.0485
# 0.0314, 31.0728
# 0.0199, 30.7009
# exper/imgs7/e110_i000000.png
# 0.0201, 30.4909
# 0.0359, 30.2514
# 0.0248, 29.7495
# 0.0242, 29.9430
# 0.0579, 28.4697
# 0.0313, 28.7980
# 0.0159, 30.5332
# 0.0195, 28.7935
# 0.0237, 29.8653
# 0.0164, 29.6974
# exper/imgs7/e120_i000000.png
# 0.0232, 29.7213
# 0.0220, 30.6984
# 0.0245, 28.5715
# 0.0210, 29.3176
# 0.0166, 31.1172
# 0.0318, 28.5236
# 0.0134, 30.2209
# 0.0143, 29.5484
# 0.0182, 31.7431
# 0.0190, 28.4381
# exper/imgs7/e130_i000000.png
# 0.0473, 30.1641
# 0.0130, 28.2806
# 0.0265, 29.6843
# 0.0215, 30.2002
# 0.0130, 28.4276
# 0.0183, 30.8595
# 0.0147, 29.3951
# 0.0240, 28.8150
# 0.0192, 29.5392
# 0.0204, 29.3347
# exper/imgs7/e140_i000000.png
# 0.0197, 29.5431
# 0.0110, 30.6094
# 0.0146, 28.3586
# 0.0135, 30.5021
# 0.0155, 31.0355
# 0.0166, 29.0179
# 0.0109, 29.7755
# 0.0116, 28.6042
# 0.0135, 28.2166
# 0.0138, 29.0050
# exper/imgs7/e150_i000000.png
# 0.0138, 28.9347
# 0.0105, 29.3029
# 0.0081, 29.1462
# 0.0104, 29.1396
# 0.0129, 29.9593
# 0.0129, 28.4855
# 0.0089, 29.9316
# 0.0125, 29.8438
# 0.0157, 28.3603
# 0.0147, 25.9034
# exper/imgs7/e160_i000000.png
# 0.0087, 27.3430
# 0.0112, 29.0149
# 0.0083, 27.0192
# 0.0108, 28.5453
# 0.0096, 28.4511
# 0.0191, 29.2846
# 0.0129, 27.8264
# 0.0082, 27.4509
# 0.0104, 26.6050
# 0.0107, 28.3383
# exper/imgs7/e170_i000000.png
# 0.0125, 28.5605
# 0.0112, 25.9982
# 0.0075, 29.0745
# 0.0068, 28.4126
# 0.0120, 27.5619
# 0.0065, 28.3585
# 0.0185, 27.5098
# 0.0118, 28.3530
# 0.0117, 28.0030
# 0.0188, 27.7141
# exper/imgs7/e180_i000000.png
# 0.0149, 28.2647
# 0.0069, 26.8082
# 0.0089, 27.4867
# 0.0101, 26.0356
# 0.0076, 28.6480
# 0.0082, 27.8099
# 0.0070, 27.6887
# 0.0314, 27.8977
# 0.0175, 27.5460
# 0.0284, 26.3795
# exper/imgs7/e190_i000000.png
# 0.0068, 26.8965
# 0.0085, 28.2281
# 0.0083, 25.6560
# 0.0070, 28.3853
# 0.0070, 27.4427
# 0.0079, 27.9180
# 0.0120, 29.1410
# 0.0076, 27.9042
# 0.0071, 27.4163
# 0.0079, 28.0336
# exper/imgs7/e200_i000000.png
# 0.0110, 26.3875
# 0.0065, 26.6443
# 0.0065, 29.1726
# 0.0080, 27.9045
# 0.0490, 26.5813
# 0.0079, 27.2296
# 0.0079, 27.5608
# 0.0067, 27.3580
# 0.0067, 26.5400
# 0.0090, 26.3565
# exper/imgs7/e210_i000000.png
# 0.0071, 26.9569
# 0.0057, 26.6543
# 0.0109, 24.3056
# 0.0094, 26.4609
# 0.0072, 25.6353
# 0.0072, 26.9363
# 0.0066, 25.7237
# 0.0059, 26.7421
# 0.0077, 26.4547
# 0.0054, 27.1091
# exper/imgs7/e220_i000000.png
# 0.0087, 25.3521
# 0.0078, 27.3700
# 0.0152, 27.0907
# 0.0087, 24.6211
# 0.0079, 25.8891
# 0.0061, 26.8089
# 0.0056, 26.4493
# 0.0084, 25.1872
# 0.0317, 26.8651
# 0.0063, 25.0661
# exper/imgs7/e230_i000000.png
# 0.0089, 25.0656
# 0.0054, 25.7984
# 0.0092, 26.5774
# 0.0067, 26.0223
# 0.0050, 26.5490
# 0.0074, 25.6872
# 0.0133, 26.5506
# 0.0127, 26.6402
# 0.0082, 25.4075
# 0.0116, 26.5869
# exper/imgs7/e240_i000000.png
# 0.0049, 26.1838
# 0.0070, 26.9163
# 0.0063, 26.7012
# 0.0051, 24.3823
# 0.0066, 24.8018
# 0.0100, 25.2305
# 0.0039, 25.2452
# 0.0052, 24.7234
# 0.0047, 25.9097
# 0.0073, 26.2464
# exper/imgs7/e250_i000000.png
# 0.0050, 24.1465
# 0.0086, 24.2531
# 0.0089, 24.4825
# 0.0049, 24.1834
# 0.0080, 24.7320
# 0.0040, 24.5212
# 0.0068, 25.9208
# 0.0051, 26.4604
# 0.0042, 24.0275
# 0.0048, 23.4781
# exper/imgs7/e260_i000000.png
# 0.0065, 23.8230
# 0.0067, 23.3065
# 0.0079, 24.0419
# 0.0066, 24.8033
# 0.0046, 24.2192
# 0.0052, 26.0707
# 0.0032, 24.2905
# 0.0059, 23.6730
# 0.0063, 22.8270
# 0.0039, 24.2742
# exper/imgs7/e270_i000000.png
# 0.0045, 23.6840
# 0.0044, 24.9529
# 0.0045, 25.0883
# 0.0043, 24.5802
# 0.0038, 23.7476
# 0.0031, 24.9946
# 0.0057, 23.4135
# 0.0053, 23.1407
# 0.0039, 22.5043
# 0.0061, 23.1570
# exper/imgs7/e280_i000000.png
# 0.0048, 23.4226
# 0.0105, 24.2921
# 0.0031, 25.2449
# 0.0032, 24.3337
# 0.0047, 24.2640
# 0.0037, 24.0388
# 0.0038, 24.6598
# 0.0056, 23.4483
# 0.0036, 24.7154
# 0.0033, 23.1972
# exper/imgs7/e290_i000000.png
# 0.0060, 22.5911
# 0.0026, 23.9304
# 0.0056, 22.4757
# 0.0031, 23.8831
# 0.0049, 21.9219
# 0.0038, 22.3981
# 0.0042, 23.5447
# 0.0045, 23.3918
# 0.0033, 21.9127
# 0.0045, 23.4839
# exper/imgs7/e300_i000000.png
# 0.0034, 23.2092
# 0.0040, 22.3511
# 0.0041, 24.0225
# 0.0039, 23.9725
# 0.0031, 23.2418
# 0.0031, 22.7702
# 0.0055, 21.0451
# 0.0042, 24.2664
# 0.0033, 22.6513
# 0.0078, 23.3349
# exper/imgs7/e310_i000000.png
# 0.0043, 23.3502
# 0.0057, 23.4266
# 0.0042, 22.6382
# 0.0025, 21.5419
# 0.0023, 22.8275
# 0.0070, 20.0162
# 0.0035, 21.9044
# 0.0055, 23.1727
# 0.0034, 22.3167
# 0.0035, 21.2452
# exper/imgs7/e320_i000000.png
# 0.0031, 21.4530
# 0.0035, 22.5238
# 0.0028, 23.4775
# 0.0067, 22.0574
# 0.0034, 22.2789
# 0.0040, 21.8147
# 0.0024, 21.6580
# 0.0039, 21.5423
# 0.0056, 22.1291
# 0.0023, 21.8055
# exper/imgs7/e330_i000000.png
# 0.0024, 21.8872
# 0.0030, 22.3181
# 0.0024, 23.0927
# 0.0027, 22.7351
# 0.0024, 20.4704
# 0.0021, 20.3559
# 0.0035, 21.8898
# 0.0113, 21.9387
# 0.0035, 21.5198
# 0.0052, 20.9743
# exper/imgs7/e340_i000000.png
# 0.0031, 21.1908
# 0.0033, 21.9069
# 0.0060, 20.6056
# 0.0034, 20.3466
# 0.0025, 20.4734
# 0.0022, 21.2195
# 0.0029, 21.1423
# 0.0032, 19.7330
# 0.0029, 20.4059
# 0.0043, 19.9312
# exper/imgs7/e350_i000000.png
# 0.0024, 22.2775
# 0.0019, 20.5600
# 0.0023, 20.7899
# 0.0037, 21.1976
# 0.0041, 20.3029
# 0.0024, 19.8837
# 0.0027, 20.7582
# 0.0034, 19.7135
# 0.0036, 21.5464
# 0.0034, 20.7854
# exper/imgs7/e360_i000000.png
# 0.0020, 20.6064
# 0.0027, 21.6667
# 0.0023, 20.9346
# 0.0030, 21.0396
# 0.0022, 20.8881
# 0.0027, 20.7431
# 0.0028, 19.6332
# 0.0025, 19.5240
# 0.0037, 20.4748
# 0.0035, 19.4080
# exper/imgs7/e370_i000000.png
# 0.0022, 20.6028

# Example 3:
# ssh://ndioneli@localhost:9000/home/ndioneli/miniconda3/bin/python3 -u /remote/rds/users/ndioneli/pycharm_project_233/BMLOO2_All.py
# /home/ndioneli/miniconda3/lib/python3.7/site-packages/torchvision/datasets/mnist.py:43: UserWarning: train_labels has been renamed targets
#   warnings.warn("train_labels has been renamed targets")
# cuda:0
# 5.7465, 43.1000
# exper/imgs7/e000_i000000.png
# 4.5162, 42.8591
# 3.4173, 40.3686
# 3.6331, 40.7682
# 3.1428, 40.2793
# 2.2279, 38.9123
# 1.8856, 39.2028
# 1.3951, 38.6728
# 1.0303, 39.2331
# 0.9638, 39.7416
# 0.8574, 39.2998
# exper/imgs7/e010_i000000.png
# 0.7267, 38.1767
# 0.9404, 37.2237
# 0.8788, 37.7690
# 1.1643, 37.5878
# 1.0994, 38.3362
# 1.0726, 38.5032
# 0.6426, 38.3113
# 0.5710, 37.7881
# 0.4603, 36.3523
# 0.6597, 36.0904
# exper/imgs7/e020_i000000.png
# 1.0246, 36.3351
# 0.6635, 35.6143
# 0.4359, 36.2932
# 0.5053, 35.6956
# 0.4493, 36.1311
# 0.6705, 35.4996
# 0.7682, 36.6494
# 0.3624, 33.7040
# 0.4111, 35.5911
# 0.3327, 34.3462
# exper/imgs7/e030_i000000.png
# 0.4319, 33.9034
# 0.2159, 34.9503
# 0.4079, 34.9825
# 0.2824, 34.6280
# 0.2642, 34.9251
# 0.3525, 35.4927
# 0.3017, 35.4238
# 0.2579, 36.1678
# 0.1648, 32.8105
# 0.3608, 33.4898
# exper/imgs7/e040_i000000.png
# 0.1318, 34.0996
# 0.1249, 35.0168
# 0.1495, 33.3399
# 0.0994, 31.9930
# 0.2220, 30.8501
# 0.1717, 34.6405
# 0.1177, 34.5493
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
# Additional acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs)
