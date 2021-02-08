import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
imgsize = 32
def choose_dataset(select_dataset):
    if select_dataset == "mnist":
        # Use transforms.Resize(imgsize) because MNIST has 28*28=784 dimensions.
        transform = transforms.Compose([transforms.Resize(imgsize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        MNIST = torchvision.datasets.MNIST('data-cifar10', train=False, download=True, transform=transform)
        #import torchvision.datasets as dset
        #MNIST = dset.MNIST('data-cifar10', train=False, download=True, transform=transform)
        return MNIST
    elif select_dataset == "mnist2":
        #transform = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # We use transforms.Resize(imgsize) because MNIST has 28*28=784 dimensions.
        transform = transforms.Compose([transforms.Grayscale(3), transforms.Resize(imgsize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        MNIST = torchvision.datasets.MNIST('data-cifar10', train=False, download=True, transform=transform)
        return MNIST
    elif select_dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        CIFAR10 = torchvision.datasets.CIFAR10('data-cifar10', train=False, download=True, transform=transform)
        return CIFAR10
    else:
        raise ValueError("This is an unknown dataset.")
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
# Additional acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs)
