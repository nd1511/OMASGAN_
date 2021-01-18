import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
def choose_dataset(select_dataset):
    if select_dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        MNIST = torchvision.datasets.MNIST('data-cifar10', train=True, download=True, transform=transform)
        return MNIST
    elif select_dataset == "mnist2":
        transform = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        MNIST = torchvision.datasets.MNIST('data-cifar10', train=True, download=True, transform=transform)
        return MNIST
    elif select_dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        CIFAR10 = torchvision.datasets.CIFAR10('data-cifar10', train=True, download=True, transform=transform)
        return CIFAR10
    else:
        raise ValueError("Unknown dataset.")
