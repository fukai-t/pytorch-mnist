import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = torchvision.datasets.MNIST(root = '/data',
                                      train = True,
                                      download=True,
                                      transform=_transforms)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.MNIST(root='/data',
                                     train=False,
                                     download=True,
                                     transform=_transforms)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=2)
