import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

trainset = torchvision.datasets.MNIST(root = '/data',
                                      train = True,
                                      download=True)

testset = torchvision.datasets.MNIST(root='/data',
                                      train=False,
                                      download=True)
