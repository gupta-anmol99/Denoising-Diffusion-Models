import random
import imageio
import numpy as np

from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from helpers import show_first_batch, show_images, show_forward
from unet import MyUNet




class DDPMMModel(nn.Module):
    def __init__(self, network, n_steps = 1000, min_beta = 10**-4, max_beta = 0.02,
                device=None, image_chw=(1,28,28)):
        super(DDPMMModel, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)

        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eps=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eps is None:
            eps = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eps
        return noisy


    def backward(self, x, t):

        return self.network(x, t)







