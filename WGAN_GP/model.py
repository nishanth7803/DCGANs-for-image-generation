import torch
import torch.nn as nn

def Generator(noise_dim, n_filters, color_channels):
    generator = nn.Sequential(
        nn.ConvTranspose2d(noise_dim, n_filters*8, kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(n_filters*8),
        nn.ReLU(True),
        nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(n_filters*4),
        nn.ReLU(True),
        nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(n_filters*2),
        nn.ReLU(True),
        nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(True),
        nn.ConvTranspose2d(n_filters, color_channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    )
    return generator

def Critic(n_filters, color_channels):
    critic = nn.Sequential(
        nn.Conv2d(color_channels, n_filters, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(n_filters*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(n_filters*2, n_filters*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(n_filters*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(n_filters*4, n_filters*8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(n_filters*8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(n_filters*8, 1, kernel_size=4, stride=2, padding=0),
    )
    return critic
