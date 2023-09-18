import torch
import torch as nn

def Generator(noise_dim, base_filter, color_channels):

    generator_model = nn.Sequential(
        nn.ConvTranspose2d(noise_dim, base_filter*8, kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(base_filter*8),
        nn.ReLU(True),
        nn.ConvTranspose2d(base_filter*8, base_filter*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(base_filter*4),
        nn.ReLU(True),
        nn.ConvTranspose2d(base_filter*4, base_filter*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(base_filter*2),
        nn.ReLU(True),
        nn.ConvTranspose2d(base_filter*2, base_filter, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(base_filter),
        nn.ReLU(True),
        nn.ConvTranspose2d(base_filter, color_channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )
    
    return generator_model

def Discriminator(base_filter, color_dimensions):

    discriminator_model = nn.Sequential(
        nn.Conv2d(color_channels, base_filter, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_filter, base_filter*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(base_filter*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_filter*2, base_filter*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(base_filter*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_filter*4, base_filter*8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(base_filter*8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_filter*8, 1, kernel_size=4, stride=1, padding=0),
        nn.Sigmoid()
    )

    return discriminator_model
