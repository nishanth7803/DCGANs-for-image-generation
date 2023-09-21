import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from utils import show_tensor_images, initialize_weights
from scipy import linalg
import numpy as np
import torchvision.models as models

batch_size = 128
num_epochs = 25
noise_dim = 100
n_filter_gen = 64
n_filter_disc = 64
color_channels = 3
criterion = nn.BCELoss()
lr = 0.0002
momentum = 0.5

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

root_dir = '/kaggle/input/celeba-dataset'
faces_dataset = ImageFolder(root=root_dir, transform=transform)
dataloader = DataLoader(faces_dataset,batch_size=batch_size,shuffle=True)

generator = Generator(noise_dim, n_filter_gen, color_channels).to(device)  
discriminator = Discriminator(n_filter_disc, color_channels).to(device)

generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

initialize_weights(generator)
initialize_weights(critic)
generator.train()
critic.train()

optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(momentum, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(momentum, 0.999))

cur_step = 0
display_step = 1000
mean_generator_loss = 0
mean_discriminator_loss = 0
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_images, _ = data
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        discriminator.zero_grad()
        label = torch.full((batch_size,), 1.0, device=device)  
        output_real = discriminator(real_images).view(-1)
        errD_real = criterion(output_real, label)
        errD_real.backward()

        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_images = generator(noise)
        label.fill_(0.0)  
        output_fake = discriminator(fake_images.detach()).view(-1)
        errD_fake = criterion(output_fake, label)
        errD_fake.backward()
        
        errD = (errD_real + errD_fake)/2
        mean_discriminator_loss += errD.item() / display_step
        optimizerD.step()

        generator.zero_grad()
        label.fill_(1.0)  
        output = discriminator(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        mean_generator_loss += errG.item() / display_step

        if (cur_step % display_step == 0 and cur_step > 0):
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_tensor_images(fake_images)
            show_tensor_images(real_images)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
          
        cur_step += 1
