import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from model import Generator, Critic
from utils import gradient_penalty, initialize_weights, show_tensor_images

if torch.cuda.device_count() > 1:
    device = torch.device("cuda:0")
    print("Training on multiple GPUs:", torch.cuda.device_count())
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on a single GPU:", device)

lr = 1e-4
batch_size = 64
img_size = 64
color_channels = 3
noise_dim = 100
n_epochs = 20
n_filters_gen = 64
n_filters_critic = 128
critic_iterations = 5
lambda_gp = 10

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

root_dir = '/kaggle/input/celeba-dataset'
faces_dataset = ImageFolder(root=root_dir, transform=transform)
dataloader = DataLoader(faces_dataset,batch_size=batch_size,shuffle=True)

generator = Generator(noise_dim, n_filters_gen, color_channels).to(device)
critic = Critic(n_filters_critic, color_channels).to(device)

generator = nn.DataParallel(generator)
critic = nn.DataParallel(critic)

initialize_weights(generator)
initialize_weights(critic)
generator.train()
critic.train()

optimizer_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

step = 0
fixed_noise = torch.randn(32, noise_dim, 1, 1).to(device)

for epoch in range(n_epochs):
    for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(critic_iterations):
            noise = torch.randn(cur_batch_size, noise_dim, 1, 1, device=device)
            fake = generator(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real)-torch.mean(critic_fake))+lambda_gp*gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()

        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        generator.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_idx % 500 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{n_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            fake = generator(fixed_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            step += 1