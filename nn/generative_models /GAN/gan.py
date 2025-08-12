import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from dataclasses import dataclass


@dataclass
class GANConfig:
    latent_dim: int = 100
    img_channels: int = 1
    img_size: int = 28
    batch_size: int = 128
    epochs: int = 50
    lr: float = 0.0002
    device: str = "cuda"


class Generator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        latent_dim = config.latent_dim
        img_channels = config.img_channels
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 5, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # output between [-1, 1]
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        img_channels = config.img_channels
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            # If the last layer had img_channels outputs,
            # the discriminator would try to reconstruct the image (like an autoencoder),
            # which is not its goal. The discriminator only needs to output a single confidence score.
            nn.Sigmoid(),  # output between [0, 1]
        )

    def forward(self, x):
        return self.model(x).view(-1)


class Trainer:
    def __init__(self, config:GANConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Scale to [-1, 1]
        ])

        self.dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)

    def train(self):
        for epoch in range(self.config.epochs):
            for i, (real_imgs, _) in enumerate(self.dataloader):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                # adversarial ground truth
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)
                
                ## train discriminator
                self.optimizer_D.zero_grad()
                real_output = self.discriminator(real_imgs)
                d_loss_real = self.criterion(real_output, real_labels)

                # fake images
                noise = torch.randn(batch_size, self.config.latent_dim, 1, 1, device=self.device)
                fake_imgs = self.generator(noise)
                fake_output = self.discriminator(fake_imgs.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_D.step()
                
                ## train generator
                self.optimizer_G.zero_grad()

                # generate fake images and try to fool discriminator
                gen_output = self.discriminator(fake_imgs)
                g_loss = self.criterion(gen_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{self.config.epochs}] [Batch {i}/{len(self.dataloader)}] "
                        f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
                    
                
        
