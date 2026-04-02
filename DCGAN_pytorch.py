import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import argparse

class Generator(nn.Module):
    def __init__(self, latent_size, image_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.image_resize = image_size // 4
        self.latent_size = latent_size
        
        self.fc = nn.Linear(latent_size, self.image_resize * self.image_resize * 128)
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1),
            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=1, padding=2),
            
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 5, stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, self.image_resize, self.image_resize)
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, stride=1, padding=2),
            
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def plot_images(generator, noise_input, device, step=0, model_name="gan_pytorch"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    
    generator.eval()
    with torch.no_grad():
        images = generator(noise_input).cpu().numpy()
    generator.train()
    
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[2]
    rows = int(math.sqrt(num_images))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = images[i, 0, :, :]
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    plt.close('all')

def train():
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    model_name = "dcgan_mnist_pytorch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    generator = Generator(latent_size, 28).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr * 0.5)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, latent_size).to(device)
    
    data_iter = iter(dataloader)
    
    for i in range(train_steps):
        try:
            real_images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_images, _ = next(data_iter)
            
        real_images = real_images.to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size, 1).to(device)
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)
        
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        output_fake_for_G = discriminator(fake_images)
        loss_G = criterion(output_fake_for_G, real_labels)
        
        loss_G.backward()
        optimizer_G.step()
        
        if i % 100 == 0:
            print(f"{i}: [D loss: {loss_D.item():.6f}] [G loss: {loss_G.item():.6f}]")
            
        if (i + 1) % 500 == 0:
            plot_images(generator, fixed_noise, device, step=(i + 1), model_name=model_name)

    torch.save(generator.state_dict(), f"{model_name}_generator.pth")
    torch.save(discriminator.state_dict(), f"{model_name}_discriminator.pth")

if __name__ == "__main__":
    train()
