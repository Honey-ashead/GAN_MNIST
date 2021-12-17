import os
import argparse
import torch
import numpy as np
from torch import nn
from torch.nn import BCELoss
from torch.optim import Adam

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200, help='number of training times')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--alpha', type=float, default=1.0, help='to adjust the lr of G')
parser.add_argument('--beta', type=float, default=1.0, help='to adjust the lr of C')
parser.add_argument('--image_size', type=tuple, default=(28, 28), help='size of input images')
parser.add_argument('--channels', type=int, default=1, help='channels of input images')
parser.add_argument('--save_intervals', type=int, default=100, help='intervals between saving models')

options = parser.parse_args([])
image_shape = (options.channels, options.image_size[0], options.image_size[1])


class Generator(nn.Module):
    def __init__(self, in_feature):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, _x):
        x = self.model(_x)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, _x):
        x = _x.view(_x.size(0), -1)
        return self.model(x)


if __name__ == '__main__':
    os.makedirs('new_images', exist_ok=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(options.latent_dim).double()
    discriminator = Discriminator().double()

    loss_fn = BCELoss()
    optimizer_generator = Adam(generator.parameters())
    optimizer_discriminator = Adam(discriminator.parameters())

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        # loss_fn = loss_fn.cuda()

    dataloader = DataLoader(
        MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(options.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        ),
        batch_size=options.batch_size,
        shuffle=True
    )

    for epoch in range(options.epochs):
        for idx, (images, _) in enumerate(dataloader):
            if idx != len(dataloader) - 1:
                valid = torch.ones((256, 1), dtype=torch.float64, requires_grad=False).cuda()
                fake = torch.zeros((256, 1), dtype=torch.float64, requires_grad=False).cuda()

                real_images = images.type(torch.float64).cuda()
                optimizer_generator.zero_grad()
                z = torch.tensor(np.random.normal(0, 1, (images.shape[0], options.latent_dim))).cuda()
                generate_images = generator(z)

                # train generator
                gen_loss = loss_fn(discriminator(generate_images), valid)
                gen_loss.backward()
                optimizer_generator.step()

                # train discriminator
                optimizer_discriminator.zero_grad()
                real_loss = loss_fn(discriminator(real_images), valid)
                fake_loss = loss_fn(discriminator(generate_images.detach()), fake)
                dis_loss = (real_loss + fake_loss) / 2
                dis_loss.backward()
                optimizer_discriminator.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, options.epochs, idx + 1, len(dataloader), dis_loss.item(), gen_loss.item())
                )

                batches_done = epoch * len(dataloader) + epoch
                if batches_done % options.save_intervals == 0:
                    save_image(generate_images.data[:1024], "new_images/%d.png" % batches_done, nrow=32, normalize=True)
