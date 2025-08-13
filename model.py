import torch
import torch.nn as nn


def gradient_penalty(critic, labels, real, fake, device = "cpu"):
    Batch_size, C, H, W = real.shape
    epsilon = torch.rand((Batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images, labels)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)

    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, embed_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim + embed_size, features_g * 16, kernel_size=4, stride=1, padding=0),
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.embed = nn.Embedding(num_classes, embed_size)
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)

class Critic(nn.Module):
    def __init__(self, channel_img, features_d, num_classes, img_size):
        super(Critic, self).__init__()
        self.img_size = img_size
        self.crit = nn.Sequential(
            nn.Conv2d(channel_img + 1, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            self._block(features_d * 8, features_d * 16, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid()
        )

        self.embed = nn.Embedding(num_classes, img_size*img_size)
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x,embedding], dim=1)
        return self.crit(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

def test():
    N, in_channel, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N, in_channel, H, W))
    crit = Critic(in_channel, 8)
    initialize_weights(crit)
    assert crit(x).shape == (N, 1, 1, 1), "Discriminator output shape mismatch"
    z = torch.randn((N, z_dim, 1, 1))
    gen = Generator(z_dim, in_channel, 8)
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channel, H, W), "Generator output shape mismatch"
    print("Discriminator and Generator are working correctly.")
