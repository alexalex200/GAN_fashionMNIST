import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model import Generator, Critic, initialize_weights, gradient_penalty
matplotlib.use("TkAgg")

def show_tensor_image(tensor_img):
    np_img = tensor_img.cpu().numpy()
    np_img = np.transpose(np_img,(1,2,0))
    plt.figure(figsize=(8,8))
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.FashionMNIST(root="data/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, GEN_EMBEDDING).to(device)
crit = Critic(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)

initialize_weights(gen)
initialize_weights(crit)

gen_opt = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
crit_opt = optim.Adam(crit.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)

gen.train()
crit.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        print(labels)
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((cur_batch_size, Z_DIM, 1, 1)).to(device)
            fake = gen(noise, labels)
            critic_real = crit(real, labels).reshape(-1)
            critic_fake = crit(fake, labels).reshape(-1)
            gp = gradient_penalty(crit, labels, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
            crit.zero_grad()
            loss_critic.backward(retain_graph=True)
            crit_opt.step()

        output = crit(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        if batch_idx % 100 == 0:
            torch.save(gen.state_dict(), f"saved_models/gen.pth")
            torch.save(crit.state_dict(), f"saved_models/crit.pth")
            with torch.no_grad():
                fake = gen(noise, labels)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                show_tensor_image(img_grid_real)
                show_tensor_image(img_grid_fake)





