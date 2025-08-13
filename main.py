import torch
import torchvision.datasets as datasets
import torchvision.utils as utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
matplotlib.use("TkAgg")


Z_DIM = 100
CHANNEL_IMAGE = 1
FEATURES_G = 64
NUM_CLASSES = 10
EMBEDDED_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator(Z_DIM,CHANNEL_IMAGE,FEATURES_G, NUM_CLASSES, EMBEDDED_SIZE)
gen.load_state_dict(torch.load("saved_models/gen_.pth"))
gen.to(device)
gen.eval()

data_set = datasets.FashionMNIST(root="data/")

with torch.no_grad():
    for label in range(10):
        labels = torch.tensor([label]*32).to(device)
        random_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)
        fake = gen(random_noise, labels)

        grid = utils.make_grid(fake, normalize=True, nrow=8)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
        plt.title(f"Class: {label} ({data_set.classes[label]})")
        plt.axis("off")
        plt.show()


