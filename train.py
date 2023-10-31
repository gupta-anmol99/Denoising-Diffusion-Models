import random
import imageio
import numpy as np

from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from helpers import show_first_batch, show_images, show_forward, generate_new_images
from unet import MyUNet
from ddpm import DDPMMModel

def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


def main():
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
    STORE_PATH_FASHION = f"ddpm_model_fashion.pt"

    no_train = False
    fashion = True
    batch_size = 128
    n_epochs = 10
    lr = 0.001
    store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)])
    
    ds_fn = FashionMNIST if fashion else MNIST
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)


    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
    ddpm = DDPMMModel(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)


    store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
    if not no_train:
        training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), 
                      device=device, store_path=store_path)

    best_model = DDPMMModel(MyUNet(), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Model loaded")

    print("Generating new images")
    generated = generate_new_images(
    best_model,
    n_samples=100,
    device=device,
    gif_name="fashion.gif" if fashion else "mnist.gif"
    )
    show_images(generated, "Final result")

if __name__ == "__main__":
    main()