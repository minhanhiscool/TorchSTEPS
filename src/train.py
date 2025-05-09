from torchLSTM import ConvLSTM_Encoder_Decoder
from pySTEPS.loadRadar import loadRadar
from pySTEPS.tradPred import tradPred
from matplotlib import pyplot as plt
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch import nn, optim
from datetime import datetime, timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_batch(x, H_out=512, W_out=512):
    """
    Takes a batch of input data of shape (B, T, C, H_in, W_in) and resizes it to (B, T, C, H_out, W_out)
    Typically H_out and W_out are 512

    Args:
        x (torch.Tensor): Input data of shape (B, T, C, H_in, W_in)
        H_out (int, optional): Height of the output tensor. Defaults to 512.
        W_out (int, optional): Width of the output tensor. Defaults to 512.

    Returns:
        torch.Tensor: Resized input data of shape (B, T, C, H_out, W_out):
    """
    # x: (B, T, C, H_in, W_in)
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)
    x = F.interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=False)
    return x.view(B, T, C, H_out, W_out)


def random_time():
    """
    Produces a random time between 2025-04-16 13:45 and 2025-04-26 10:15

    Returns:
        datetime: A random datetime between 2025-04-16 13:45 and 2025-04-26 10:15
    """
    start_time = datetime(2025, 4, 16, 13, 45)
    end_time = datetime(2025, 4, 26, 10, 15)

    delta = timedelta(minutes=5)
    num_steps = int((end_time - start_time) / delta)

    random_step = random.randint(0, num_steps - 1)
    random_time = start_time + random_step * delta

    return random_time


def train(B, T_in, T_out, C, H, W):
    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model = model.to(device)

    # Initializes lists of NumPy Arrays, which will be condensed to Tensors
    x_stack = np.empty((B, T_in, H, W))
    precip_extrapolation_stack = np.empty((B, T_out, H, W))
    precip_anvil_stack = np.empty((B, T_out, H, W))
    precip_steps_mean_stack = np.empty((B, T_out, H, W))
    ground_truth_stack = np.empty((B, T_out, H, W))

    for i in range(B):
        init_time = random_time()
        training_data, precip_extrapolation, precip_anvil, precip_steps_mean = tradPred(
            init_time
        )  # (T_in/T_out, H, W)
        _, ground_truth, _ = loadRadar(
            init_time + timedelta(minutes=T_in * 5), n_images=T_out
        )  # (T_out, H, W)

        x_stack[i] = training_data
        precip_extrapolation_stack[i] = precip_extrapolation
        precip_anvil_stack[i] = precip_anvil
        precip_steps_mean_stack[i] = precip_steps_mean
        ground_truth_stack[i] = ground_truth

    # Condense all arrays to Tensors, ready for forward pass
    x = resize_batch(torch.tensor(x_stack).unsqueeze(2).float().to(device))
    m1 = resize_batch(
        torch.tensor(precip_extrapolation_stack).unsqueeze(2).float().to(device)
    )
    m2 = resize_batch(torch.tensor(precip_anvil_stack).unsqueeze(2).float().to(device))
    m3 = resize_batch(
        torch.tensor(precip_steps_mean_stack).unsqueeze(2).float().to(device)
    )
    ground_truth = resize_batch(
        torch.tensor(ground_truth_stack).unsqueeze(2).float().to(device)
    )
    # By this point, all tensors should have a size (B, T_out/T_in, 1/C, 512, 512)

    out = model(x, m1, m2, m3, ground_truth)

    for t in range(T_out):
        plt.figure(figsize=(10, 10))
        plt.imshow(out[0, t, 0].cpu().detach().numpy())
        plt.title(f"Predicted Precipitation at Time {t * 5} minutes")
        plt.axis("off")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    train(1, 6, 18, 1, 1219, 1196)
