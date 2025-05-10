import torch.nn.functional as F
import random
from datetime import datetime, timedelta


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
