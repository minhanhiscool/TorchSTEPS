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


def all_time():
    """
    Produces a list of all times between 2025-04-16 14:30 and 2025-04-26 12:30

    Returns:
        List[datetime]: List of all times
    """
    start_time = datetime(2025, 4, 16, 14, 30)
    end_time = datetime(2025, 4, 26, 12, 30)

    total_minutes = int(end_time.timestamp() - start_time.timestamp()) // 60
    all_time_list = [
        start_time + timedelta(minutes=i * 5) for i in range(total_minutes // 5 + 1)
    ]

    random.shuffle(all_time_list)
    return all_time_list[:100]
