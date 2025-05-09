from convlstm import ConvLSTM
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class ConvLSTM_Encoder_Decoder(nn.Module):
    """
    Implements a encoder-decoder architecture with ConvLSTM with early fusion


    Attributes:
        encoder (ConvLSTM): Encoder ConvLSTM
        decoder (ConvLSTM): Decoder ConvLSTM
        conv_last (nn.Conv2d): Convolutional layer for the last output
        teacher_forcing_ratio (float): Probability of using ground truth as input at each time step
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=2,
        batch_first=True,
        bias=True,
        return_all_layers=True,
    ):
        """
        Initialize the ConvLSTM_Encoder_Decoder model

        Args:
            input_dim (int): Number of input channels in the input sequence
            hidden_dim (int or List[int]): Number of hidden channels in each ConvLSTM cell
            kernel_size (tuple or List[tuple]): Size of the convolutional kernel in each ConvLSTM cell
            num_layers (int): Number of ConvLSTM cells in the model, default is 2
            batch_first (bool): If set to true, input tensor is in the shape of (B, T, C, H, W) instead of (T, B, C, H, W), default is True
            bias (bool): Whether or not to add the bias to the ConvLSTM cell, default is True
            return_all_layers (bool): If set to true, will return all intermediate outputs from the encoder and decoder, default is True

        """
        super(ConvLSTM_Encoder_Decoder, self).__init__()
        self.encoder = ConvLSTM(
            input_dim,
            hidden_dim,
            kernel_size,
            num_layers,
            batch_first,
            bias,
            return_all_layers,
        )
        self.decoder = ConvLSTM(
            input_dim + 3,
            hidden_dim,
            kernel_size,
            num_layers,
            batch_first,
            bias,
            return_all_layers,
        )
        in_ch = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        self.conv_last = nn.Conv2d(
            in_ch, out_channels=1, kernel_size=1, padding=0, bias=True
        )
        if bias:
            nn.init.zeros_(self.conv_last.bias)

        self.teacher_forcing_ratio = 0.4

    def forward(self, x, m1, m2, m3, ground_truth=None):
        """
        The forward pass function for the ConvLSTM_Encoder_Decoder model

        Args:
            x (Tensor): Input tensor of shape (B, T_in, C, H, W) where
                        B is the batch size, T_in is the number of input frames, C is the number of channels,
                        H and W are height and width
            m1 (Tensor): Guidance model 1 tensor of shape (B, T_out, 1, H, W) (T_out is the number of output frames)
            m2 (Tensor): Guidance model 2 tensor of shape (B, T_out, 1, H, W)
            m3 (Tensor): Guidance model 3 tensor of shape (B, T_out, 1, H, W)
            ground_truth (Tensor, optional): Ground truth tensor of shape (B, T_out, 1, H, W) (T_out is the number of output frames) if provided, default is None

        Returns:
            Tensor: Output tensor of shape (B, T_out, 1, H, W)
        """
        device = x.device

        B, T_out, _, H, W = m1.shape  # (B, T_out, 1, H, W)
        _, last_states = self.encoder(x)

        # last_states has list of size T_in, each element is a tuple of hidden state and cell state

        prev_pred = torch.zeros((B, 1, 1, H, W), device=device)  # (B, 1, 1, H, W)
        dec_states = last_states
        output = []
        for i in range(T_out):
            f1 = m1[:, i].to(device)  # (B, 1, H, W)
            f2 = m2[:, i].to(device)  # (B, 1, H, W)
            f3 = m3[:, i].to(device)  # (B, 1, H, W)

            f = torch.cat([f1, f2, f3], dim=1)  # (B, 3, H, W)
            f_expanded = f.unsqueeze(1)  # (B, 1, 3, H, W)

            if (
                ground_truth is not None
                and torch.rand(1).item() < self.teacher_forcing_ratio
            ):
                prev = ground_truth[:, i].to(device)  # (B, 1, H, W)
            else:
                prev = prev_pred[:, 0]  # (B, 1, H, W)

            inp = torch.cat([f, prev], dim=1)  # (B, 4, H, W)
            inp = inp.unsqueeze(1)  # (B, 1, 4, H, W)

            y, dec_states = self.decoder(inp, f_expanded, dec_states)
            dec_states = [(h.detach(), c.detach()) for h, c in dec_states]
            h = y[0].squeeze(1)
            pred = self.conv_last(h)
            output.append(pred.unsqueeze(1))

            prev_pred = pred.unsqueeze(1)

        return torch.cat(output, dim=1)
