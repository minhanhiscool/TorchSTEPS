import torch
from torch.utils.data import Dataset
from datetime import timedelta
from pySTEPS.loadRadar import loadRadar
from pySTEPS.displayRadar import displayRadar
from pySTEPS.tradPred import tradPred


class RadarDataset(Dataset):
    def __init__(self, times_list, T_in, T_out, H, W):
        self.times = times_list
        self.T_in = T_in
        self.T_out = T_out
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        init_time = self.times[index]

        x, extrap, anvil, steps = tradPred(init_time)
        _, ground_truth, _ = loadRadar(
            init_time + timedelta(minutes=5 * self.T_in), n_images=self.T_out
        )

        displayRadar(x, "Radar", init_time)
        displayRadar(
            extrap, "Extrapolation", init_time + timedelta(minutes=5 * self.T_in)
        )
        displayRadar(anvil, "Anvil", init_time + timedelta(minutes=5 * self.T_in))
        displayRadar(steps, "STEPS", init_time + timedelta(minutes=5 * self.T_in))
        displayRadar(
            ground_truth, "Ground truth", init_time + timedelta(minutes=5 * self.T_in)
        )

        def to_tensor(arr):
            t = torch.tensor(arr).unsqueeze(1).float()
            t = torch.nan_to_num(t, nan=0.0)
            t = torch.clamp(t, min=0.0, max=150.0)
            t = t / 150.0
            return t

        return (
            to_tensor(x),
            to_tensor(extrap),
            to_tensor(anvil),
            to_tensor(steps),
            to_tensor(ground_truth),
        )
