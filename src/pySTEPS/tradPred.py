from pysteps import motion, nowcasts
import numpy as np
from datetime import datetime
from pySTEPS.loadRadar import loadRadar


def tradPred(init_time=datetime(2025, 4, 25, 15, 30, 0)):
    """
    Inputs training data and outputs extrapolated rainfall, ANVIL prediction and STEP prediction

    Args:
        init_time (datetime, optional): Initial time of the radar data. Defaults to datetime(2025, 4, 25, 15, 30, 0).

    Returns:
        List[np.array]: a list of original rainfall data
        List[np.array]: a list of extrapolated rainfall data
        List[np.array]: a list of ANVIL prediction
        List[np.array]: a list of STEP prediction

        (All List[] size are of size (T_in/T_out, H, W)
    """
    radar_stack, rainfall_stack, current_time = loadRadar(init_time)
    for i, arr in enumerate(radar_stack):
        print(f"Radar image {i} has shape {arr.shape}")

    motion_field = motion.darts.DARTS(radar_stack)

    extrapolate = nowcasts.get_method("extrapolation")
    anvil = nowcasts.get_method("anvil")
    steps = nowcasts.get_method("steps")

    n_leadtimes = 18

    precip_nowcast_extrapolation = extrapolate(
        rainfall_stack[-1], motion_field, n_leadtimes
    )
    precip_nowcast_anvil = anvil(
        rainfall_stack[-4:],
        motion_field,
        n_leadtimes,
        fft_method="pyfftw",
        num_workers=6,
    )
    precip_nowcast_steps = steps(
        rainfall_stack[-3:],
        motion_field,
        n_leadtimes,
        n_ens_members=6,
        n_cascade_levels=6,
        precip_thr=0.5,
        kmperpixel=1,
        timestep=5,
        noise_method="nonparametric",
        fft_method="pyfftw",
        vel_pert_method="bps",
        mask_method="incremental",
        seed=42,
        num_workers=6,
    )
    precip_nowcast_steps_mean = np.mean(precip_nowcast_steps, axis=0)

    return (
        rainfall_stack,
        precip_nowcast_extrapolation,
        precip_nowcast_anvil,
        precip_nowcast_steps_mean,
    )


if __name__ == "__main__":
    tradPred()
