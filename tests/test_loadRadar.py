from datetime import datetime
from unittest.mock import patch
import numpy as np
from loadRadar import generateExpectedFilenames, loadRadar, color_to_rain


# Test for generateExpectedFilenames
def test_generateExpectedFilenames():
    # Set a known initial time
    initial_time = datetime(2025, 5, 3, 14, 0, 0)
    filenames = generateExpectedFilenames(initial_time, n_images=3, step_minutes=10)

    # Manually create the expected filenames
    expected_filenames = [
        "dpsri_70km_2025050314000000dBR.dpsri.png",
        "dpsri_70km_2025050314100000dBR.dpsri.png",
        "dpsri_70km_2025050314200000dBR.dpsri.png",
    ]

    assert filenames == expected_filenames


# Test for loadRadar
@patch("loadRadar.grabRadarSG")  # Mock grabRadarSG
def test_loadRadar(mock_grabRadarSG):
    # Set mock behaviors
    mock_grabRadarSG.return_value = datetime(2025, 4, 26, 11, 30, 0)

    # Create a mock image object with RGB data
    radar_stack, rainfall_stack, color_map, radar_time = loadRadar()

    # Check if the radar_stack and rainfall_stack have correct shape
    assert radar_stack.shape == (25, 1196, 1219)
    assert rainfall_stack.shape == (25, 1196, 1219)

    # Ensure color_to_rain mapping was used
    rainfall_values = {v[0] for v in color_to_rain.values()}
    radar_values = {v[1] for v in color_to_rain.values()}
    rainfall_values.add(0.0)
    radar_values.add(0.0)

    unique_rain = np.unique(rainfall_stack)
    unique_radar = np.unique(radar_stack)

    tol = 1e-6

    def close_enough(x, allowed):
        diff = np.abs(x[:, None] - allowed[None, :])
        mask = diff <= tol

        ok = mask.any(axis=1)

        bad = x[~ok]

        return ok.all(), bad

    ok1, bad1 = close_enough(unique_rain, np.array(list(rainfall_values)))
    ok2, bad2 = close_enough(unique_radar, np.array(list(radar_values)))

    assert ok1, f"Bad rainfall values: {bad1}"
    assert ok2, f"Bad radar values: {bad2}"

    # Test if the expected time was returned
    assert radar_time == datetime(2025, 4, 26, 11, 50, 0)
