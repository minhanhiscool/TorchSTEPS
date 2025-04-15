from pysteps import motion, nowcasts
import time
from loadRadar import loadRadar
from displayRadar import displayRadar

geodata = {
    "projection": "+proj=longlat +datum=WGS84",  # Assuming WGS84 projection
    "x1": 103.3404,  # Longitude of lower-left corner
    "y1": 0.7370,  # Latitude of lower-left corner
    "x2": 104.5987,  # Longitude of upper-right corner
    "y2": 1.9712,  # Latitude of upper-right corner
    "yorigin": "upper",  # Assuming the first element is at the upper border
}


def extrapolate():
    radar_stack, rainfall_stack, color_to_rain, current_time = loadRadar()

    training_data = radar_stack[-5:]
    oflow_method = motion.get_method("LK")
    motion_field = oflow_method(training_data)

    start = time.time()

    anvil = nowcasts.get_method("anvil")

    n_leadtimes = 20

    precip_nowcast = anvil(rainfall_stack[-4:], motion_field, n_leadtimes)

    end = time.time()
    print(f"Applying ANVIL to the radar rainfall fields took {(end - start)} seconds")

    displayRadar(precip_nowcast, "Predicted Rainfall using ANVIL", current_time)


if __name__ == "__main__":
    extrapolate()
