import glob
from datetime import timedelta
from PIL import Image
import numpy as np
from grabRadarSG import grabRadarSG
from displayRadar import displayRadar

color_to_rain = {
    (255, 0, 255): (150, 80),
    (212, 0, 170): (125, 77.5),
    (182, 0, 106): (100, 75),
    (193, 0, 0): (75, 72.5),
    (229, 0, 0): (50, 70),
    (255, 31, 0): (40, 67.5),
    (255, 73, 0): (30, 65),
    (255, 114, 0): (25, 62.5),
    (251, 138, 1): (20, 60),
    (255, 165, 0): (16, 57.5),
    (255, 178, 0): (12, 55),
    (255, 198, 0): (10, 52.5),
    (255, 220, 0): (8, 50),
    (255, 240, 0): (6, 47.5),
    (255, 255, 0): (4, 45),
    (255, 255, 59): (3.5, 42.5),
    (72, 255, 70): (3, 40),
    (0, 255, 0): (2.5, 37.5),
    (0, 245, 7): (2, 35),
    (0, 218, 13): (1.5, 32.5),
    (0, 202, 17): (1, 30),
    (0, 183, 41): (0.88, 27.5),
    (0, 162, 53): (0.75, 25),
    (0, 137, 56): (0.63, 22.5),
    (0, 128, 69): (0.5, 20),
    (0, 131, 125): (0.38, 17.5),
    (0, 151, 154): (0.25, 15),
    (0, 186, 191): (0.18, 12.5),
    (0, 209, 213): (0.10, 10),
    (0, 239, 239): (0.08, 7.5),
    (0, 255, 255): (0.05, 5),
}


def loadRadar():
    initial_time = grabRadarSG()
    # Path to your folder with PNG files
    png_folder = "../radImg/"

    # Retrieve all PNG file paths and sort them chronologically
    image_files = sorted(glob.glob(png_folder + "*.png"))

    # Assuming images are recorded every 5 minutes, two hours of data equals 24 + 1 images.

    # If you want the most recent two hours, select the last 25 images:
    n_images = 25
    two_hours_files = image_files[-n_images:]

    # Load images as grayscale arrays (if your PNGs are color-coded you might need to convert using a color-to-rainfall lookup)

    radar_images = []
    rainfall_images = []
    for img in two_hours_files:
        img_array = np.array(Image.open(img).convert("RGB"))
        print(img_array.shape)
        rainfall = np.zeros(
            (img_array.shape[0], img_array.shape[1]), dtype=np.float32()
        )
        radar_img = np.zeros(
            (img_array.shape[0], img_array.shape[1]), dtype=np.float32()
        )

        for (
            color,
            (rain, intensity),
        ) in color_to_rain.items():
            mask = np.all(np.abs(img_array - np.array(color)) <= 10, axis=-1)
            rainfall[mask] = rain
            radar_img[mask] = intensity

        # Convert the rainfall values to grayscale
        rainfall_images.append(rainfall)
        radar_images.append(radar_img)

    # Stack the images into a 3D NumPy array: (time, height, width)
    radar_stack = np.stack(radar_images)
    rainfall_stack = np.stack(rainfall_images)

    # Optional: Display the last frame (the most recent image)

    displayRadar(rainfall_stack, "Observed Rainfall", initial_time)

    return (
        radar_stack,
        rainfall_stack,
        color_to_rain,
        initial_time + timedelta(minutes=120),
    )


if __name__ == "__main__":
    loadRadar()
