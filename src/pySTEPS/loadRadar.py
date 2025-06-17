import glob
import sys
from datetime import timedelta
from PIL import Image
import numpy as np
from pySTEPS.displayRadar import displayRadar
from skimage.transform import resize


# Maps color (RGB) in radar to rainfall intensity (mm/hr and dBR)
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

COLOR_THRESHOLD = 10


def generateExpectedFilenames(initial_time, n_images=25, step_minutes=5):
    """
    Generate the expected filenames given different initial time and number of images

    Args:
        initial_time (datetime): The initial time of the radar data.
        n_images (int, optional): The number of images to generate. Defaults to 25.
        step_minutes (int, optional): The step in minutes between each image. Defaults to 5.

    Returns:
        list: A list of expected filenames.
    """

    images = []
    for i in range(n_images):
        image_time = initial_time + timedelta(minutes=i * step_minutes)
        image_filename = (
            f"dpsri_70km_{image_time.strftime('%Y%m%d%H%M%S')}00dBR.dpsri.png"
        )
        images.append(image_filename)
    return images


def broadcastColor(img):
    colors = np.array(list(color_to_rain.keys()), dtype=np.uint8)
    rain_vals = np.array([v[0] for v in color_to_rain.values()], float)
    int_vals = np.array([v[1] for v in color_to_rain.values()], float)

    h, w = img.shape[:2]
    pix = img.reshape(-1, 3)

    is_black = np.all(pix == 0, axis=1)

    diff = np.abs(pix[:, None, :] - colors[None, :, :])
    match = np.all(diff <= COLOR_THRESHOLD, axis=2)

    match[is_black] = False

    any_match = match.any(axis=1)
    idx = match.argmax(axis=1)

    rain_flat = np.zeros(pix.shape[0], dtype=np.float32)
    radar_flat = np.zeros(pix.shape[0], dtype=np.float32)

    valid = any_match
    rain_flat[valid] = rain_vals[idx[valid]]
    radar_flat[valid] = int_vals[idx[valid]]

    rainfall = rain_flat.reshape(h, w)
    radar_img = radar_flat.reshape(h, w)

    return rainfall, radar_img


def loadRadar(initial_time, n_images=6):
    """
    Load radar images from local directory and converts them to a numpy array
    Images are pulled using grabRadarSG as PNG files.

    Args:
        initial_time (datetime): The initial time of the radar data.
        n_images (int, optional): The number of images to load. Defaults to 8.

    Returns:
        tuple: a tuple containing:
            - np.ndarray: A 2D array of radar images.
            - np.ndarray: A 2D array of rainfall images.
            - datetime: The time of the radar images.

    """
    # Path to folder with radImg files
    venv_path = sys.prefix
    png_folder = venv_path + "/../radImg/"

    # Generate expected filenames of each radar imahe
    images_filenames = generateExpectedFilenames(initial_time, n_images, step_minutes=5)

    radar_images = []
    rainfall_images = []

    # Loop through the images and convert them to numpy arrays
    for img in images_filenames:
        full_path = png_folder + img

        # If file not found, print a warning
        if glob.glob(full_path) == []:
            print("!!! WARNING: Image not found: ", full_path)
            continue

        # Convert image to an 3D numpy array
        img_array = np.array(Image.open(full_path).convert("RGB"))

        rainfall, radar_img = broadcastColor(img_array)

        rainfall_resize = resize(
            rainfall,
            (512, 512),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        radar_img_resize = resize(
            radar_img,
            (512, 512),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        rainfall_images.append(rainfall_resize)
        radar_images.append(radar_img_resize)

    # Stack the images into a 3D NumPy array: (time, height, width)
    radar_stack = np.stack(radar_images)
    rainfall_stack = np.stack(rainfall_images)

    # Optional: Display the last frame (the most recent image)
    # displayRadar(rainfall_stack, "Observed Rainfall", initial_time)

    return (
        radar_stack,
        rainfall_stack,
        initial_time + timedelta(minutes=n_images * 5),
    )


if __name__ == "__main__":
    loadRadar()
