import requests
import random
from datetime import datetime, timedelta
import os


def round_down_to_nearest_5(dt):
    """
    Rounds down the given datetime to the nearest 5 minutes

    Args:
        dt (datetime): The datetime to round down.

    Returns:
        datetime: The rounded down datetime.
    """

    dt = dt.replace(second=0, microsecond=0)
    minute = dt.minute - (dt.minute % 5)
    return dt.replace(minute=minute)


def grabRadarSG():
    """
    Grabs the 70km radar images from weather.gov.sg and saves them to the
    local directory

    Returns:
        datetime: The current time rounded down to the nearest 5 minutes
    """

    current_time = datetime.now()

    max_seconds = (
        (9 * 24 + 22) * 60 * 60
    )  # Max time difference between current time and start time, which is 9 days and 22 hours

    current_time = round_down_to_nearest_5(
        current_time - timedelta(seconds=random.randint(0, max_seconds))
    )  # Randomly choose a time between now and 9 days and 22 hours ago

    print(f"Current rounded time: {current_time}")

    # Total images for two hours (images every 5 minutes)
    n_images = 24

    # The most recent image is at current_time,
    # so the start time is 23 intervals earlier (i.e. 2345 = 120 minutes ago)
    start_time = current_time - timedelta(minutes=120)
    print(f"Start time: {start_time}")

    # Base URL pattern for the 70km radar images
    base_url = "https://www.weather.gov.sg/files/rainarea/50km/v3/"

    # Directory to save downloaded images
    save_dir = "../radImg"
    os.makedirs(save_dir, exist_ok=True)

    current_download_time = start_time
    for i in range(n_images + 1):
        # Construct timestamp and filename according to the pattern
        # Pattern: dpsri_70km_YYYYmmddHHMMSS000dBR.dpsri.png
        timestamp = current_download_time.strftime("%Y%m%d%H%M%S")
        filename = f"dpsri_70km_{timestamp}00dBR.dpsri.png"
        file_url = base_url + filename
        print(f"Downloading {file_url} ...")

        # Headers to bypass blocked requests. Seriously why tho???
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0"
        }

        response = requests.get(file_url, headers=headers)
        if response.status_code == 200:
            # Success = save file
            with open(os.path.join(save_dir, filename), "wb") as f:
                f.write(response.content)
            print(f"Saved: {filename}")
        else:
            # Failed then print a warning
            print(
                f"Failed to download {file_url} (status code: {response.status_code})"
            )

        # Increment time by 5 minutes for the next image
        current_download_time += timedelta(minutes=5)

    return start_time


if __name__ == "__main__":
    grabRadarSG()
