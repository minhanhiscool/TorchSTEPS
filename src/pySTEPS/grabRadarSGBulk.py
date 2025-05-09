# ----------------------------------------------------------------------------
# # This module downloads/processes data from NEA (https://www.nea.gov.sg).
# As per NEA Terms of Use, this data may only be used for non-commercial,
# personal or research purposes. Do NOT redistribute NEA data.
# ----------------------------------------------------------------------------

import requests
import time
import random
from datetime import datetime, timedelta
import os
from tqdm import tqdm

MAX_RETRIES = 100


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


def safe_request(url, headers, retries=MAX_RETRIES, delay=1):
    """
    Handles safe requests to a given URL. If the request fails, it retries
    the request a specified number of times with a specified delay between
    each retry.

    Args:
        url (str): The URL to fetch.
        headers (dict): The headers to include in the request.
        retries (int, optional): The number of retries. Defaults to 5.
        delay (int, optional): The delay between retries. Defaults to 1.

    Returns:
        requests.Response: The response object if successful, None otherwise.
    """
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                print("❌ File not found on server")
                return None
            else:
                print(f"⚠️ Unexpected status code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(
                f"❌ Error fetching URL: {url}. Retrying in {delay} seconds. Error: {e}"
            )
        time.sleep(delay + random.random())
    return None


def grabRadarSG():
    """
    Grabs the 70km radar images from weather.gov.sg and saves them to the
    local directory

    Returns:
        datetime: The current time rounded down to the nearest 5 minutes
    """

    current_time = round_down_to_nearest_5(datetime.now())

    print(f"Current rounded time: {current_time}")

    # Total images for two hours (images every 5 minutes)
    n_images = 2880

    # The most recent image is at current_time,
    # so the start time is 23 intervals earlier (i.e. 2345 = 120 minutes ago)
    start_time = current_time - timedelta(days=10)
    print(f"Start time: {start_time}")

    # Base URL pattern for the 70km radar images
    base_url = "https://www.weather.gov.sg/files/rainarea/50km/v3/"

    # Directory to save downloaded images
    save_dir = "../radImg"
    os.makedirs(save_dir, exist_ok=True)

    current_download_time = start_time
    for i in tqdm(range(n_images + 1), desc="Downloading radar frames", unit="img"):
        # Construct timestamp and filename according to the pattern
        # Pattern: dpsri_70km_YYYYmmddHHMMSS000dBR.dpsri.png
        timestamp = current_download_time.strftime("%Y%m%d%H%M%S")
        filename = f"dpsri_70km_{timestamp}00dBR.dpsri.png"
        file_url = base_url + filename

        # Headers to bypass blocked requests. Seriously why tho???
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0"
        }

        response = safe_request(file_url, headers, retries=5, delay=1)

        if response:
            # Success = save file
            with open(os.path.join(save_dir, filename), "wb") as f:
                f.write(response.content)

        else:
            print(f"Failed to download image {filename}")

        # Increment time by 5 minutes for the next image
        current_download_time += timedelta(minutes=5)

    return start_time


if __name__ == "__main__":
    grabRadarSG()
