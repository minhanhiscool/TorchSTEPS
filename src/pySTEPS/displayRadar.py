from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from datetime import timedelta

geodata = {
    "projection": "+proj=longlat +datum=WGS84",  # Assuming WGS84 projection
    "x1": 103.3404,  # Longitude of lower-left corner
    "y1": 0.7370,  # Latitude of lower-left corner
    "x2": 104.5987,  # Longitude of upper-right corner
    "y2": 1.9712,  # Latitude of upper-right corner
    "yorigin": "upper",  # Assuming the first element is at the upper border
}


def displayRadar(stack, title, timeS):
    """
    Display radar images in a matplotlib animation

    Args:
        stack (numpy.ndarray): A 3D array of radar images.
        title (str): The title of the figure.
        timeS (datetime): The time of the radar images.
    """

    plt.axis("off")

    for i in stack:
        plt.clf()

        plt.suptitle(title)
        plt.title(timeS)
        plot_precip_field(
            i,
            geodata=geodata,
            axis="off",
            map_kwargs={
                "drawlonlatlines": False,
                "drawlonlatlabels": False,
                "scale": "10m",
            },
        )

        timeS += timedelta(minutes=5)
        plt.pause(0.5)
