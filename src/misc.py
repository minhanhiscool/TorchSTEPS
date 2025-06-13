import torch.nn.functional as F
import random
from datetime import datetime, timedelta


def all_time():
    """
    Produces a list of all times between 2025-04-16 14:30 and 2025-04-26 12:30

    Returns:
        List[datetime]: List of all times
    """
    start_time = datetime(2025, 4, 16, 14, 30)
    end_time = datetime(2025, 4, 26, 12, 30)

    total_minutes = int(end_time.timestamp() - start_time.timestamp()) // 60
    all_time_list = [
        start_time + timedelta(minutes=i * 5) for i in range(total_minutes // 5 + 1)
    ]

    random.shuffle(all_time_list)
    return all_time_list
