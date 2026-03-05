import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import imageio
from tqdm import tqdm
from typing import List, Deque, Any

from nav2d import config


def normalize_pos(pos: np.ndarray) -> np.ndarray:
    """Normalizes a position based on the map scale."""
    map_size = np.array(config.map_size)
    center = map_size / 2
    radius = map_size - center
    return (pos - center) / radius


def denormalize_pos(pos: np.ndarray) -> np.ndarray:
    """Converts a normalized position back to map scale."""
    map_size = np.array(config.map_size)
    center = map_size / 2
    radius = map_size - center
    return center + pos * radius


def get_new_eps(epsilon: float, decay: float = 0.995, min_eps: float = 0.05) -> float:
    """Decays the epsilon value for epsilon-greedy exploration."""
    return max(min_eps, decay * epsilon)


def check_update_conditions(t: int, num_steps_upd: int, memory_buffer: Deque[Any], batch_size: int = 64) -> bool:
    """Checks if it's time to update the network based on steps and memory size."""
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > batch_size:
        return True
    return False


def plot_history(point_history: List[float], **kwargs):
    """Plots the reward history over episodes and saves it as a PNG."""
    lower_limit = kwargs.get("lower_limit", 0)
    upper_limit = kwargs.get("upper_limit", len(point_history))

    # Default window size is 10% of the data
    window_size = kwargs.get("window_size", max(1, (upper_limit * 10) // 100))

    plot_rolling_mean_only = kwargs.get("plot_rolling_mean_only", False)
    plot_data_only = kwargs.get("plot_data_only", False)

    points = point_history[lower_limit:upper_limit]
    episode_num = list(range(lower_limit, upper_limit))

    # Use Pandas to calculate the rolling mean (moving average).
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"
    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)

    yNumFmt = mticker.StrMethodFormatter("{x:,}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)

    plt.savefig('reward_history.png')
    plt.close()  # Free up memory


def create_video(frames: List[np.ndarray], filename: str, fps: int = 30):
    """Compiles a list of NumPy image arrays into an MP4 video."""
    print(f"Saving video to {filename}...")
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in tqdm(frames, desc="Processing Frames"):
            video.append_data(frame)
    print(f"Video {filename} Created Successfully!")