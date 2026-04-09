import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import imageio
from tqdm import tqdm
from typing import List, Deque, Any
import os
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


def plot_history(point_history: List[float], filename: str = 'reward_history.png', **kwargs):
    """Plots the reward history over episodes with moving average and variance shading."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    lower_limit = kwargs.get("lower_limit", 0)
    upper_limit = kwargs.get("upper_limit", len(point_history))

    # Default window size is 10% of the data
    window_size = kwargs.get("window_size", max(1, (upper_limit * 10) // 100))

    points = point_history[lower_limit:upper_limit]
    episode_num = list(range(lower_limit, upper_limit))

    # Use Pandas to calculate the rolling mean and standard deviation (variance)
    series = pd.Series(points)
    rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, min_periods=1).std()

    plt.figure(figsize=(12, 8), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("#f4f4f4")
    plt.grid(color='white', linestyle='-', linewidth=2)

    # Plot raw data lightly in the background
    plt.plot(episode_num, points, alpha=0.3, color="cyan", label="Raw Episode Reward")

    # Plot the moving average prominently
    plt.plot(episode_num, rolling_mean, linewidth=2, color="blue", label=f"Moving Average ({window_size} ep)")

    # Fill the variance (Standard Deviation)
    plt.fill_between(
        episode_num,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="blue",
        alpha=0.2,
        label="Variance (±1 Std Dev)"
    )

    text_color = "#333333"
    plt.xlabel("Episode", color=text_color, fontsize=16, fontweight='bold')
    plt.ylabel("Total Reward", color=text_color, fontsize=16, fontweight='bold')
    plt.title("Agent Training Performance", color=text_color, fontsize=20, fontweight='bold')

    yNumFmt = mticker.StrMethodFormatter("{x:,.0f}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="both", colors=text_color, labelsize=12)

    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()

    print(f"Saving learning curve plot to {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close()

def create_video(frames: List[np.ndarray], filename: str, fps: int = 30):
    """Compiles a list of NumPy image arrays into an MP4 video."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(f"Saving video to {filename}...")
    writer_kwargs = {'fps': fps}
    if filename.endswith('.gif'):
        writer_kwargs['loop'] = 0

    with imageio.get_writer(filename, **writer_kwargs) as video:
        for frame in tqdm(frames, desc="Processing Frames"):
            video.append_data(frame)
    print(f"Video {filename} Created Successfully!")