import math
import random
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import pygame
import imageio
import time
from tqdm import tqdm

from nav2d import config

SEED = 1  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy (900 steps to reach .01 for e=.995).
E_MIN = 0.05  # Minimum ε value for the ε-greedy policy.


def normalize_pos(pos: np.ndarray) -> np.ndarray:
    map_size = np.array(config.map_size)
    center = map_size / 2
    radius = map_size - center

    return (pos - center) / radius


def denormalize_pos(pos: np.ndarray) -> np.ndarray:
    map_size = np.array(config.map_size)
    center = map_size / 2
    radius = map_size - center

    return center + pos * radius

def update_target_network(q_network, target_q_network):

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

def get_action(Q, epsilon=0.0):
    # epsilon-greedy selection {0: 'RIGHT', 1:'LEFT', 2:'FORWARD', 3:'SPRINT'}
    rand = np.random.rand()
    if rand < epsilon:
        return np.random.randint(0, 4, 1)[0]
    else:
        return np.argmax(Q.numpy()[0])

def get_new_eps(epsilon):

    return max(E_MIN, E_DECAY * epsilon)

def get_experiences(memory_buffer):

    experiences = random.sample(memory_buffer, MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def plot_history(point_history, **kwargs):
    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # Generate x-axis for plotting.
    episode_num = [x for x in range(lower_limit, upper_limit)]

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

def create_video(frames, filename, fps=30):
    print(f"Saving video to {filename}...")
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in tqdm(frames, desc="Processing Frames"):
            video.append_data(frame)
    print(f"Video {filename} Created Successfully!")
