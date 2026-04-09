import numpy as np
import tensorflow as tf
import os

from nav2d import config
from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map
from nav2d.utils import create_video

from nav2d.dqn_agent import DQNAgent
from nav2d.ppo_agent import PPOAgent


def evaluate_agent(model_path: str, agent_type: str = 'dqn', num_episodes: int = 3, video_name: str = "eval_video.mp4"):
    print(f"--- Evaluating {agent_type.upper()} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the agent first.")
        return

    # 1. Initialize Gymnasium Environment
    robot = VelRobot(0.5, 0.5)
    env_map = Map()
    env = NavigationEngine(robot=robot, env_map=env_map)

    # 2. Instantiate and Load the Agent
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim=env.observation_size, action_dim=env.action_size)
        # Epsilon 0.0 forces purely greedy exploitation
        get_action = lambda obs: agent.get_action(obs, epsilon=0.0)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim=env.observation_size, action_dim=env.action_size)
        # For PPO evaluation, we take the most probable action instead of sampling
        get_action = lambda obs: int(np.argmax(agent.actor(obs.reshape(1, -1)).numpy()[0]))
    else:
        raise ValueError("agent_type must be 'dqn' or 'ppo'")

    print(f"Loading weights from {model_path}...")
    agent.load(model_path)

    frames = []

    # 3. Evaluation Loop
    for ep in range(num_episodes):
        # We pass progress: 1.0 to ensure High School mode (full randomization)
        obs, info = env.reset(options={'progress': 1.0})
        done = False
        steps = 0
        total_reward = 0

        print(f"Starting Episode {ep + 1}...")

        while not done and steps < config.max_steps_per_episode:
            # Capture High-Res Frame for Video
            frames.append(env.render())

            # Get action from the loaded agent
            action = get_action(obs)

            # Step the Gym environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs = next_obs
            total_reward += reward
            steps += 1

        print(f"Episode {ep + 1} Finished! Steps: {steps}, Total Reward: {total_reward:.2f}")

    # 4. Save the Video (Use fps=30 for smooth playback)
    create_video(frames, video_name, fps=30)
    print("\n")


if __name__ == '__main__':
    # Make sure you have trained your agents and these files exist!
    evaluate_agent('carnav_model_final.keras', agent_type='dqn', video_name='dqn_eval_video.mp4')
    evaluate_agent('ppo_actor_model_final.keras', agent_type='ppo', video_name='ppo_eval_video.mp4')