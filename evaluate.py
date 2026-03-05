import numpy as np
import tensorflow as tf
from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map
from nav2d.utils import create_video


def evaluate_agent(model_path, agent_type='dqn', num_episodes=3, video_name="eval_video.mp4"):
    print(f"Loading {agent_type.upper()} model from {model_path}...")

    # Load the trained keras model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize environment
    robot = VelRobot(0.5, 0.5)
    env_map = Map()
    env = NavigationEngine(robot=robot, Map=env_map)

    frames = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0

        print(f"Starting Episode {ep + 1}...")

        while not done and steps < 300:  # 300 step timeout limit
            # Capture frame for video
            frames.append(env.render())

            # Reshape observation for the neural network
            obs_reshaped = np.expand_dims(obs, axis=0)

            if agent_type == 'dqn':
                # DQN: Get Q-values and pick the action with the highest Q-value (no epsilon exploration)
                q_values = model(obs_reshaped).numpy()
                action = np.argmax(q_values[0])
            elif agent_type == 'ppo':
                # PPO Actor: Get action probabilities and pick the most likely action (deterministic)
                probs = model(obs_reshaped).numpy()[0]
                action = np.argmax(probs)

            # Step the environment
            obs, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {ep + 1} Finished! Steps: {steps}, Total Reward: {total_reward:.2f}")

    print(f"Evaluation complete. Generating video: {video_name}")
    # Utilize the existing create_video function from nav2d.utils
    create_video(frames, video_name)
    print("Video saved successfully!")


if __name__ == '__main__':
    # --- EVALUATE DQN ---
    # Make sure 'carnav_model.keras' exists in your directory from the DQN training
    evaluate_agent('carnav_model.keras', agent_type='dqn', video_name='dqn_navigation.mp4')

    # --- EVALUATE PPO ---
    # Make sure 'ppo_actor_model.keras' exists from the PPO training
    # evaluate_agent('ppo_actor_model.keras', agent_type='ppo', video_name='ppo_navigation.mp4')