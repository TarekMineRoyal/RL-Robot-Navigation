import numpy as np
import tensorflow as tf

from nav2d import config
from nav2d.ppo_agent import PPOAgent, PPOBuffer
from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map


def main():
    # 1. Initialize the Environment Components
    robot = VelRobot(0.5, 0.5)
    env_map = Map()
    env = NavigationEngine(robot=robot, env_map=env_map)

    state_dim = env.observation_size
    action_dim = env.action_size

    # 2. Instantiate PPO Agent and Memory Buffer using Config
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    # We will assume you update ppo_agent.py later to pull its inner hyperparams from config too!
    buffer = PPOBuffer(size=config.ppo_steps_per_epoch, state_dim=state_dim)

    # 3. Main Training Loop
    obs, info = env.reset(options={'progress': 0.0})
    ep_ret, ep_len = 0, 0
    episodes_completed = 0
    total_epoch_rewards = []

    print("Starting PPO Training...")

    for epoch in range(config.ppo_epochs):
        epoch_rewards = []

        for t in range(config.ppo_steps_per_epoch):
            # A. Get action, value, and log probability from the agent
            action, value, logp = agent.get_action(obs)

            # B. Step the environment (Modernized Gym signature)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_ret += reward
            ep_len += 1

            # C. Store trajectory step in buffer
            buffer.store(obs, action, reward, value, logp)

            # Update current observation
            obs = next_obs

            # D. Check if episode is over or trajectory is cut off
            timeout = (ep_len == config.max_steps_per_episode)
            terminal = done or timeout
            epoch_ended = (t == config.ppo_steps_per_epoch - 1)

            if terminal or epoch_ended:
                # If trajectory didn't reach a terminal state, bootstrap value target
                if epoch_ended and not terminal:
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps.', flush=True)

                if timeout or epoch_ended:
                    _, value, _ = agent.get_action(obs)
                else:
                    value = 0

                buffer.finish_path(value)

                if terminal:
                    epoch_rewards.append(ep_ret)
                    episodes_completed += 1

                    # Reset environment for the next episode
                    progress = epoch / config.ppo_epochs
                    obs, info = env.reset(options={'progress': progress})
                    ep_ret, ep_len = 0, 0

        # 4. End of Epoch: Update Networks
        # Retrieve the accumulated batch of trajectories
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buffer.get()

        # Convert numpy arrays to TensorFlow tensors
        obs_tensor = tf.convert_to_tensor(obs_buf, dtype=tf.float32)
        act_tensor = tf.convert_to_tensor(act_buf, dtype=tf.int32)
        adv_tensor = tf.convert_to_tensor(adv_buf, dtype=tf.float32)
        ret_tensor = tf.convert_to_tensor(ret_buf, dtype=tf.float32)
        logp_tensor = tf.convert_to_tensor(logp_buf, dtype=tf.float32)

        # Perform multiple steps of gradient descent on the collected batch
        for _ in range(agent.train_iters):
            loss_a, loss_c = agent.train_step(obs_tensor, act_tensor, adv_tensor, ret_tensor, logp_tensor)

        # Calculate Average Reward for logging
        avg_reward = np.mean(epoch_rewards) if len(epoch_rewards) > 0 else 0
        total_epoch_rewards.append(avg_reward)

        print(f"Epoch {epoch + 1}/{config.ppo_epochs} | Episodes: {len(epoch_rewards)} | "
              f"Avg Reward: {avg_reward:.2f} | Actor Loss: {loss_a.numpy():.4f} | Critic Loss: {loss_c.numpy():.4f}")

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            agent.actor.save('ppo_actor_model.keras')

    print("PPO Training Finished!")
    agent.actor.save('ppo_actor_model_final.keras')


if __name__ == '__main__':
    main()