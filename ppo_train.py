import numpy as np
import tensorflow as tf

# Import the PPO components we just built
from nav2d.ppo_agent import PPOAgent, PPOBuffer

# Import your Lab3 environment components
from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map


def main():
    # 1. Initialize the Environment Components
    robot = VelRobot(0.5, 0.5)
    env_map = Map()

    # We pass an empty obstacle list initially because our updated
    # engine.reset() function dynamically spawns them anyway.
    env = NavigationEngine(robot=robot, Map=env_map)

    state_dim = env.observation_size
    action_dim = env.action_size

    # 2. PPO Hyperparameters
    steps_per_epoch = 4000  # Number of steps to collect before updating networks
    epochs = 150  # Total number of training epochs
    max_ep_len = 200  # Maximum steps per episode before timeout

    # 3. Instantiate PPO Agent and Memory Buffer
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    buffer = PPOBuffer(size=steps_per_epoch, state_dim=state_dim)

    # 4. Main Training Loop
    obs = env.reset()
    ep_ret, ep_len = 0, 0
    episodes_completed = 0
    total_epoch_rewards = []

    print("Starting PPO Training...")

    for epoch in range(epochs):
        epoch_rewards = []

        for t in range(steps_per_epoch):
            # A. Get action, value, and log probability from the agent
            action, value, logp = agent.get_action(obs)

            # B. Step the environment
            next_obs, reward, done = env.step(action)
            ep_ret += reward
            ep_len += 1

            # C. Store trajectory step in buffer
            buffer.store(obs, action, reward, value, logp)

            # Update current observation
            obs = next_obs

            # D. Check if episode is over or trajectory is cut off
            timeout = (ep_len == max_ep_len)
            terminal = done or timeout
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or epoch_ended:
                # If trajectory didn't reach a terminal state, bootstrap value target
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                if timeout or epoch_ended:
                    _, value, _ = agent.get_action(obs)
                else:
                    value = 0

                buffer.finish_path(value)

                if terminal:
                    epoch_rewards.append(ep_ret)
                    episodes_completed += 1

                # Reset environment for the next episode
                obs, ep_ret, ep_len = env.reset(), 0, 0

        # 5. End of Epoch: Update Networks
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

        print(
            f"Epoch {epoch + 1}/{epochs} | Episodes: {len(epoch_rewards)} | Avg Reward: {avg_reward:.2f} | Actor Loss: {loss_a.numpy():.4f} | Critic Loss: {loss_c.numpy():.4f}")

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            agent.actor.save('ppo_actor_model.keras')

    print("PPO Training Finished!")
    agent.actor.save('ppo_actor_model_final.keras')


if __name__ == '__main__':
    main()