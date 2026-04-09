import numpy as np

from nav2d import config
from nav2d import utils
from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map
from nav2d.dqn_agent import DQNAgent


def main():
    # 1. Initialize the Environment Components
    robot = VelRobot(0.5, 0.5)
    env_map = Map()
    env = NavigationEngine(robot=robot, env_map=env_map)

    # 2. Initialize the DQN Agent (Now inherits from BaseAgent)
    agent = DQNAgent(state_dim=env.observation_size, action_dim=env.action_size)

    # 3. Training Tracking Variables
    total_point_history = []
    total_rewards = 0
    epsilon = 1.0
    previous_phase = 0

    print("Starting DQN Training...")

    # 4. Main Training Loop
    # 4. Main Training Loop
    for ep in range(config.dqn_num_episodes):
        # Calculate curriculum progress and pass to environment
        progress = ep / config.dqn_num_episodes

        # --- Exploration Injection Logic ---
        if progress < 0.2:
            current_phase = 0
        elif progress < 0.5:
            current_phase = 1
        else:
            current_phase = 2

        if current_phase > previous_phase:
            print(f"\n--- Curriculum Phase Upgraded to Phase {current_phase}! Boosting Exploration ---")
            epsilon = max(epsilon, 0.5)  # Boost epsilon to prevent getting stuck in local optima
            previous_phase = current_phase

        state, info = env.reset(options={'progress': progress})
        total_points = 0

        for t in range(config.max_steps_per_episode):
            # A. Get action from the agent (epsilon-greedy)
            action = agent.get_action(state, epsilon)

            # B. Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # C. Store trajectory step in the agent's memory
            agent.store(state, action, reward, next_state, done)

            # D. Learn / Update network if conditions are met
            if utils.check_update_conditions(t, config.dqn_update_every, agent.memory):
                experiences = agent.get_experiences()
                agent.train_step(*experiences)

            state = next_state.copy()
            total_points += reward

            if done:
                break

        # 5. End of Episode Metrics & Updates
        # Exponentially Weighted Moving Average (EWMA) for smooth logging
        total_rewards = total_rewards + config.dqn_soft_upd * (total_points - total_rewards)
        total_point_history.append(total_points)

        # Decay epsilon
        epsilon = utils.get_new_eps(epsilon)

        print(
            f"\rEpisode {ep + 1} | Points: {total_points:.2f} | EWMA Reward: {total_rewards:.2f} | Epsilon: {epsilon:.3f}",
            end="")

        # 6. Periodic Model Saving & Evaluation
        if (ep + 1) % config.dqn_num_p_av == 0:
            av_latest_points = np.mean(total_point_history[-config.dqn_num_p_av:])
            print(f"\rEpisode {ep + 1} | Total Average: {av_latest_points:.2f}")
            agent.save('carnav_model.keras')

            # Early stopping if the environment is considered solved
            if av_latest_points > 800.0 and ep > config.dqn_num_episodes / 2:
                print(f"\n\nEnvironment solved in {ep + 1} episodes!")
                break

    print("\nDQN Training Finished!")
    agent.save('carnav_model_final.keras')

    # Plot the results with the new variance-shaded function
    utils.plot_history(total_point_history, filename='dqn_learning_curve.png')


if __name__ == '__main__':
    main()