from typing import List, Optional, Union, Any
from collections import deque, namedtuple
import time
import numpy as np
import pygame
import tensorflow as tf

# Import the new classes along with the old ones
from .elements import StaticObstacle, MovingCreature, GoalOrbitingCreature
from nav2d import config
from nav2d.elements import (Charger, Map, VelRobot)
from nav2d.utils import denormalize_pos, create_video
from nav2d import config
from nav2d import utils


class NavigationEngine:
    # Generalized obstacle_list typing to handle our new custom classes
    def __init__(self, robot: VelRobot, Map: Map):
        self.screen = None

        pygame.display.set_caption("2D robot navigation")
        icon = pygame.image.load(f"{config.root}/assets/robot.png")

        try:
            pygame.display.set_icon(icon)
        except Exception:
            pass

        self.robot = robot
        self.Map = Map
        # Goal + Obstacles (X, Y) distances
        self.observation_size = 2 + 2 + 200
        # Left, Right, Forward, sprint
        self.action_size = 4

        self.obj_collision_threshold = config.obj_collision_threshold

    def render(self):
        # call other functions before calling it
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(config.map_size)

        # white background
        self.screen.fill((155, 255, 255))

        # plot all obstacles
        for obs in self.obstacle_list:
            # Note: Ensure your new obstacle classes have a render_info() method
            # in elements.py if you plan to use this visualizer!
            if hasattr(obs, 'render_info'):
                self.screen.blit(*obs.render_info(scale=config.scale))

        walls, goal = self.Map.render_info()
        # plot walls
        for line in walls:
            pygame.draw.line(self.screen, *line)

        # plot goal
        self.screen.blit(*goal)

        # plot robot
        robot_image, robot_pos = self.robot.render_info(scale=config.scale)
        robot_image = pygame.transform.rotate(robot_image, self.robot.orient * 180 / np.pi - 90)
        self.screen.blit(robot_image, robot_pos)

        pygame.display.update()
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def get_robot_status(self):
        hit_wall = self.hit_wall()
        reach_goal = self.hit_object(self.Map.goal)
        hit_obstacle = False
        for obs in self.obstacle_list:
            if self.hit_object(obs):
                hit_obstacle = True
                break

        return reach_goal, hit_obstacle, hit_wall

    def hit_wall(self) -> bool:
        if self.robot.x <= 0 or self.robot.x >= 1 or self.robot.y <= 0 or self.robot.y >= 1:
            return True
        else:
            return False

    # Relaxed type hint to Any to accept StaticObstacle, MovingCreature, etc.
    def hit_object(self, obj: Any) -> bool:
        dx = self.robot.x - obj.x
        dy = self.robot.y - obj.y

        if abs(dx) <= self.obj_collision_threshold \
                and abs(dy) <= self.obj_collision_threshold:
            return True

        return False

    def step(self, action: int):
        action_space = {0: 'RIGHT', 1: 'LEFT', 2: 'FORWARD', 3: 'SPRINT'}
        angle = self.robot.orient
        if action_space[action] == 'LEFT':
            self.robot.orient += np.pi / 2
        elif action_space[action] == 'RIGHT':
            self.robot.orient -= np.pi / 2
        elif action_space[action] == 'FORWARD':
            self.robot.move(config.robot_vel_scale * np.cos(angle), \
                            config.robot_vel_scale * np.sin(angle))
        elif action_space[action] == 'SPRINT':
            self.robot.move(2 * config.robot_vel_scale * np.cos(angle), \
                            2 * config.robot_vel_scale * np.sin(angle))

        angle = self.robot.orient
        if angle > 2 * np.pi:
            self.robot.orient -= 2 * np.pi
        elif angle < -2 * np.pi:
            self.robot.orient += 2 * np.pi

        # --- NEW: MOVE THE CREATURES ---
        if hasattr(self, 'moving_creature'):
            self.moving_creature.move()
        if hasattr(self, 'orbiting_creature'):
            self.orbiting_creature.move(self.Map.goal.x, self.Map.goal.y)

        # Rebuild the obstacle list so Lidar and collision detection use fresh coordinates
        if hasattr(self, 'static_obstacles'):
            self.obstacle_list = self.static_obstacles + [self.moving_creature, self.orbiting_creature]
        # -------------------------------

        reach_goal, hit_obstacle, hit_wall = self.get_robot_status()

        # 1. Calculate base sparse rewards
        reward = reach_goal * config.reach_goal_reward + \
                 hit_obstacle * config.hit_obstacle_reward + \
                 hit_wall * config.hit_wall_reward + \
                 config.step_penalty

        # 2. Add Dense Distance Reward (Reward getting closer, penalize moving away)
        current_distance = np.hypot(self.robot.x - self.Map.goal.x, self.robot.y - self.Map.goal.y)
        distance_delta = self.previous_distance - current_distance

        # Multiply by a scaling factor (e.g., 100) so the network actually notices it
        reward += (distance_delta * 100.0)
        self.previous_distance = current_distance  # Update for the next step

        # 3. Add Lidar Safety Penalty (Penalize getting too close to obstacles to prevent scraping)
        # observation array index 4 to 204 contains the lidar data
        lidar_data = self.get_lidar_data()
        min_lidar_dist = np.min(lidar_data)
        if min_lidar_dist < 0.05:  # If an object is within 5% of the map size
            reward -= 2.0  # Give a warning penalty

        done = False
        if reach_goal or hit_obstacle or hit_wall:
            done = True

        observation = self.get_state()

        return observation, reward, done

    def reset(self, pos=None):
        # --- NEW: SPAWN LOGIC ---
        # 1. Randomize Robot and Goal positions
        if not pos:
            self.robot.reset(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        else:
            self.robot.reset(pos[0], pos[1])

        self.Map.goal.x = np.random.uniform(0.1, 0.9)
        self.Map.goal.y = np.random.uniform(0.1, 0.9)

        # Check distance to ensure robot doesn't spawn on top of the goal
        while np.hypot(self.robot.x - self.Map.goal.x, self.robot.y - self.Map.goal.y) < 0.3:
            self.Map.goal.x = np.random.uniform(0.1, 0.9)
            self.Map.goal.y = np.random.uniform(0.1, 0.9)

        # 2. Spawn 3 Static Obstacles
        self.static_obstacles = []
        for _ in range(3):
            ox, oy = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
            self.static_obstacles.append(StaticObstacle(ox, oy))

        # 3. Spawn 1 Random Moving Creature
        cx, cy = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        self.moving_creature = MovingCreature(cx, cy, velocity=0.02)

        # 4. Spawn 1 Goal-Orbiting Creature
        self.orbiting_creature = GoalOrbitingCreature(self.Map.goal.x, self.Map.goal.y, velocity=0.02)

        # Combine all hazards into a single list for the Lidar and collision detection
        self.obstacle_list = self.static_obstacles + [self.moving_creature, self.orbiting_creature]

        # Store initial distance to goal for reward shaping
        self.previous_distance = np.hypot(self.robot.x - self.Map.goal.x, self.robot.y - self.Map.goal.y)

        return self.get_state()

    def compute_loss(self, experiences, gamma, network, target):
        states, actions, rewards, next_states, done_vals = experiences
        max_qsa = tf.reduce_max(target(next_states), axis=-1)
        y_targets = rewards * done_vals + (1 - done_vals) * (rewards + gamma * max_qsa)
        q_values = network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
        loss = tf.keras.losses.MSE(y_targets, q_values)
        return loss

    @tf.function
    def agent_learn(self, experiences, gamma, network, target, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, gamma, network, target)

        gradients = tape.gradient(loss, network.trainable_variables)

        optimizer.apply_gradients(zip(gradients, network.trainable_variables))

        utils.update_target_network(network, target)
        return loss

    def get_lidar_data(self):
        max_dist = 0.20  # 20% of the normalized map
        num_rays = 200
        # 360 degrees spread, relative to robot's current orientation
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False) + self.robot.orient

        rx, ry = self.robot.x, self.robot.y
        ray_dx = np.cos(angles)
        ray_dy = np.sin(angles)

        distances = np.full(num_rays, max_dist)

        # --- 1. Wall Intersections ---
        # Safe division to prevent division by zero warnings
        ray_dx_safe = np.where(ray_dx == 0, 1e-6, ray_dx)
        ray_dy_safe = np.where(ray_dy == 0, 1e-6, ray_dy)

        # Calculate distance to intersection with the 4 walls (x=0, x=1, y=0, y=1)
        dist_x1 = np.where(ray_dx > 0, (1.0 - rx) / ray_dx_safe, np.inf)
        dist_x0 = np.where(ray_dx < 0, (0.0 - rx) / ray_dx_safe, np.inf)
        dist_y1 = np.where(ray_dy > 0, (1.0 - ry) / ray_dy_safe, np.inf)
        dist_y0 = np.where(ray_dy < 0, (0.0 - ry) / ray_dy_safe, np.inf)

        wall_dists = np.minimum.reduce([dist_x1, dist_x0, dist_y1, dist_y0])
        distances = np.minimum(distances, wall_dists)

        # --- 2. Obstacle Intersections ---
        radius = self.obj_collision_threshold  # The collision radius of obstacles

        for obs in self.obstacle_list:
            ox, oy = obs.x, obs.y
            vx, vy = ox - rx, oy - ry

            # Projection of vector to obstacle onto the ray vector
            d = vx * ray_dx + vy * ray_dy

            # Distance squared from obstacle center to ray line
            v_sq = vx ** 2 + vy ** 2
            h_sq = v_sq - d ** 2

            # Check if ray points towards obstacle (d>0) and intersects its radius
            hit_mask = (d > 0) & (h_sq < radius ** 2)

            hit_dists = np.full(num_rays, np.inf)
            # Pythagorean theorem to find the exact boundary intersection distance
            hit_dists[hit_mask] = d[hit_mask] - np.sqrt(radius ** 2 - h_sq[hit_mask])

            distances = np.minimum(distances, hit_dists)

        return distances

    def get_state(self):
        lidar_dists = self.get_lidar_data()
        robot_pos = np.array([self.robot.x, self.robot.y])
        goal_pos = np.array([self.Map.goal.x, self.Map.goal.y])
        return np.concatenate((robot_pos, goal_pos, lidar_dists))

    def run(self):

        state_size = self.observation_size
        num_actions = self.action_size
        num_episodes = 1000
        max_num_steps = 200
        epsilon = 1
        MEMORY_SIZE = 100000  # size of memory buffer
        GAMMA = 0.995  # discount factor
        NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
        num_p_av = 100
        soft_upd = .01
        total_point_history = []
        total_rewards = 0
        self.obj_collision_threshold = .02

        # Create the Q-Network with larger layers to handle the 204-dimension input
        q_network = tf.keras.Sequential([
            tf.keras.Input(shape=(state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(int(num_actions), activation='linear')
        ])

        # Create the target Q-Network (must match exactly)
        target_q_network = tf.keras.Sequential([
            tf.keras.Input(shape=(state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(int(num_actions), activation='linear')
        ])

        optimizer = tf.keras.optimizers.Adam()

        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        memory_buffer = deque(maxlen=MEMORY_SIZE)

        target_q_network.set_weights(q_network.get_weights())

        for i in range(num_episodes):

            state = self.reset()
            total_points = 0

            for t in range(max_num_steps):

                state_qn = np.expand_dims(state, axis=0)
                q_values = q_network(state_qn)
                action = utils.get_action(q_values, epsilon)
                next_state, reward, done = self.step(action)

                memory_buffer.append(experience(state, action, reward, next_state, done))

                update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

                if update:
                    experiences = utils.get_experiences(memory_buffer)

                    self.agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer)

                state = next_state.copy()
                total_points += reward

                if done:
                    break

            total_rewards = total_rewards + soft_upd * (total_points - total_rewards)
            total_point_history.append(total_rewards)
            av_latest_points = np.mean(total_point_history[-num_p_av:])

            epsilon = utils.get_new_eps(epsilon)

            print(f"\rEpisode {i + 1} | Average Reward: {total_rewards:.2f}", end="")

            if (i + 1) % num_p_av == 0:
                print(f"\rEpisode {i + 1} | Total Average: {av_latest_points:.2f}")
                q_network.save('carnav_model.keras')

            if av_latest_points > 80.0 and i > num_episodes / 2:
                print(f"\n\nEnvironment solved in {i + 1} episodes!")
                break

        utils.plot_history(total_point_history, plot_data_only=True)