import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from nav2d import config
from nav2d.elements import (
    VelRobot, Map, StaticObstacle, MovingCreature, GoalOrbitingCreature
)


class NavigationEngine(gym.Env):
    # Standard Gym metadata for rendering
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, robot: VelRobot, env_map: Map):
        super().__init__()

        self.screen = None

        # Initialize display if not headless
        pygame.display.set_caption("2D Robot Navigation")
        try:
            icon = pygame.image.load(f"{config.root}/assets/robot.png")
            pygame.display.set_icon(icon)
        except Exception:
            pass

        self.robot = robot
        self.env_map = env_map

        # Dimensions pulled from config
        self.observation_size = config.observation_size
        self.action_size = config.action_size

        # Define Action and Observation Spaces for Gymnasium
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32
        )

        # Internal state
        self.previous_distance = 0.0
        self.obstacle_list = []
        self.static_obstacles = []
        self.moving_creature = None
        self.orbiting_creature = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """Resets the environment and returns the initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Extract training progress (0.0 to 1.0) from options, default to 1.0 (hardest)
        progress = options.get('progress', 1.0) if options else 1.0

        # 1. Spawn Static Obstacles at FIXED positions
        fixed_positions = [(0.3, 0.3), (0.7, 0.7), (0.3, 0.7)]
        self.static_obstacles = [
            StaticObstacle(x, y) for (x, y) in fixed_positions[:config.static_obstacle_count]
        ]

        # 2. Spawn Moving Creatures
        cx, cy = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        self.moving_creature = MovingCreature(cx, cy)

        # We will initialize the orbiting creature after the goal is set
        self.obstacle_list = self.static_obstacles + [self.moving_creature]

        # 3. Curriculum Learning: Safe Robot & Goal Spawning
        valid_spawn = False
        while not valid_spawn:
            valid_spawn = True

            # Re-roll the moving creature so it doesn't trap the fixed robot in an infinite loop
            self.moving_creature.reset()

            # --- Phase A: Kindergarten (Progress < 20%) ---
            if progress < 0.2:
                # Robot in center, Goal very close
                self.robot.reset(x=0.5, y=0.5)
                angle = np.random.uniform(-np.pi, np.pi)
                dist = np.random.uniform(0.1, 0.2)
                gx = np.clip(0.5 + dist * np.cos(angle), 0.1, 0.9)
                gy = np.clip(0.5 + dist * np.sin(angle), 0.1, 0.9)
                self.env_map.goal.reset(x=gx, y=gy)

            # --- Phase B: Middle School (Progress < 50%) ---
            elif progress < 0.5:
                # Robot in safe corner, Goal anywhere valid
                self.robot.reset(x=0.1, y=0.1)
                self.env_map.goal.reset()

            # --- Phase C: High School (Progress >= 50%) ---
            else:
                # Full randomization
                self.robot.reset()
                self.env_map.goal.reset()

            # Check minimum distance to goal (Skip this check for Phase A, which is deliberately close)
            if progress >= 0.2 and np.hypot(self.robot.x - self.env_map.goal.x, self.robot.y - self.env_map.goal.y) < 0.2:
                valid_spawn = False
                continue

            # Check collision with hazards (static + moving)
            for obs in self.obstacle_list:
                if self._check_collision(obs):
                    valid_spawn = False
                    break

            # Also ensure goal doesn't spawn inside an obstacle
            for obs in self.obstacle_list:
                if np.hypot(self.env_map.goal.x - obs.x,
                            self.env_map.goal.y - obs.y) < config.obj_collision_threshold * 2:
                    valid_spawn = False
                    break

        # 4. Finalize Orbiting Creature now that Goal is locked
        self.orbiting_creature = GoalOrbitingCreature(
            self.env_map.goal.x, self.env_map.goal.y
        )
        self.obstacle_list.append(self.orbiting_creature)

        # 5. Initialize reward shaping metrics
        self.previous_distance = np.hypot(
            self.robot.x - self.env_map.goal.x,
            self.robot.y - self.env_map.goal.y
        )

        observation = self.get_state()
        info = {}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one timestep in the environment.
        Returns: observation, reward, terminated, truncated, info (Gym standard)
        """
        # 1. Process Robot Action
        action_space = {0: 'RIGHT', 1: 'LEFT', 2: 'FORWARD', 3: 'SPRINT'}
        act_str = action_space.get(action, 'FORWARD')

        if act_str == 'LEFT':
            self.robot.orient += np.pi / 2
        elif act_str == 'RIGHT':
            self.robot.orient -= np.pi / 2
        elif act_str == 'FORWARD':
            self.robot.move(config.robot_vel_scale * np.cos(self.robot.orient),
                            config.robot_vel_scale * np.sin(self.robot.orient))
        elif act_str == 'SPRINT':
            self.robot.move(2 * config.robot_vel_scale * np.cos(self.robot.orient),
                            2 * config.robot_vel_scale * np.sin(self.robot.orient))

        # Normalize orientation
        self.robot.orient = (self.robot.orient + np.pi) % (2 * np.pi) - np.pi

        # 2. Move dynamic obstacles
        if self.moving_creature:
            self.moving_creature.move()
        if self.orbiting_creature:
            self.orbiting_creature.move(self.env_map.goal.x, self.env_map.goal.y)

        # 3. Check Collisions & Status
        reach_goal, hit_obstacle, hit_wall = self._get_robot_status()

        # 4. Calculate Rewards
        reward = self._calculate_reward(reach_goal, hit_obstacle, hit_wall)

        # 5. Determine if episode ended
        terminated = bool(reach_goal or hit_obstacle or hit_wall)
        truncated = False  # Time limits should be handled by the training loop now

        observation = self.get_state()
        info = {
            "reach_goal": reach_goal,
            "hit_obstacle": hit_obstacle,
            "hit_wall": hit_wall
        }

        return observation, reward, terminated, truncated, info

    def _get_robot_status(self) -> Tuple[bool, bool, bool]:
        hit_wall = (self.robot.x <= 0 or self.robot.x >= 1 or
                    self.robot.y <= 0 or self.robot.y >= 1)

        reach_goal = self._check_collision(self.env_map.goal)

        hit_obstacle = any(self._check_collision(obs) for obs in self.obstacle_list)

        return reach_goal, hit_obstacle, hit_wall

    def _check_collision(self, obj: Any) -> bool:
        dx = abs(self.robot.x - obj.x)
        dy = abs(self.robot.y - obj.y)
        return dx <= config.obj_collision_threshold and dy <= config.obj_collision_threshold

    def _calculate_reward(self, reach_goal: bool, hit_obstacle: bool, hit_wall: bool) -> float:
        # Base sparse rewards
        reward = (reach_goal * config.reach_goal_reward +
                  hit_obstacle * config.hit_obstacle_reward +
                  hit_wall * config.hit_wall_reward +
                  config.step_penalty)

        # Dense Distance Reward
        current_distance = np.hypot(self.robot.x - self.env_map.goal.x, self.robot.y - self.env_map.goal.y)
        distance_delta = self.previous_distance - current_distance
        reward += (distance_delta * config.dense_reward_scale)
        self.previous_distance = current_distance

        # NEW: Dense Alignment Reward (Reward the agent for facing the goal)
        dx = self.env_map.goal.x - self.robot.x
        dy = self.env_map.goal.y - self.robot.y
        angle_to_goal = np.arctan2(dy, dx)

        # Calculate angle difference strictly wrapped between [-pi, pi]
        angle_diff = (angle_to_goal - self.robot.orient + np.pi) % (2 * np.pi) - np.pi

        # cos(angle_diff) is 1.0 when perfectly facing the goal, and -1.0 when facing away.
        # We scale it slightly so it guides behavior without overpowering the main distance/goal rewards.
        alignment_bonus = np.cos(angle_diff) * 0.5
        reward += alignment_bonus

        # Lidar Safety Penalty (Proportional Repulsion Field)
        lidar_data = self.get_lidar_data()
        min_lidar = np.min(lidar_data)

        if min_lidar < config.lidar_penalty_threshold:
            # Calculate how deep into the danger zone the robot is (0.0 to 1.0)
            danger_ratio = 1.0 - (min_lidar / config.lidar_penalty_threshold)

            # Scale the penalty: closer to wall = closer to full penalty
            reward += config.lidar_penalty_value * danger_ratio

        return reward

    def get_state(self) -> np.ndarray:
        # 1. Normalize Lidar to range [0.0, 1.0]
        lidar_dists = self.get_lidar_data() / config.lidar_max_dist

        # 2. Calculate Relative Distance (Normalized to max possible map diagonal)
        dx = self.env_map.goal.x - self.robot.x
        dy = self.env_map.goal.y - self.robot.y
        dist = np.hypot(dx, dy)
        norm_dist = dist / np.sqrt(2.0)  # Max diagonal of a 1x1 map is ~1.414

        # 3. Calculate Relative Angle (Encoded as Sine and Cosine)
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = angle_to_goal - self.robot.orient

        # Wrap angle strictly between [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        sin_a = np.sin(angle_diff)
        cos_a = np.cos(angle_diff)

        # Final State Vector: [distance, sin, cos, ...lidar]
        state = np.concatenate(([norm_dist, sin_a, cos_a], lidar_dists))
        return state.astype(np.float32)

    def get_lidar_data(self) -> np.ndarray:
        """Casts rays to detect walls and obstacles, returning distance array."""
        angles = np.linspace(-np.pi, np.pi, config.lidar_num_rays, endpoint=False) + self.robot.orient
        ray_dx, ray_dy = np.cos(angles), np.sin(angles)
        distances = np.full(config.lidar_num_rays, config.lidar_max_dist)

        rx, ry = self.robot.x, self.robot.y

        # 1. Vectorized Wall Intersections
        ray_dx_safe = np.where(ray_dx == 0, 1e-6, ray_dx)
        ray_dy_safe = np.where(ray_dy == 0, 1e-6, ray_dy)

        dist_x1 = np.where(ray_dx > 0, (1.0 - rx) / ray_dx_safe, np.inf)
        dist_x0 = np.where(ray_dx < 0, (0.0 - rx) / ray_dx_safe, np.inf)
        dist_y1 = np.where(ray_dy > 0, (1.0 - ry) / ray_dy_safe, np.inf)
        dist_y0 = np.where(ray_dy < 0, (0.0 - ry) / ray_dy_safe, np.inf)

        wall_dists = np.minimum.reduce([dist_x1, dist_x0, dist_y1, dist_y0])
        distances = np.minimum(distances, wall_dists)

        # 2. Fully Vectorized Obstacle Intersections (No more Python loops!)
        if self.obstacle_list:
            # Create (N, 2) array of obstacle positions
            obs_positions = np.array([[obs.x, obs.y] for obs in self.obstacle_list])

            # Vectors from robot to obstacles -> Shape: (N, 2)
            vx = obs_positions[:, 0] - rx
            vy = obs_positions[:, 1] - ry

            # Project obstacle vectors onto all 200 rays -> Shape: (N, 200)
            d = np.outer(vx, ray_dx) + np.outer(vy, ray_dy)

            # Calculate orthogonal squared distance to the ray line -> Shape: (N, 200)
            v_sq = (vx ** 2 + vy ** 2).reshape(-1, 1)
            h_sq = v_sq - d ** 2

            radius = config.obj_collision_threshold
            r_sq = radius ** 2

            # Identify which rays hit which obstacles
            hit_mask = (d > 0) & (h_sq < r_sq)

            # Calculate exact distance to the hit surface
            hit_dists = np.full(d.shape, np.inf)
            hit_dists[hit_mask] = d[hit_mask] - np.sqrt(r_sq - h_sq[hit_mask])

            # Collapse the matrix to find the closest obstacle for each ray
            min_obs_dists = np.min(hit_dists, axis=0)
            distances = np.minimum(distances, min_obs_dists)

        return distances

    def render(self) -> np.ndarray:
        """Renders the environment to a Pygame surface and returns the RGB array."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(config.map_size)

        self.screen.fill((155, 255, 255))

        # Render elements
        for obs in self.obstacle_list:
            self.screen.blit(*obs.render_info(scale=config.scale))

        walls, goal = self.env_map.render_info()
        for line in walls:
            pygame.draw.line(self.screen, *line)
        self.screen.blit(*goal)

        robot_image, robot_pos = self.robot.render_info(scale=config.scale)
        robot_image = pygame.transform.rotate(robot_image, self.robot.orient * 180 / np.pi - 90)
        self.screen.blit(robot_image, robot_pos)

        pygame.display.update()
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))