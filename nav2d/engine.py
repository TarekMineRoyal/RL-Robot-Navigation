import numpy as np
import pygame
from typing import Tuple, Dict, Any

from nav2d import config
from nav2d.elements import (
    VelRobot, Map, StaticObstacle, MovingCreature, GoalOrbitingCreature
)


class NavigationEngine:
    def __init__(self, robot: VelRobot, env_map: Map):
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

        # Internal state
        self.previous_distance = 0.0
        self.obstacle_list = []
        self.static_obstacles = []
        self.moving_creature = None
        self.orbiting_creature = None

    def reset(self, pos: Tuple[float, float] = None) -> np.ndarray:
        """Resets the environment and returns the initial observation."""
        # 1. Spawn Robot and Goal
        if not pos:
            self.robot.reset()
        else:
            self.robot.reset(pos[0], pos[1])

        self.env_map.goal.reset()

        # Ensure robot doesn't spawn on top of the goal
        while np.hypot(self.robot.x - self.env_map.goal.x, self.robot.y - self.env_map.goal.y) < 0.3:
            self.env_map.goal.reset()

        # 2. Spawn Static Obstacles
        self.static_obstacles = [
            StaticObstacle(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            for _ in range(config.static_obstacle_count)
        ]

        # 3. Spawn Moving Creatures
        cx, cy = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        self.moving_creature = MovingCreature(cx, cy)

        self.orbiting_creature = GoalOrbitingCreature(
            self.env_map.goal.x, self.env_map.goal.y
        )

        # 4. Consolidate hazards
        self.obstacle_list = self.static_obstacles + [self.moving_creature, self.orbiting_creature]

        # 5. Initialize reward shaping metrics
        self.previous_distance = np.hypot(
            self.robot.x - self.env_map.goal.x,
            self.robot.y - self.env_map.goal.y
        )

        return self.get_state()

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

        # Lidar Safety Penalty
        lidar_data = self.get_lidar_data()
        if np.min(lidar_data) < config.lidar_penalty_threshold:
            reward += config.lidar_penalty_value

        return reward

    def get_state(self) -> np.ndarray:
        lidar_dists = self.get_lidar_data()
        robot_pos = np.array([self.robot.x, self.robot.y])
        goal_pos = np.array([self.env_map.goal.x, self.env_map.goal.y])
        return np.concatenate((robot_pos, goal_pos, lidar_dists))

    def get_lidar_data(self) -> np.ndarray:
        """Casts rays to detect walls and obstacles, returning distance array."""
        angles = np.linspace(-np.pi, np.pi, config.lidar_num_rays, endpoint=False) + self.robot.orient
        ray_dx, ray_dy = np.cos(angles), np.sin(angles)
        distances = np.full(config.lidar_num_rays, config.lidar_max_dist)

        rx, ry = self.robot.x, self.robot.y

        # 1. Wall Intersections
        ray_dx_safe = np.where(ray_dx == 0, 1e-6, ray_dx)
        ray_dy_safe = np.where(ray_dy == 0, 1e-6, ray_dy)

        dist_x1 = np.where(ray_dx > 0, (1.0 - rx) / ray_dx_safe, np.inf)
        dist_x0 = np.where(ray_dx < 0, (0.0 - rx) / ray_dx_safe, np.inf)
        dist_y1 = np.where(ray_dy > 0, (1.0 - ry) / ray_dy_safe, np.inf)
        dist_y0 = np.where(ray_dy < 0, (0.0 - ry) / ray_dy_safe, np.inf)

        wall_dists = np.minimum.reduce([dist_x1, dist_x0, dist_y1, dist_y0])
        distances = np.minimum(distances, wall_dists)

        # 2. Obstacle Intersections
        radius = config.obj_collision_threshold

        for obs in self.obstacle_list:
            vx, vy = obs.x - rx, obs.y - ry
            d = vx * ray_dx + vy * ray_dy  # Projection

            h_sq = (vx ** 2 + vy ** 2) - d ** 2  # Distance squared to ray line
            hit_mask = (d > 0) & (h_sq < radius ** 2)

            hit_dists = np.full(config.lidar_num_rays, np.inf)
            hit_dists[hit_mask] = d[hit_mask] - np.sqrt(radius ** 2 - h_sq[hit_mask])

            distances = np.minimum(distances, hit_dists)

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