import copy
from typing import List, Tuple

import pygame

from nav2d import config

import numpy as np

import numpy as np
import pygame


class StaticObstacle:
    def __init__(self, x, y, radius=0.05):
        self.x = x
        self.y = y
        self.radius = radius

    def render_info(self, scale):
        size = int(self.radius * scale * 2)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 0, 0), (size // 2, size // 2), size // 2)  # Red
        return surf, (int((self.x - self.radius) * scale), int((self.y - self.radius) * scale))


class MovingCreature:
    def __init__(self, x, y, radius=0.05, velocity=0.02):
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity = velocity
        self.angle = np.random.uniform(-np.pi, np.pi)

    def move(self):
        next_x = self.x + self.velocity * np.cos(self.angle)
        next_y = self.y + self.velocity * np.sin(self.angle)

        if next_x <= 0 or next_x >= 1.0:
            self.angle = np.pi - self.angle
            next_x = np.clip(next_x, 0.01, 0.99)
        if next_y <= 0 or next_y >= 1.0:
            self.angle = -self.angle
            next_y = np.clip(next_y, 0.01, 0.99)

        self.x = next_x
        self.y = next_y

    def render_info(self, scale):
        size = int(self.radius * scale * 2)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 165, 0), (size // 2, size // 2), size // 2)  # Orange
        return surf, (int((self.x - self.radius) * scale), int((self.y - self.radius) * scale))


class GoalOrbitingCreature:
    def __init__(self, goal_x, goal_y, radius=0.05, orbit_radius=0.15, velocity=0.02):
        self.radius = radius
        self.orbit_radius = orbit_radius
        self.velocity = velocity
        self.orbit_angle = np.random.uniform(-np.pi, np.pi)
        self.x = goal_x + self.orbit_radius * np.cos(self.orbit_angle)
        self.y = goal_y + self.orbit_radius * np.sin(self.orbit_angle)

    def move(self, goal_x, goal_y):
        angular_velocity = self.velocity / self.orbit_radius
        self.orbit_angle += angular_velocity
        self.orbit_angle = (self.orbit_angle + np.pi) % (2 * np.pi) - np.pi
        self.x = goal_x + self.orbit_radius * np.cos(self.orbit_angle)
        self.y = goal_y + self.orbit_radius * np.sin(self.orbit_angle)

    def render_info(self, scale):
        size = int(self.radius * scale * 2)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (128, 0, 128), (size // 2, size // 2), size // 2)  # Purple
        return surf, (int((self.x - self.radius) * scale), int((self.y - self.radius) * scale))

class ObjectBase:
    def __init__(self, image_path: str, shape: np.ndarray, init_x: float, init_y: float):
        self.image = pygame.image.load(image_path)
        self.shape = shape
        self.image = pygame.transform.scale(self.image, tuple(self.shape))

        self.x = init_x
        self.y = init_y
        self.orient = np.pi/2
        self.init_x = init_x
        self.init_y = init_y
        self.prev_x = init_x
        self.prev_y = init_y

        self.kept_heading = None

    def center(self):
        return int(self.x), int(self.y)

    def accurate_center(self):
        return self.x, self.y

    def heading(self):
        if self.kept_heading is not None:
            return self.kept_heading

        if self.prev_x == self.x and self.prev_y == self.y:
            return 0

        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        angle = np.arctan2(dy, dx)
        return angle

    def render_info(self, scale):
        return self.image, (int(scale*self.x - self.shape[0] / 2), config.map_size[1] - int(scale*self.y - self.shape[1] / 2))

    def move(self, dx, dy) -> np.ndarray:
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += dx
        self.y += dy

        if self.kept_heading is not None:
            self.kept_heading = None

        return np.array([self.x, self.y])

    def reset(self, x=None, y=None):
        if x and y:
            self.x, self.y = x, y
        else:
            self.x = np.round(np.random.rand()*.6 + .2, 2)
            self.y = np.round(np.random.rand()*.6 + .2, 2)
        self.orient = (np.pi*np.random.randint(0, 4, 1)[0])/2

    def stay(self):
        self.kept_heading = self.heading()
        self.x = self.prev_x
        self.y = self.prev_y


class VelRobot(ObjectBase):
    def __init__(self, init_x: float, init_y: float):
        super(VelRobot, self).__init__(f"{config.root}/assets/robot.png",
                                       np.array((30, 30)),
                                       init_x, init_y)


class Charger(ObjectBase):
    def __init__(self, x: float, y: float):
        super(Charger, self).__init__(f"{config.root}/assets/charger.png",
                                      np.array((30, 30)),
                                      x, y)

class Goal(ObjectBase):
    def __init__(self, x: float, y: float):
        super(Goal, self).__init__(f"{config.root}/assets/goal.png",
                                   np.array((30, 30)),
                                   x, y)


class Map:
    def __init__(self, lines=None, goal_pos=(0.5, 0.5)):
        self.lines = lines if lines is not None else []
        self.goal = Charger(*goal_pos)

    def render_info(self):
        return self.lines, self.goal.render_info(scale=config.scale)