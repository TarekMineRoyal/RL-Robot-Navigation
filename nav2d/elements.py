import numpy as np
import pygame
from nav2d import config


class ObjectBase:
    def __init__(self, image_path: str, shape: np.ndarray, init_x: float, init_y: float):
        self.image = pygame.image.load(image_path)
        self.shape = shape
        self.image = pygame.transform.scale(self.image, tuple(self.shape))

        self.x = init_x
        self.y = init_y
        self.orient = np.pi / 2
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
        return np.arctan2(dy, dx)

    def render_info(self, scale):
        # Calculates pixel position on the screen
        pos_x = int(scale * self.x - self.shape[0] / 2)
        pos_y = config.map_size[1] - int(scale * self.y - self.shape[1] / 2)
        return self.image, (pos_x, pos_y)

    def move(self, dx, dy) -> np.ndarray:
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += dx
        self.y += dy

        if self.kept_heading is not None:
            self.kept_heading = None

        return np.array([self.x, self.y])

    def reset(self, x=None, y=None):
        if x is not None and y is not None:
            self.x, self.y = x, y
        else:
            self.x = np.round(np.random.rand() * 0.6 + 0.2, 2)
            self.y = np.round(np.random.rand() * 0.6 + 0.2, 2)
        self.orient = (np.pi * np.random.randint(0, 4, 1)[0]) / 2

    def stay(self):
        self.kept_heading = self.heading()
        self.x = self.prev_x
        self.y = self.prev_y


# ==========================================
# Obstacles / Hazards
# ==========================================

class StaticObstacle(ObjectBase):
    def __init__(self, x: float, y: float, radius: float = 0.05):
        self.radius = radius
        size = int(self.radius * 2 * config.scale)
        super().__init__(f"{config.root}/assets/cat.png", np.array((size, size)), x, y)

    def move(self, *args, **kwargs):
        # Static obstacles do not move
        pass


class MovingCreature(ObjectBase):
    def __init__(self, x: float, y: float, radius: float = 0.05, velocity: float = config.creature_velocity):
        self.radius = radius
        self.velocity = velocity
        self.angle = np.random.uniform(-np.pi, np.pi)

        size = int(self.radius * 2 * config.scale)
        super().__init__(f"{config.root}/assets/cat.png", np.array((size, size)), x, y)

    def move(self, *args, **kwargs):
        next_x = self.x + self.velocity * np.cos(self.angle)
        next_y = self.y + self.velocity * np.sin(self.angle)

        # Bounce off walls
        if next_x <= 0 or next_x >= 1.0:
            self.angle = np.pi - self.angle
            next_x = np.clip(next_x, 0.01, 0.99)
        if next_y <= 0 or next_y >= 1.0:
            self.angle = -self.angle
            next_y = np.clip(next_y, 0.01, 0.99)

        # Update positions
        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = next_x, next_y


class GoalOrbitingCreature(ObjectBase):
    def __init__(self, goal_x: float, goal_y: float, radius: float = 0.05, orbit_radius: float = 0.15,
                 velocity: float = config.creature_velocity):
        self.radius = radius
        self.orbit_radius = orbit_radius
        self.velocity = velocity
        self.orbit_angle = np.random.uniform(-np.pi, np.pi)

        init_x = goal_x + self.orbit_radius * np.cos(self.orbit_angle)
        init_y = goal_y + self.orbit_radius * np.sin(self.orbit_angle)

        size = int(self.radius * 2 * config.scale)
        super().__init__(f"{config.root}/assets/cat.png", np.array((size, size)), init_x, init_y)

    def move(self, goal_x: float, goal_y: float):
        angular_velocity = self.velocity / self.orbit_radius
        self.orbit_angle += angular_velocity
        self.orbit_angle = (self.orbit_angle + np.pi) % (2 * np.pi) - np.pi

        self.prev_x, self.prev_y = self.x, self.y
        self.x = goal_x + self.orbit_radius * np.cos(self.orbit_angle)
        self.y = goal_y + self.orbit_radius * np.sin(self.orbit_angle)


# ==========================================
# Core Elements
# ==========================================

class VelRobot(ObjectBase):
    def __init__(self, init_x: float, init_y: float):
        super().__init__(f"{config.root}/assets/robot.png", np.array((30, 30)), init_x, init_y)


class Charger(ObjectBase):
    def __init__(self, x: float, y: float):
        super().__init__(f"{config.root}/assets/charger.png", np.array((30, 30)), x, y)


class Goal(ObjectBase):
    def __init__(self, x: float, y: float):
        super().__init__(f"{config.root}/assets/goal.png", np.array((30, 30)), x, y)


class Map:
    def __init__(self, lines=None, goal_pos=(0.5, 0.5)):
        self.lines = lines if lines is not None else []
        self.goal = Charger(*goal_pos)

    def render_info(self):
        return self.lines, self.goal.render_info(scale=config.scale)