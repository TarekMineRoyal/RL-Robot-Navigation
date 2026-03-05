import copy
from typing import List, Tuple

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


class Cat(ObjectBase):
    def __init__(self, init_x: float, init_y: float):
        super(Cat, self).__init__(f"{config.root}/assets/cat.png",
                                  np.array((30, 30)),
                                  init_x, init_y)


class Charger(ObjectBase):
    def __init__(self, x: float, y: float):
        super(Charger, self).__init__(f"{config.root}/assets/charger.png",
                                      np.array((30, 30)),
                                      x, y)


class Flight(ObjectBase):
    def __init__(self, x: float, y: float):
        super(Flight, self).__init__(f"{config.root}/assets/flight.png",
                                     np.array((30, 30)),
                                     x, y)


class Car(ObjectBase):
    def __init__(self, x: float, y: float):
        super(Car, self).__init__(f"{config.root}/assets/Car.png",
                                  np.array((30, 30)),
                                  x, y)


class Goal(ObjectBase):
    def __init__(self, x: float, y: float):
        super(Goal, self).__init__(f"{config.root}/assets/goal.png",
                                   np.array((30, 30)),
                                   x, y)


class Map:
    def __init__(self, lines: List[Tuple], goal_pos: Tuple[float, float]):
        self.lines = lines
        self.goal = Charger(*goal_pos)

    def render_info(self):
        return self.lines, self.goal.render_info(scale=config.scale)


def create_open_space() -> Map:
    return Map([], goal_pos=(400, 400))


def create_map(scale=800, indx=0, goal=(.5, .5)) -> Map:
    edge1, edge2 =  .0125*scale, .9875*scale
    line0 = (0, [edge1, edge1], [edge1, edge2], 5)
    line1 = (0, [edge1, edge2], [edge2, edge2], 5)
    line2 = (0, [edge2, edge2], [edge2, edge1], 5)
    line3 = (0, [edge2, edge1], [edge1, edge1], 5)
    goal_pos = goal

    frame = [line0, line1, line2, line3]
    if indx == 0:
        return Map(frame, goal_pos)
    elif indx == 1:
        Map_lines = copy.deepcopy(frame)
        wall0 = (0, [10, 200], [200, 200], 5)
        wall1 = (0, [10, 650], [650, 650], 5)
        wall2 = (0, [10, 500], [400, 500], 5)
        wall3 = (0, [300, 10], [300, 400], 5)
        wall4 = (0, [500, 10], [500, 500], 5)
        wall5 = (0, [500, 400], [650, 400], 5)
        wall6 = (0, [650, 200], [790, 200], 5)
        Map_lines.extend([wall0, wall1, wall2, wall3, wall4, wall5, wall6])

        return Map(Map_lines, goal_pos)
    elif indx == 2:
        goal_pos = (700, 400)
        Map(frame, goal_pos)

        return Map(frame, goal_pos)
    else:
        raise NotImplementedError()
