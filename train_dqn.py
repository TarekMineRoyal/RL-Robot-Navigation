from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map

if __name__ == '__main__':
    robot = VelRobot(0.5, 0.5)
    env_map = Map()
    env = NavigationEngine(robot=robot, Map=env_map)
    env.run()