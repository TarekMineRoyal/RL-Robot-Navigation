import pygame
from nav2d.engine import NavigationEngine
from nav2d.elements import VelRobot, Map


def main():
    # 1. Initialize the Environment
    robot = VelRobot(0.5, 0.5)
    env_map = Map()
    env = NavigationEngine(robot=robot, env_map=env_map)

    env.reset()

    # Setup Pygame clock for frame rate limiting
    clock = pygame.time.Clock()
    running = True

    print("Environment loaded successfully!")
    print("Controls: UP=Forward, DOWN=Sprint, LEFT=Turn Left, RIGHT=Turn Right")
    print("Close the window to exit.")

    while running:
        action = -1  # Default to no action

        # Process Pygame events to prevent the window from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = 0
                elif event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3

        # Only step the environment if a key was pressed
        if action != -1:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action} | Reward: {reward:.2f}")

            if terminated or truncated:
                print(
                    f"Episode finished! (Goal: {info['reach_goal']}, Obstacle: {info['hit_obstacle']}, Wall: {info['hit_wall']})")
                print("Resetting environment...")
                env.reset()

        # Render the environment to the screen
        env.render()
        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()


if __name__ == '__main__':
    main()