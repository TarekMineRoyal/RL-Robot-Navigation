from typing import List, Optional, Union
from collections import deque, namedtuple
import time
import numpy as np
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from nav2d import config
from nav2d.elements import (Cat, Charger, Map, VelRobot)
from nav2d.utils import denormalize_pos, create_video
from nav2d import config
from nav2d import utils

class NavigationEngine:
    def __init__(self, robot: VelRobot, obstacle_list: List[Cat], Map: Map):
        self.screen = None

        pygame.display.set_caption("2D robot navigation")
        icon = pygame.image.load(f"{config.root}/assets/robot.png")

        try:
            pygame.display.set_icon(icon)
        except Exception:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.robot = robot
        self.obstacle_list = obstacle_list
        self.Map = Map
        # Goal + Obstacles (X, Y) distances 
        self.observation_size = 2*(1 + len(obstacle_list)) + 1
        # Left, Right, Forward, sprint
        self.action_size = 4

        self.plan_lines = []
        self.obj_collision_threshold = config.obj_collision_threshold

    def set_plan(self, plan: np.ndarray, denormalize: bool = False):
        assert plan.shape[1:] == (2,), f"shape {plan.shape[1:]} == (2, )"
        if denormalize:
            plan = denormalize_pos(plan)
        self.plan_lines = np.stack([plan[:-1], plan[1:]], axis=1)

    def render(self):
        # call other functions before calling it
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(config.map_size)

        # white background
        self.screen.fill((155, 255, 255))

        # plot all obstacles
        for obs in self.obstacle_list:
            self.screen.blit(*obs.render_info(scale=config.scale))

        walls, goal = self.Map.render_info()
        # plot walls
        for line in walls:
            pygame.draw.line(self.screen, *line)

        # plot plan
        for plan_line in self.plan_lines:
            pygame.draw.line(self.screen, "green", plan_line[0], config.map_size[1] - plan_line[1], 1)

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

    def dist_goal(self) -> np.ndarray:
        # distance to goal in the normalize coordinator
        robot_pos = np.array([self.robot.x, self.robot.y])
        goal_pos = np.array([self.Map.goal.x, self.Map.goal.y])
        dist = np.linalg.norm(goal_pos - robot_pos), np.arctan2(-goal_pos[1]+robot_pos[1], goal_pos[0]-robot_pos[0])
        return dist
    
    def dist_obtacles(self):
        obs_dists = np.array([])
        robot_pos = np.array([self.robot.x, self.robot.y])
        
        for obs in self.obstacle_list:            
            obs_pos = np.array([obs.x, obs.y])
            dist = np.linalg.norm(obs_pos - robot_pos), np.arctan2(-obs_pos[1]+robot_pos[1], obs_pos[0]-robot_pos[0])
            obs_dists = np.concatenate((obs_dists, dist), axis=0)
            
        return obs_dists
    
    def hit_wall(self) -> bool:
        if self.robot.x <= 0 or self.robot.x >= 1 or self.robot.y <= 0 or self.robot.y >= 1:
            return True
        else:
            return False

    def hit_object(self, obj: Union[Charger, Cat]) -> bool:
        dx = self.robot.x - obj.x
        dy = self.robot.y - obj.y

        if abs(dx) <= self.obj_collision_threshold \
                and abs(dy) <= self.obj_collision_threshold:
            return True

        return False
    
    def step(self, action: int):
        
        action_space = {0: 'RIGHT', 1:'LEFT', 2:'FORWARD', 3:'SPRINT'}
        angle = self.robot.orient
        if action_space[action] == 'LEFT':
            self.robot.orient += np.pi/2
        elif action_space[action] == 'RIGHT':
            self.robot.orient -= np.pi/2
        elif action_space[action] == 'FORWARD':
            self.robot.move(config.robot_vel_scale*np.cos(angle), \
                config.robot_vel_scale*np.sin(angle))
        elif action_space[action] == 'SPRINT':
            self.robot.move(2*config.robot_vel_scale*np.cos(angle), \
                2*config.robot_vel_scale*np.sin(angle))
        angle = self.robot.orient
        if angle > 2*np.pi:
            self.robot.orient -= 2*np.pi
        elif angle < -2*np.pi:
            self.robot.orient += 2*np.pi
            
        reach_goal, hit_obstacle, hit_wall = self.get_robot_status()
        
        reward = reach_goal*config.reach_goal_reward + \
            hit_obstacle*config.hit_obstacle_reward + \
            hit_wall*config.hit_wall_reward + \
            config.step_penalty + \
            config.trun_penalty*(action==0 or action==1)

        done = False
        if reach_goal or hit_obstacle or hit_wall:
            done = True
        
        # absolute robot coordinates in observation?? 
        observation = np.concatenate((self.dist_goal(), self.dist_obtacles(), np.array([self.robot.orient])), axis=0)
        return observation, reward, done
    
    def reset(self, pos=None):     
        if not pos:
            self.robot.reset()
            #self.Map.goal.x = np.round(np.random.rand()*.6 + .2, 2)
            #self.Map.goal.y = np.round(np.random.rand()*.6 + .2, 2)
        else:
            self.robot.reset(pos[0], pos[1])
        
        return np.concatenate((self.dist_goal(), self.dist_obtacles(), np.array([self.robot.orient])), axis=0)
    
    def compute_loss(self, experiences, gamma, network, target):
        states, actions, rewards, next_states, done_vals = experiences
        max_qsa = tf.reduce_max(target(next_states), axis=-1)
        y_targets = rewards*done_vals + (1-done_vals)*(rewards + gamma*max_qsa)
        q_values = network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
        loss = MSE(y_targets, q_values)    
        return loss

    @tf.function
    def agent_learn(self, experiences, gamma, network, target, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, gamma, network, target)

        gradients = tape.gradient(loss, network.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))

        utils.update_target_network(network, target)
        return loss

    def run(self):
        
        state_size = self.observation_size
        num_actions = self.action_size
        num_episodes = 4000
        max_num_steps = 100
        epsilon = 1.0
        MEMORY_SIZE = 100_000     # size of memory buffer
        GAMMA = 0.995             # discount factor
        NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
        num_p_av = 100
        soft_upd = .01
        total_point_history = []
        total_rewards = 0
        self.obj_collision_threshold = .02
        
        # Create the Q-Network
        q_network = Sequential([
            tf.keras.Input(state_size),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='linear')
            ])

        # Create the target Q^-Network
        target_q_network = Sequential([
            tf.keras.Input(state_size),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='linear')
            ])
        
        optimizer = Adam()
        
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        memory_buffer = deque(maxlen=MEMORY_SIZE)

        target_q_network.set_weights(q_network.get_weights())

        for i in range(num_episodes):
            
            pos = (.5, .5)  if i < num_episodes/4 else None
            state = self.reset(pos)
            total_points = 0
            
            for t in range(max_num_steps):
                
                state_qn = np.expand_dims(state, axis=0)
                q_values = q_network(state_qn)
                action = utils.get_action(q_values, epsilon)
                next_state, reward, done = self.step(action)
                #print(action, next_state[-3:])
                
                memory_buffer.append(experience(state, action, reward, next_state, done))
                
                update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
                
                if update:
                    experiences = utils.get_experiences(memory_buffer)
                    
                    self.agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer)
                
                state = next_state.copy()
                total_points += reward
                
                if done:
                    break
                
            total_rewards = total_rewards + soft_upd*(total_points - total_rewards)    
            total_point_history.append(total_rewards)
            av_latest_points = np.mean(total_point_history[-num_p_av:])
            
            epsilon = .3 if (t == num_episodes/2 or t == num_episodes/4) else utils.get_new_eps(epsilon)


            print(f"\rEpisode {i+1} | Average Reward: {total_rewards:.2f}", end="")

            if (i+1) % num_p_av == 0:
                print(f"\rEpisode {i+1} | Total Average: {av_latest_points:.2f}")
                q_network.save('carnav_model.keras')
            
            if av_latest_points >= 80.0 and i > num_episodes/2:
                print(f"\n\nEnvironment solved in {i+1} episodes!")
                break
                    
        utils.plot_history(total_point_history, plot_data_only=True)
