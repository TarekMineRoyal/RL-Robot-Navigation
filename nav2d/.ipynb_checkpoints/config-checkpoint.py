import os

root = os.path.dirname(os.path.abspath(__file__))

wall_collision_threshold = .04
obj_collision_threshold = .04
scale = 600
map_size = (scale, scale)

robot_vel_scale = .02

# Shape the rewards
reach_goal_reward = 100
hit_obstacle_reward = -50
hit_wall_reward = -50
step_penalty = -0.01
trun_penalty = -1

reward_sensitive = 1.0
