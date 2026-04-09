import os

# ==========================================
# Paths & General
# ==========================================
root = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Map & Simulation Parameters
# ==========================================
scale = 600
map_size = (scale, scale)
max_steps_per_episode = 200

# ==========================================
# Entities & Collision Parameters
# ==========================================
wall_collision_threshold = 0.04
obj_collision_threshold = 0.04
robot_vel_scale = 0.03
creature_velocity = 0.01
static_obstacle_count = 3
moving_creature_count = 1
orbiting_creature_count = 1

# ==========================================
# LiDAR Sensor Parameters
# ==========================================
lidar_num_rays = 200
lidar_max_dist = 0.20  # 20% of the normalized map
lidar_penalty_threshold = 0.20  # Within 20% of map size triggers warning
lidar_penalty_value = -1.0

# ==========================================
# Reward Shaping
# ==========================================
reach_goal_reward = 1000.0
hit_obstacle_reward = -250.0
hit_wall_reward = -250.0
step_penalty = -1.0
dense_reward_scale = 100.0  # Multiplier for distance delta to goal
reward_sensitive = 1.0

# ==========================================
# RL Common Architecture
# ==========================================
# Observation: [robot_x, robot_y, goal_x, goal_y, lidar_ray_1 ... lidar_ray_N]
observation_size = 3 + lidar_num_rays
action_size = 4  # 0: RIGHT, 1: LEFT, 2: FORWARD, 3: SPRINT

# ==========================================
# DQN Hyperparameters
# ==========================================
dqn_num_episodes = 3000
dqn_memory_size = 100000
dqn_gamma = 0.995
dqn_update_every = 4
dqn_num_p_av = 100  # For rolling average
dqn_soft_upd = 0.01

# ==========================================
# PPO Hyperparameters
# ==========================================
ppo_steps_per_epoch = 4000
ppo_epochs = 150
ppo_clip_ratio = 0.2
ppo_target_kl = 0.01
ppo_train_iters = 20
ppo_gamma = 0.99
ppo_lam = 0.95
ppo_actor_lr = 3e-4
ppo_critic_lr = 1e-3