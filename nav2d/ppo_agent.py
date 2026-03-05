import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import scipy.signal


# Math helper function to calculate discounted cumulative sums
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """A buffer for storing trajectories experienced by a PPO agent."""

    def __init__(self, size, state_dim):
        self.obs_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = 0.99, 0.95
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # Generalized Advantage Estimation (GAE)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # Buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]


class PPOAgent:
    def __init__(self, state_dim=204, action_dim=4):
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.train_iters = 80

        # Build Actor (Policy) Network
        obs_input = Input(shape=(state_dim,))
        x = Dense(256, activation='relu')(obs_input)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        action_probs = Dense(action_dim, activation='softmax')(x)
        self.actor = Model(inputs=obs_input, outputs=action_probs)
        self.actor_optimizer = Adam(learning_rate=3e-4)

        # Build Critic (Value) Network
        v_input = Input(shape=(state_dim,))
        v = Dense(256, activation='relu')(v_input)
        v = Dense(256, activation='relu')(v)
        v = Dense(128, activation='relu')(v)
        state_value = Dense(1, activation='linear')(v)
        self.critic = Model(inputs=v_input, outputs=state_value)
        self.critic_optimizer = Adam(learning_rate=1e-3)

    def get_action(self, obs):
        obs = obs.reshape(1, -1)
        probs = self.actor(obs).numpy()[0]
        val = self.critic(obs).numpy()[0][0]

        # Sample an action from the categorical distribution
        action = np.random.choice(len(probs), p=probs)
        logp = np.log(probs[action] + 1e-10)
        return action, val, logp

    @tf.function
    def train_step(self, obs, acts, advs, returns, old_log_probs):
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            # 1. Critic Loss (Mean Squared Error)
            v_pred = tf.squeeze(self.critic(obs))
            loss_c = tf.reduce_mean(tf.square(v_pred - returns))

            # 2. Actor Loss (Clipped Surrogate Objective)
            logits = self.actor(obs)
            action_masks = tf.one_hot(acts, 4)
            probs = tf.reduce_sum(action_masks * logits, axis=1)
            log_probs = tf.math.log(probs + 1e-10)

            ratio = tf.exp(log_probs - old_log_probs)
            clip_adv = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advs
            loss_a = -tf.reduce_mean(tf.minimum(ratio * advs, clip_adv))

        # Apply gradients
        grads_a = tape_a.gradient(loss_a, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables))

        grads_c = tape_c.gradient(loss_c, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables))

        return loss_a, loss_c