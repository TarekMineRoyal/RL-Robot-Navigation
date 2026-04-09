import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import scipy.signal

from nav2d import config
from nav2d.base_agent import BaseAgent


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
        self.gamma, self.lam = config.ppo_gamma, config.ppo_lam
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
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # ADVANTAGE NORMALIZATION: Crucial for PPO stability
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]


class PPOAgent(BaseAgent):
    def __init__(self, state_dim=None, action_dim=None):
        super().__init__(
            state_dim=state_dim or config.observation_size,
            action_dim=action_dim or config.action_size
        )
        self.clip_ratio = config.ppo_clip_ratio
        self.target_kl = config.ppo_target_kl
        self.train_iters = config.ppo_train_iters
        self.entropy_coef = 0.01  # Weight of the entropy bonus

        # Standard orthogonal init for hidden layers
        hidden_init = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        # Small gain for policy output ensures initial probabilities are uniform
        action_init = tf.keras.initializers.Orthogonal(gain=0.01)
        # Unscaled gain for the critic's state-value estimation
        value_init = tf.keras.initializers.Orthogonal(gain=1.0)

        # Build Actor (Policy) Network
        obs_input = Input(shape=(self.state_dim,))
        x = Dense(256, activation='tanh', kernel_initializer=hidden_init)(obs_input)
        x = Dense(256, activation='tanh', kernel_initializer=hidden_init)(x)
        x = Dense(128, activation='tanh', kernel_initializer=hidden_init)(x)
        action_probs = Dense(self.action_dim, activation='softmax', kernel_initializer=action_init)(x)
        self.actor = Model(inputs=obs_input, outputs=action_probs)
        self.actor_optimizer = Adam(learning_rate=config.ppo_actor_lr)

        # Build Critic (Value) Network
        v_input = Input(shape=(self.state_dim,))
        v = Dense(256, activation='tanh', kernel_initializer=hidden_init)(v_input)
        v = Dense(256, activation='tanh', kernel_initializer=hidden_init)(v)
        v = Dense(128, activation='tanh', kernel_initializer=hidden_init)(v)
        state_value = Dense(1, activation='linear', kernel_initializer=value_init)(v)
        self.critic = Model(inputs=v_input, outputs=state_value)
        self.critic_optimizer = Adam(learning_rate=config.ppo_critic_lr)

    def get_action(self, obs):
        obs = obs.reshape(1, -1)
        probs = self.actor(obs).numpy()[0]
        val = self.critic(obs).numpy()[0][0]

        action = np.random.choice(len(probs), p=probs)
        logp = np.log(probs[action] + 1e-10)
        return action, val, logp

    @tf.function
    def train_step(self, obs, acts, advs, returns, old_log_probs):
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            # 1. Critic Loss
            v_pred = tf.squeeze(self.critic(obs))
            loss_c = tf.reduce_mean(tf.square(v_pred - returns))

            # 2. Actor Loss
            logits = self.actor(obs)
            action_masks = tf.one_hot(acts, self.action_dim)
            probs = tf.reduce_sum(action_masks * logits, axis=1)
            log_probs = tf.math.log(probs + 1e-10)

            ratio = tf.exp(log_probs - old_log_probs)
            clip_adv = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advs

            # ENTROPY BONUS: Encourages exploration by penalizing overconfidence
            entropy = tf.reduce_mean(-tf.reduce_sum(logits * tf.math.log(logits + 1e-10), axis=1))

            # Final actor loss = surrogate loss - entropy bonus (we subtract to maximize entropy)
            loss_a = -tf.reduce_mean(tf.minimum(ratio * advs, clip_adv)) - (self.entropy_coef * entropy)

        grads_a = tape_a.gradient(loss_a, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables))

        grads_c = tape_c.gradient(loss_c, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables))

        return loss_a, loss_c

    # Standardized Save/Load interface
    def save(self, filepath: str):
        self.actor.save(filepath)

    def load(self, filepath: str):
        self.actor = tf.keras.models.load_model(filepath)