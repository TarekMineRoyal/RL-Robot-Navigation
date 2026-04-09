import random
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf

from nav2d import config
from nav2d.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(self, state_dim: int = None, action_dim: int = None):
        super().__init__(
            state_dim=state_dim or config.observation_size,
            action_dim=action_dim or config.action_size
        )

        # Hyperparameters
        self.gamma = config.dqn_gamma
        self.batch_size = 64
        self.tau = 1e-3  # Soft update parameter

        # Experience Replay Buffer
        self.memory = deque(maxlen=config.dqn_memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Networks & Optimizer
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam()

    def _build_model(self) -> tf.keras.Model:
        """Builds the deep neural network using Orthogonal Initialization for stability."""
        # Orthogonal init prevents vanishing gradients when processing large sensor arrays
        initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(self.action_dim, activation='linear', kernel_initializer=initializer)
        ])
        return model

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_dim)

        state_qn = np.expand_dims(state, axis=0)
        q_values = self.q_network(state_qn)
        return int(np.argmax(q_values.numpy()[0]))

    def get_experiences(self) -> tuple:
        experiences = random.sample(self.memory, self.batch_size)
        states = tf.convert_to_tensor([e.state for e in experiences], dtype=tf.float32)
        actions = tf.convert_to_tensor([e.action for e in experiences], dtype=tf.float32)
        rewards = tf.convert_to_tensor([e.reward for e in experiences], dtype=tf.float32)
        next_states = tf.convert_to_tensor([e.next_state for e in experiences], dtype=tf.float32)
        done_vals = tf.convert_to_tensor([float(e.done) for e in experiences], dtype=tf.float32)
        return states, actions, rewards, next_states, done_vals

    def compute_loss(self, states, actions, rewards, next_states, done_vals) -> tf.Tensor:
        max_qsa = tf.reduce_max(self.target_network(next_states), axis=-1)
        y_targets = rewards * done_vals + (1 - done_vals) * (rewards + self.gamma * max_qsa)

        q_values = self.q_network(states)
        indices = tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1)
        q_values_for_actions = tf.gather_nd(q_values, indices)

        return tf.keras.losses.MSE(y_targets, q_values_for_actions)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, done_vals) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.compute_loss(states, actions, rewards, next_states, done_vals)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        self._update_target_network()
        return loss

    def _update_target_network(self):
        for target_weights, q_net_weights in zip(self.target_network.weights, self.q_network.weights):
            target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)

    # Standardized Save/Load interface
    def save(self, filepath: str):
        self.q_network.save(filepath)

    def load(self, filepath: str):
        self.q_network = tf.keras.models.load_model(filepath)
        self.target_network.set_weights(self.q_network.get_weights())