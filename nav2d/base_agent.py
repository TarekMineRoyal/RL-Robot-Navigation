from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class BaseAgent(ABC):
    """
    Abstract Base Class for all Reinforcement Learning agents.
    Enforces a strict interface for interacting with the environment and saving/loading models.
    """
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def get_action(self, state: np.ndarray, *args, **kwargs):
        """Returns an action given the current environment state."""
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Performs a single step of network weight updates."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Saves the agent's primary model(s) to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Loads the agent's primary model(s) from disk."""
        pass