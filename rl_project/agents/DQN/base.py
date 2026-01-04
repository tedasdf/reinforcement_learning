from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, state_dim, action_dim, policy=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = policy

    @abstractmethod
    def select_action(self, state):
        """Return action for given state."""
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """Update agent parameters."""
        pass

    def set_policy(self, policy):
        self.policy = policy
