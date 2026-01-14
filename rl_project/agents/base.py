from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractAgent(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def act(self, obs):
        """
        Returns:
            action
            log_prob
            value
            entropy
        """
        pass

    @abstractmethod
    def store(self, transition):
        pass

    @abstractmethod
    def compute_loss(self, next_value):
        pass

    @abstractmethod
    def clear_memory(self):
        pass
