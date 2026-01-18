import os
import torch
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, device, network, action_space):
        self.device = device
        self.network = network.to(device)   # <-- probably missing
        self.memory = []
        self.action_space = action_space

    @abstractmethod
    def setup_network(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_action(self, state_tensor):
        """Return action and any extra info needed for memory"""
        raise NotImplementedError

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, extra):
        """Store a transition in memory"""
        raise NotImplementedError

    @abstractmethod
    def process_memory(self, next_state=None):
        """Compute returns or advantages from memory"""
        raise NotImplementedError
    
    @abstractmethod
    def memory_clear(self):
        self.memory = [] 

    def format_action(self, action_tensor):
        """
        Converts model output into an env-compatible action
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            # scalar int
            return action_tensor.item()

        elif isinstance(self.action_space, gym.spaces.Box):
            # continuous vector
            action = action_tensor.detach().cpu().numpy()[0]
            return np.clip(action, self.action_space.low, self.action_space.high)

        else:
            raise NotImplementedError(f"Unsupported action space: {self.action_space}")

    def save_checkpoint(self, episode, reward, path="checkpoints/"):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, f"agent_ep_{episode}.pth")
        torch.save({
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'reward': reward
        }, file_path)
        print(f"Checkpoint saved: {file_path}")

    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint['episode']



