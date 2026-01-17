import os
import torch
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, device, network):
        self.device = device
        self.network = network.to(device)   # <-- probably missing
        self.memory = []

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



