


from rl_project_new.agents.base import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(self, network, gama, clip_epsilon):
        super().__init__(network, gamma)
        self.clip_epsilon = clip_epsilon


    def store_transition(self, state, action, reward, next_state, done, extra):
        return super().store_transition(state, action, reward, next_state, done, extra)
    
    def process_memory(self, next_state=None):
        return super().process_memory(next_state)
    
    def update_networks(self, loss, optimizers):
        return super().update_networks(loss, optimizers)
    
    def memory_clear(self):
        return super().memory_clear()   
    
    def format_action(self, action_tensor):
        return super().format_action(action_tensor)
    
import torch 
import torch.nn as nn
from torch.dsistributions import Categorical



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    print(f"Observation Dimension: {obs_dim}, Action Dimension: {act_dim}")

