import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_project_new.algorithms.DDPG import ActorNetwork_new, CriticNetwork_new

class DeepDetNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, sigma=0.15, theta=0.2):
        super().__init__()
        # self.backbone = CNNBackbone(in_channels=in_channels)
        
        self.actor = ActorNetwork_new(state_dim, hidden_dim[0], hidden_dim[1], action_dim, 0.003)
        self.critic = CriticNetwork_new(state_dim, hidden_dim[0], hidden_dim[1], action_dim, 0.003)


    def critic_forward(self, state, action):
        return self.critic(state, action)


    def forward(self, state, use_noise=True):
        action = self.actor(state)
    
        value = self.critic(state, action)
        return action, value

    
    def loss(self, current_qs , target_qs ):
        return F.mse_loss(current_qs, target_qs, reduction='mean')
    

        