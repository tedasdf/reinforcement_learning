import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_project_new.algorithms.utils import CNNBackbone, ActorNetwork



class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()

        self.state_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2)
        )

        self.action_network = nn.Linear(action_dim, hidden_dim_2)

        self.final_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1)
        )

        nn.init.uniform_(self.final_network[1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.final_network[1].bias,   -3e-4, 3e-4)

    def forward(self, state, action):
        h = self.state_network(state) + self.action_network(action)
        return self.final_network(h)
    

class DeepDetNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, sigma):
        self.backbone = CNNBackbone(in_channels=in_channels)

        self.actor = ActorNetwork(self.backbone.output_dim , hidden_dim , 1)
        self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

        self.sigma = sigma
        self.mean = 0
        self.theta = 0.15
        

    def forward(self, state):
        features = self.backbone(state)
        action = self.actor(features)
        action += action + self.theta * (self.mean - action) + self.sigma * torch.randn_like(action)
        value = self.critic(features)

        return action, value
    
    def loss(self, current_qs , target_qs ):
        return F.mse_loss(current_qs, target_qs, reduction='mean')
    

        