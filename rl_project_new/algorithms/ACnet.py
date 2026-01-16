import torch.nn as nn
from rl_project_new.algorithms.utils import CNNBackbone, ActorNetwork, CriticNetwork


class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, action_dim):
        super().__init__()

        self.backbone = CNNBackbone(in_channels=in_channels)  # frame stack = 4
    
        self.actor = ActorNetwork(self.backbone.output_dim , hidden_dim , action_dim)
        self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

    
    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor(features)
        value = self.critic(features)

        return logits, value
    

    def a2c_loss(self, value, target, log_prob):
        advantage = target - value
        
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)

        
        # One scalar loss for this specific transition
        return actor_loss, critic_loss
    
    def gae_loss(self, advantage, log_prob, value):
        actor_loss = -(log_prob * advantage.detach())  # advantage = GAE
        critic_loss = (value - (value + advantage)).pow(2)  
        return actor_loss, critic_loss