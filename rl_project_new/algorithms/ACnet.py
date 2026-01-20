import torch.nn as nn
from rl_project_new.algorithms.utils import CNNBackbone, ActorNetwork, CriticNetwork


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.backbone = CNNBackbone(in_channels=state_dim)  # frame stack = 4
    
        self.actor = ActorNetwork(self.backbone.output_dim , hidden_dim , action_dim)
        self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

    
    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor(features)
        value = self.critic(features)

        return logits, value
    
    
    def loss(self, advantage, log_prob, value):
        actor_loss = -(log_prob * advantage.detach())  # advantage = GAE
        critic_loss = (value - (value + advantage)).pow(2).mean()
        return actor_loss, critic_loss