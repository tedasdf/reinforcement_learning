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
    

    def ac_loss(self):
        raise NotImplementedError

    def a2c_loss(self, value, target, action, dist):
        log_prob = dist.log_prob(action)
        advantage = target - value
        
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy = dist.entropy()
        
        # One scalar loss for this specific transition
        return actor_loss + 0.5 * critic_loss - 0.01 * entropy