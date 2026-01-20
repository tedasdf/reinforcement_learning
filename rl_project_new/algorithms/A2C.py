import torch
import torch.nn as nn
from rl_project_new.algorithms.utils import CNNBackbone, ActorNetwork, CriticNetwork

"""
    introduce significant improvement in stability and performance by employing the advantage
    function to reduce the variance inherent in basic policy gradient estimate.

    Mechanism?
        - Parallel ACtors: 
        - Experience Collection
        - Synchronization 
    
    
    
    
    How ?
        
        - 

"""

class AdvantgeActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_num):
        super().__init__()

        self.actor_num = actor_num


        self.backbone = CNNBackbone(in_channels=state_dim)  # frame stack = 4
        
        self.actors = nn.ModuleList([
            ActorNetwork(self.backbone.output_dim, hidden_dim, action_dim) 
            for _ in range(actor_num)
        ])
        
        self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

    def forward(self, x , actor_idx=None):
        features = self.backbone(x)

        if actor_idx is not None:
            logits = self.actors[actor_idx](features)
        else:
            logits = [actor(features) for actor in self.actors]

            logits = torch.stack(logits, dim=0)

        value = self.critic(features)
        return logits, value

    def a2c_loss(self, value, target, log_prob):
        advantage = target - value
        
        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
    
        return actor_loss, critic_loss
    