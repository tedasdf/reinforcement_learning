import torch.nn as nn
from rl_project.utils.helpers import Q_network


"""
https://arxiv.org/pdf/1511.06581
"""

class DuelingAgent(nn.Module):
    def __init__(
            self,
            state_dim , 
            hidden_dim, 
            action_dim,
            replay_buffer,
            tau
        ):
        super().__init__(self)
        self.q_network = nn.Sequential(
            nn.Con
        )
        self.target_network = Q_network(state_dim, hidden_dim , action_dim)


        self.replay_buffer = replay_buffer

        ### epsilon 
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 10000

        self.target_update_frequency = 100
        self.gamma = 0.99

        self.total_steps = 0

        self.tau = tau

    def 