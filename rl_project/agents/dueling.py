import torch
import torch.nn as nn
from rl_project.utils.helpers import Q_network


"""
https://arxiv.org/pdf/1511.06581
"""

class DuelingDQNNetwork(nn.Module):
    def __init__(self, action_dim, q_type):
        super().__init__()
        # feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # value & advantage heads
        self.val_fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv_fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.q_type = q_type 

    def forward(self, x):
        f = self.feature(x)
        f = f.view(f.size(0), -1)
        V = self.val_fc(f)
        A = self.adv_fc(f)
        if self.q_type == 'mean':
            return V + (A - A.mean(dim=1, keepdim=True))
        return V + (A - A.max(dim=1, keepdim=True).values)

class DuelingAgent(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, action_dim, replay_buffer, tau, q_type):
        super().__init__(state_dim, hidden_dim, action_dim, replay_buffer, tau)
        # overwrite networks
        assert q_type in ['mean', 'max'], "q_type must be 'mean' or 'max'"
        self.q_network = DuelingDQNNetwork(action_dim, q_type)
        self.target_network = DuelingDQNNetwork(action_dim, q_type)
        self.target_network.load_state_dict(self.q_network.state_dict())
        for p in self.target_network.parameters():
            p.requires_grad = False

    def forward(self, state):
        return self.q_network(state)
    
    def target_calculation(self, state):
        with torch.no_grad():
            return self.target_network(state)
