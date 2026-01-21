import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rl_project_new.algorithms.utils import CNNBackbone, ActorNetwork


class ActorNetwork_new(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, n_actions, final_layer_bound):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, n_actions),
            nn.Tanh()
        )

        self.init_network(final_layer_bound)

    def init_network(self, final_layer_bound):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size(1)
                bound = 1. / np.sqrt(fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.uniform_(layer.bias, -bound, bound)

        self.apply(init_layer)
        nn.init.uniform_(self.action_net[-2].weight, -final_layer_bound, final_layer_bound)
        nn.init.uniform_(self.action_net[-2].bias, -final_layer_bound, final_layer_bound)
            

    def forward(self, state):
        return self.action_net(state)
    
class CriticNetwork_new(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, n_actions, final_layer_bound):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1), #fc1
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2)
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(n_actions, hidden_dim_2),
            nn.ReLU(),
        )

        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1)
        )

        self.init_network(final_layer_bound)


    def init_network(self, final_layer_bound):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size(1)
                bound = 1. / np.sqrt(fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.uniform_(layer.bias, -bound, bound)

        self.apply(init_layer)
        nn.init.uniform_(self.output_net[-1].weight, -final_layer_bound, final_layer_bound)
        nn.init.uniform_(self.output_net[-1].bias, -final_layer_bound, final_layer_bound)
            
    def forward(self, state, action):
        state_value = self.state_net(state)
        action_value = self.action_net(action)

        state_action_value = self.output_net(torch.add(state_value, action_value))
        return state_action_value

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt 
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x 
        return x
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class DeepDetNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, sigma, theta, actor_bound, critic_bound):
        super().__init__()
        # self.backbone = CNNBackbone(in_channels=in_channels)
        
        self.actor = ActorNetwork_new(state_dim, hidden_dim[0], hidden_dim[1], action_dim, actor_bound)
        self.critic = CriticNetwork_new(state_dim, hidden_dim[0], hidden_dim[1], action_dim, critic_bound)
        self.noise = OUActionNoise(mu=np.zeros(action_dim), sigma=sigma, theta=theta, dt=1e-2)


    def critic_forward(self, state, action):
        return self.critic(state, action)


    def forward(self, state, use_noise=True):
        action = self.actor(state)
    
        if use_noise:
            action = torch.add(action,torch.tensor(self.noise(), dtype=action.dtype, device=action.device))
    
        value = self.critic(state, action)
        return action, value

    
    def loss(self, current_qs , target_qs ):
        return F.mse_loss(current_qs, target_qs, reduction='mean')
    

        