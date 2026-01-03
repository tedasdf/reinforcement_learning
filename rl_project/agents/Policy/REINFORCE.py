

import torch.nn as nn

# class ACNetwork(nn.Module):
#     def __init__(self, ):

#         self.actor = Actor(state_dim, hidden_dim, )

#         self.critic = Critic(state_dim , hidden_dim, )

#     def forward(self, ):
#         return 




class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        
        self.network = nn.Sequential(
            nn.Linear(state_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, action_dim, hidden_dim, ):

        self.network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    import gymnasium as gym
    import torch

    env = gym.make("ALE/Pacman-v5", render_mode="human")
    
    # Reset the environment
    obs, info = env.reset(seed=42)

    done = False
    while not done:
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()