
import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # infer output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU()
        )

        self.output_dim = 512

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ActorNetwork(nn.Module):
    def __init__(self , state_dim, hidden_dim , action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)
    


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)
    

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.backbone = CNNBackbone(in_channels=4)  # frame stack = 4
    
        self.actor = ActorNetwork(self.backbone.output_dim , hidden_dim , action_dim)
        self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

        self.gamma = 0.99
    
    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


    def loss_fn(self, action ,value ,reward, next_value , done):
        log_prob = dist.log_prob(action)
        # TD target
        td_target = reward + self.gamma * next_value * (1 - done)

        # Advantage
        advantage = td_target - value

        actor_loss = -log_prob * advantage.detach()

        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss
        return loss
    

if __name__ == "__main__":

    import torch.optim as optim
    import torch.nn as nn
    from gymnasium.wrappers import (
        ResizeObservation,
        GrayScaleObservation,
        FrameStack
    )
    import gymnasium as gym
    import ale_py, torch , wandb
    import numpy as np

    wandb.init(
        project="actor-critic-atari",
        name="ms_pacman_cnn_ac",
        config={
            "lr": 1e-4,
            "gamma": 0.99,
            "episodes": 5
        }
    )

    gym.register_envs(ale_py)

    env = gym.make('ALE/Pacman-v5', render_mode="human")  # render_mode="human" shows the game
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)
    num_episodes = 5  # number of episodes to run
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128

    total_steps  = 0

    agent = ActorCriticNetwork(state_dim, hidden_dim , action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)


    global_step = 0 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for episode in range(num_episodes):
        state, _ = env.reset()  # reset env at start of each episode
        episode_reward = 0
        done = False

        while not done :   

            state_tensor = torch.tensor(
                np.array(state), dtype=torch.float32, device=device
            ).unsqueeze(0) / 255.0

            # Forward pass
            logits, value = agent(state)
                
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            frame = env.render()

            done = terminated or truncated
            episode_reward += reward

            with torch.no_grad():
                next_state_tensor = torch.tensor(
                    np.array(next_state), dtype=torch.float32, device=device
                ).unsqueeze(0) / 255.0
                _, next_value = agent(next_state)

            loss = agent.loss_fn(action, value, reward, next_value, float(done))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            wandb.log({
                "loss": loss.item(),
                "reward": reward,
                "value": value.item(),
            }, step=global_step)
            
            
            state = next_state
            total_steps += 1
           
        wandb.log({"episode_reward": episode_reward}, step=global_step)
        print(f"Episode {episode+1} | Reward: {episode_reward}")
    env.close()
    wandb.finish()