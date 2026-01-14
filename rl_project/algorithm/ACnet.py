
import torch
import torch.nn as nn

from rl_project.agents.base import AbstractAgent

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
    def __init__(self, in_channels, hidden_dim, action_dim):
        super().__init__()

        self.backbone = CNNBackbone(in_channels=in_channels)  # frame stack = 4
    
        self.actor = ActorNetwork(self.backbone.output_dim , hidden_dim , action_dim)
        self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

        self.gamma = 0.99
    
    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor(features)
        value = self.critic(features)
                        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return  action , logits, value, dist
    
    # bootstrap
    def bootstrap(self, next_state, device):
        with torch.no_grad():
            next_state_tensor = torch.from_numpy(np.array(next_state)).float().to(device)
            if next_state_tensor.ndim == 3:
                next_state_tensor = next_state_tensor.unsqueeze(0) # Add batch dim -> (1, 3, 84, 84)
            next_state_tensor /= 255.0
            _, next_value = agent(next_state_tensor)
        return next_value


    def loss_fn(self, action, value, reward, next_value, done, dist):
        log_prob = dist.log_prob(action)

        td_target = reward + self.gamma * next_value.detach() * (1 - done)

        
        advantage = td_target - value

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy = dist.entropy().mean()

        return actor_loss + 0.5 * critic_loss - 0.01 * entropy





if __name__ == "__main__":

    import torch.optim as optim
    import torch.nn as nn
    from gymnasium.wrappers import (
        GrayscaleObservation, 
        ResizeObservation, 
        FrameStackObservation
    )
    import gymnasium as gym
    import ale_py, torch , wandb
    import numpy as np


    n_step = 5


    wandb.init(
        project="actor-critic-atari",
        name="ms_pacman_cnn_ac_testing2",
        config={
            "lr": 1e-4,
            "gamma": 0.99,
            "episodes": 5
        }
    )

    gym.register_envs(ale_py)

    env = gym.make('ALE/Pacman-v5', render_mode="rgb_array")  # render_mode="human" shows the game
    env = GrayscaleObservation(env, keep_dim=False) # Result: (210, 160)
    env = ResizeObservation(env, (84, 84))          # Result: (84, 84)
    env = FrameStackObservation(env, stack_size=3)
    num_episodes = 5000  # number of episodes to run
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_steps  = 0

    agent = ActorCriticNetwork(state_dim, hidden_dim, action_dim).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=0.001)


    global_step = 0 

    for episode in range(num_episodes):

        state, _ = env.reset()  # reset env at start of each episode
        episode_reward = 0
        done = False

        episode_frames = [] # Collect frames for wandb video

        while not done :   
            
            rollout_data = []

            for _ in range(n_step):
                # If shape is (84, 84, 4), we need to move the 4 to the front
                state_tensor = torch.from_numpy(np.array(state)).float().to(device)
                if state_tensor.ndim == 3:
                    state_tensor = state_tensor.unsqueeze(0) # Add batch dim -> (1, 3, 84, 84)
                state_tensor /= 255.0
                # Forward pass

                action , logits, value, dist = agent(state_tensor)
                    
                
                next_state, reward, terminated, truncated, info = env.step(action.item())
                if episode % 1 == 0: 
                    # RGB array needs to be (Channels, Height, Width) for wandb.Video
                    frame = env.render()
                    episode_frames.append(np.transpose(frame, (2, 0, 1)))

                done = terminated or truncated
                rollout_data.append((value, reward, dist.log_prob(action), dist.entropy()))
                state = next_state
                episode_reward += reward
                if done: break

            next_value = agent.bootstrap(next_state, device)
            

            loss = agent.loss_fn(action, value, reward, next_value, float(done), dist)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Inside the while loop (Every Step)
            wandb.log({
                "loss": loss.item(),
                "reward": reward,
                "value": value.item()
            })

            
            state = next_state
            total_steps += 1

        # Log to WandB
        log_data = {"episode_reward": episode_reward, "episode": episode}
        
        if episode_frames:
            # Log as a video to see the agent actually playing
            log_data["gameplay_video"] = wandb.Video(np.array(episode_frames), fps=30, format="mp4")
        
        wandb.log(log_data)
        print(f"Episode {episode+1} | Reward: {episode_reward}")
        
    wandb.finish()





    # class ActorCriticNetworkKai(AbstractAgent):
#     def __init__(self, in_channels):
#         self.backbone = CNNBackbone(in_channels=in_channels)  # frame stack = 4

#         self.actor = ActorNetwork(self.backbone.output_dim , hidden_dim , action_dim)
#         self.critic = CriticNetwork(self.backbone.output_dim , hidden_dim)

#     def forward(self, state):
#         features = self.backbone(state)
#         logits = self.actor(features)
#         value = self.critic(features)
#         return logits, value


#     def act(self):
#         logits, value = agent(state_tensor)
                
#         dist = torch.distributions.Categorical(logits=logits)
#         action = dist.sample()
#         return action , logits ,value
    
#     def boostrap(self):
