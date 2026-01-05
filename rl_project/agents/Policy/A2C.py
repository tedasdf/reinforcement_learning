import torch.nn as nn
import gymnasium as gym
import ale_py
import torch
import torch.optim as optim
from torch.distributions import Categorical
from ACnet import ActorCriticNetwork
import numpy as np

# Register ALE environments
gym.register_envs(ale_py)

if __name__ == "__main__":

    import wandb


    num_envs = 4
    env_id = "ALE/MsPacman-v5"
    gamma = 0.99
    learning_rate = 1e-4
    hidden_dim = 128
    n_steps = 5  # n-step rollout
    max_episodes = 5000
    print_every = 10


    wandb.init(
        project="actor-critic-atari",
        name="pacman-vectorized-nstep",
        config={
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "num_envs": num_envs,
            "gamma": gamma,
            "entropy_coef": 0.01,
            "critic_coef": 0.5,
        }
    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vectorized env
    envs = gym.make_vec(env_id, num_envs=num_envs)
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    action_dim = action_space.n

    # Create shared Actor-Critic network
    agent = ActorCriticNetwork(4, hidden_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Reset environments
    obs, info = envs.reset()
    episode_rewards = np.zeros(num_envs)

    total_episodes = 0
    while total_episodes < max_episodes:

        rollout = []
        episode_frames = [] # Collect frames for wandb video

        # 1️⃣ Collect n-step rollout
        for step_idx in range(n_steps):
            obs_tensor = torch.from_numpy(obs).float().to(device) / 255.0
            logits, values = agent(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()

            next_obs, rewards, terminateds, truncateds, infos = envs.step(actions.cpu().numpy())
            frame = envs.render() 
            if frame is not None:
                # wandb.Video expects (Channels, Height, Width) for each frame
                # So we transpose from (H, W, C) to (C, H, W)
                episode_frames.append(frame.transpose(2, 0, 1))
                    
            dones = terminateds | truncateds

            # Track cumulative rewards
            episode_rewards += rewards

            # Store rollout data
            rollout.append({
                'obs': obs_tensor,
                'logits': logits,
                'values': values.squeeze(-1),
                'actions': actions,
                'log_probs': dist.log_prob(actions),
                'rewards': torch.tensor(rewards, device=device, dtype=torch.float),
                'dones': torch.tensor(dones, device=device, dtype=torch.float)
            })

            # Reset episode rewards for finished episodes
            for i, done in enumerate(dones):
                if done:
                    total_episodes += 1
                    print(f"Env {i} finished episode {total_episodes}, reward: {episode_rewards[i]}")
                    episode_rewards[i] = 0

            obs = next_obs

        # 2️⃣ Compute n-step returns
        obs_tensor = torch.from_numpy(obs).float().to(device) / 255.0
        _, next_value = agent(obs_tensor)
        next_value = next_value.squeeze(-1)

        returns = []
        R = next_value
        for step in reversed(rollout):
            R = step['rewards'] + gamma * R * (1 - step['dones'])
            returns.insert(0, R)

        # 3️⃣ Compute actor, critic, and entropy loss
        actor_loss, critic_loss, entropy_loss = 0, 0, 0
        for step, R in zip(rollout, returns):
            advantage = R - step['values']
            actor_loss += -(step['log_probs'] * advantage.detach()).mean()
            critic_loss += advantage.pow(2).mean()
            dist = Categorical(logits=step['logits'])
            entropy_loss += dist.entropy().mean()

        actor_loss /= n_steps
        critic_loss /= n_steps
        entropy_loss /= n_steps

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        # 4️⃣ Backprop & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LOGGING LOGIC
        log_data = {
            "train/total_loss": loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/critic_loss": critic_loss.item(),
            "train/entropy": entropy_loss.item(),
        }

        # Only upload video every 50 episodes to save time/bandwidth
        if total_episodes % 50 == 0 and len(episode_frames) > 0:
            # Stack frames into (Time, Channels, Height, Width)
            video_array = np.array(episode_frames)
            log_data["gameplay_video"] = wandb.Video(video_array, fps=30, format="mp4")
            # Clear frames so we don't log the same ones twice
            episode_frames = [] 

        wandb.log(log_data)
        rollout = []

    envs.close()