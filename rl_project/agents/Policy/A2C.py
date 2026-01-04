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

    num_envs = 4
    env_id = "ALE/MsPacman-v5"
    gamma = 0.99
    learning_rate = 1e-4
    hidden_dim = 128
    n_steps = 5  # n-step rollout
    max_episodes = 500
    print_every = 10

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

        # 1️⃣ Collect n-step rollout
        for step_idx in range(n_steps):
            obs_tensor = torch.from_numpy(obs).float().to(device) / 255.0
            logits, values = agent(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()

            next_obs, rewards, terminateds, truncateds, infos = envs.step(actions.cpu().numpy())
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

        rollout = []

    envs.close()


    #     returns = []
    #     next_value = torch.zeros(num_envs, device=device)  # bootstrap value for last step

    #     for step in reversed(rollout):
    #         reward = step['rewards']
    #         done = step['dones']
    #         next_value = reward + gamma * next_value * (1 - done)
    #         returns.insert(0, next_value)



    #     actor_loss = 0
    #     critic_loss = 0
    #     entropy_loss = 0

    #     for step, R in zip(rollout, returns):
    #         advantage = R - step['values']
    #         actor_loss += -(step['log_probs'] * advantage.detach()).mean()
    #         critic_loss += advantage.pow(2).mean()
    #         # entropy bonus
    #         dist = torch.distributions.Categorical(logits=agent(step['obs'])[0])
    #         entropy_loss += dist.entropy().mean()

    #     # Average over steps
    #     actor_loss /= n_steps
    #     critic_loss /= n_steps
    #     entropy_loss /= n_steps

    #     loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss


    # logits = agent(obs_tensor)

    # dist = Categorical(logits=logits)
    # actions = dist.sample()
    # # Take random actions
    # actions = envs.action_space.sample()
    # next_obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())

    envs.close()

    # from gymnasium.wrappers import (
    #     GrayscaleObservation, 
    #     ResizeObservation, 
    #     FrameStackObservation
    # )
    # import gymnasium as gym
    # import ale_py
    # import numpy as np


    # gym.register_envs(ale_py)

    # env = gym.make('ALE/Pacman-v5', render_mode="rgb_array")  # render_mode="human" shows the game
    # env = GrayscaleObservation(env, keep_dim=False) # Result: (210, 160)
    # env = ResizeObservation(env, (84, 84))          # Result: (84, 84)
    # env = FrameStackObservation(env, stack_size=3)

    # envs = pufferlib.vector.make(
    #     make_pacman_env,
    #     num_envs=8,
    #     backend=pufferlib.vector.Serial # <-- FIX IS HERE
    # )

    # # Standard RL loop usage
    # obs, infos = envs.reset()
    

    # print(f"Observation Shape: {obs.shape}") 
    # assert obs.shape == (8, 3, 84, 84), "Unexpected observation shape!"

    # # 3. Take a test step
    # # Create random actions for all 8 environments
    # actions = np.array([envs.single_action_space.sample() for _ in range(8)])
    # next_obs, rewards, terminals, truncateds, infos = envs.step(actions)

    # # 4. Verify outputs
    # print(f"Rewards: {rewards}")
    # print(f"Terminals: {terminals}")
    # print(f"Step successful!")

    # actor_1 = ActorNet()
    # actor_2 = ActorNet()
    


    # gym.register_envs(ale_py)

    # env = gym.make('ALE/Pacman-v5', render_mode="rgb_array")  # render_mode="human" shows the game
    # env = GrayscaleObservation(env, keep_dim=False) # Result: (210, 160)
    # env = ResizeObservation(env, (84, 84))          # Result: (84, 84)
    # env = FrameStackObservation(env, stack_size=3)

    # state, _ = env.reset()  # reset env at start of each episode


    # logits_1, value_1 = actor_1(state)
    # logits_2 , value_2 = actor_2(state)