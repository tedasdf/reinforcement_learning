import torch.nn as nn


class ActorNet():
    def __init__(self):
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError

class CriticNet():
    def __init__(self):
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError
    

# class A2Cnetwork(nn.Module):
#     def __init__(self, n_actor):
        

#         for i in 



class A2Cnetwork(nn.Module):
    def __init__(self, n_actor):
        raise NotImplementedError
    
    def forward():
        raise NotImplementedError
    
    def loss():
        raise NotImplementedError
    

    def collect():
        raise NotImplementedError
    




import gymnasium as gym
import ale_py

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

if __name__ == "__main__":
    from ACnet import ActorCriticNetwork
    import torch
    import torch.optim as optim

    num_envs = 4 
    env_id = "ALE/MsPacman-v5"

    gamma = 0.99
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Using gymnasium.make_vec is generally preferred for vectorized Atari envs
    # as it handles the wrappers (like resizing/grayscale) more easily.
    envs = gym.make_vec(env_id, num_envs=num_envs)

    hidden_dim = 128
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    state_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
    action_dim = action_space.n
    
    agent = ActorCriticNetwork(state_dim, hidden_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Reset all environments
    obs, info = envs.reset()

    n_steps = 5 
    rollout = [] 

    for step in range(n_steps):
        
        obs_tensor = torch.from_numpy(obs).float().to(device) / 255.0
        logits, value = agent(obs_tensor)

        dist = torch.distributions.Categorical(logits)
        actions = dist.samples()

        next_obs, rewards, dones, infos = envs.step(actions.cpu().numpy())

        # Store step data
        rollout.append({
            'obs': obs_tensor,
            'actions': actions,
            'rewards': torch.tensor(rewards, device=device, dtype=torch.float),
            'values': value.squeeze(-1),
            'dones': torch.tensor(dones, device=device, dtype=torch.float),
            'log_probs': dist.log_prob(actions)
        })

        obs = next_obs

    print(rollout)

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