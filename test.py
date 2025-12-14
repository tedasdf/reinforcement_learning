# from algorithms.montecarlo.monte_carlo import MonteCarlo
# from env.OneDMaze import OneDimensionEnv


# def generate_episode(env, mc, max_steps=100):
#     episode = []
#     state = env.reset()
#     done = False
#     steps = 0

#     while not done and steps < max_steps:
#         action = mc.random_policy()
#         next_state, reward, done, _ = env.step(action)

#         episode.append((state, reward))
#         state = next_state
#         steps += 1

#     return episode


# def train_mc(env, mc, num_episodes=5000):
#     for ep in range(num_episodes):
#         episode = generate_episode(env, mc)
#         mc.update(episode)

#         if (ep + 1) % 500 == 0:
#             print(f"Episode {ep+1}")


# if __name__ == "__main__":
#     env = OneDimensionEnv()


#     action_types = env.action_space

#     print(action_types)

#     mc = MonteCarlo(
#         states=env.state_types,
#         actions=action_types,
#         discount_rate=1.0,
#         alpha=0.05
#     )
    
#     train_mc(env, mc, num_episodes=5000)

#     print("\nEstimated state values:")
#     for state, value in mc.val_states.items():
#         print(f"{state}: {value:.3f}")
import gym
import pufferlib.emulation

# class SampleGymEnv(gym.Env):
#     def __init__(self):
#         self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
#         self.action_space = gym.spaces.Discrete(2)

#     def reset(self):
#         return self.observation_space.sample()

#     def step(self, action):
#         return self.observation_space.sample(), 0.0, False, {}

if __name__ == '__main__':
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

    env = gym.make('ALE/Breakout-v5', render_mode="rgb_array") 
    gymnasium_env = pufferlib.GymToGymnasium(env)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(gymnasium_env)
    observations, info = puffer_env.reset()
    action = puffer_env.action_space.sample()
    observation, reward, terminal, truncation, info = puffer_env.step(action)
    print(observation.shape)