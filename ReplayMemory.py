import collections
import random
import numpy as np

# Transition = namedtuple('Transition',
#                         ('state','action','next_state','reward'))

# class ReplayMemory:
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         self.memory.append(Transition(*args))
    
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

class ReplayBuffer:
    def __init__(self, capacity):
        # Use a deque as it automatically handles max size
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """Stores a transition tuple in the buffer."""
        # Ensure states are NumPy arrays for consistency
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # Add the experience tuple to the deque
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a mini-batch of experiences."""
        # Randomly select indices for the batch
        batch_indices = random.sample(range(len(self.buffer)), batch_size)
        # Retrieve the experiences corresponding to the sampled indices
        experiences = [self.buffer[i] for i in batch_indices]

        # Unzip the batch into separate arrays for states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to NumPy arrays for batch processing by the network
        return (np.concatenate(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.concatenate(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


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
    print("HEALODF")

    memory = ReplayBuffer(10000)

    gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

    env = gym.make('ALE/Breakout-v5', render_mode="rgb_array") 

    env = pufferlib.emulation.GymnasiumPufferEnv(env)
    
    print("HELO ")
    try:
        state, info = env.reset()
    except pufferlib.pufferlib.APIUsageError as e:
        print("Caught APIUsageError!")
        print(e)
        raise ValueError
    state, info = env.reset()

    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated

    print(state.shape)
    print(next_state.shape)
    memory.store(state, action, reward, next_state, done)
    print(memory.buffer[0][0].shape)
    print(memory.buffer[0][1].shape)
    # print(memory.memory)