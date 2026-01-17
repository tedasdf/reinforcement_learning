from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
import ale_py
import cv2
# | Environment                | Action space                            | Observation                                   |
# | -------------------------- | --------------------------------------- | --------------------------------------------- |
# | `Pendulum-v1`              | Continuous, 1D torque (-2 to 2)         | 3D state (cos(theta), sin(theta), velocity)   |
# | `MountainCarContinuous-v0` | Continuous, 1D acceleration (-1 to 1)   | 2D position + velocity                        |
# | `CartPoleContinuous-v1`    | Continuous, 1D force (-1 to 1)          | 4D state (cart pos, vel, pole angle, ang vel) |
# | `LunarLanderContinuous-v2` | Continuous, 2D thrust                   | 8D state                                      |
# | `CarRacing-v2`             | Continuous, 3D [steering, accel, brake] | 96x96x3 RGB                                   |
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

def make_env(env_id, frame_stack=4, render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode)

    obs_shape = env.observation_space.shape

    # Image-based envs: (H, W, C)
    if obs_shape is not None and len(obs_shape) == 3:
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=frame_stack)

    return env

# Create the environment
env = make_env('LunarLanderContinuous-v3', render_mode='human')  # or 'rgb_array' if headless


# Play for a few episodes
num_episodes = 2
for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        # Random continuous action: [steering, acceleration, brake]
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        done = terminated or truncated

        total_reward += reward
        step += 1

        # Optional: show the processed observation (stacked grayscale frames)
        # For visualization, convert to single image
        # if isinstance(obs, np.ndarray):
        #     cv2.imshow('Processed Obs', obs[:, :, -1])  # show last frame in stack
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    print(f"Episode {ep+1} finished in {step} steps, total reward: {total_reward}")

env.close()
cv2.destroyAllWindows()