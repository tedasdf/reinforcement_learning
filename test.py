# import gym
# import pufferlib.emulation


# if __name__ == '__main__':
#     import gymnasium as gym
#     import ale_py, os, wandb
    
    
#     os.environ['WANDB_API_KEY'] = '792864dae52b3846d1b891481c3ebb7abfe35dd9'
    
#     # Start a new wandb run to track this script.
#     run = wandb.init(
#         # Set the wandb entity where your project will be logged (generally your team name).
#         entity="my-awesome-team-name",
#         # Set the wandb project where this run will be logged.
#         project="my-awesome-project",
#         # Track hyperparameters and run metadata.
#         config={
#             "learning_rate": 0.02,
#             "architecture": "CNN",
#             "dataset": "CIFAR-100",
#             "epochs": 10,
#         },
#     )
    
    
#     gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

#     env = gym.make('ALE/Breakout-v5', render_mode="rgb_array") 
#     gymnasium_env = pufferlib.GymToGymnasium(env)
#     puffer_env = pufferlib.emulation.GymnasiumPufferEnv(gymnasium_env)
#     observations, info = puffer_env.reset()
#     action = puffer_env.action_space.sample()
#     observation, reward, terminal, truncation, info = puffer_env.step(action)
#     print(observation.shape)

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from rl_project.agents.Policy.ACnet import ActorCriticNetwork

cfg = OmegaConf.create(
    {
        "_target_": "rl_project.agents.Policy.ACnet.ActorCriticNetwork",
        "in_channels": 4,
        "hidden_dim": 256,
        "action_dim": 7
    }
)

obj_none = instantiate(cfg, _convert_="none")
print(type(obj_none))
assert isinstance(obj_none, ActorCriticNetwork)


print(obj_none)