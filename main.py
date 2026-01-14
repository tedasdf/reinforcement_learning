from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
import ale_py, wandb, torch
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate

class WandBLogger:
    def __init__(self, project_name, run_name, config=None):
        """
        Initialize a WandB run.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
        wandb.init(project=project_name, name=run_name, config=config)
        
        self.episode_frames = []
        self.current_episode = 0

    def log_step(self, reward, loss=None, value=None, extra_info=None):
        """
        Log data at every step.
        """
        log_dict = {"reward": reward}
        if loss is not None:
            log_dict["loss"] = loss
        if value is not None:
            log_dict["value"] = value
        if extra_info:
            log_dict.update(extra_info)
        wandb.log(log_dict)

    def store_frame(self, frame):
        """
        Store a frame for video logging.
        Frame shape should be (H, W, C)
        """
        self.episode_frames.append(np.transpose(frame, (2, 0, 1)))  # C,H,W for wandb.Video

    def log_episode(self, episode_reward):
        """
        Log data at the end of an episode, including video if available.
        """
        log_dict = {
            "episode": self.current_episode,
            "episode_reward": episode_reward
        }
        if self.episode_frames:
            log_dict["gameplay_video"] = wandb.Video(
                np.array(self.episode_frames), fps=30, format="mp4"
            )
        wandb.log(log_dict)
        self.episode_frames = []  # Clear frames for next episode
        self.current_episode += 1

    def finish(self):
        wandb.finish()

 
def env_prep():
    #### env prep
    # For vectorized envs
    env = gym.vector.SyncVectorEnv([make_wrapped_env for _ in range(4)])
    return env


def make_wrapped_env():
    gym.register_envs(ale_py)
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    return env

def set_up_agent(cfg, env):
    return instantiate(
        cfg,
        in_channels=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        _convert_="none",
    )


if __name__ == "__main__":

    num_envs = 4
    env_id = "ALE/MsPacman-v5"
    gamma = 0.99
    learning_rate = 1e-4
    hidden_dim = 128
    n_steps = 5  # n-step rollout
    max_episodes = 500
    print_every = 10

    cfg = OmegaConf.create(
        {
            "_target_": "rl_project.agents.Policy.ACnet.ActorCriticNetwork",
            "hidden_dim": 256
        }
    )


    env = env_prep()
 
    
    #### agent prep
    agent = set_up_agent(cfg)
    #### the loop

    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        
        state, _ = env.reset()
        epsiode_reward = 0
        done = False


        while not done :
            
            for _ in range(n_steps):
                state_tensor = torch.as_tensor(state).float().to(device) / 255.0


                ### agent preprocess


                next_state, reward , terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                
                episode_reward += reward

                ### agent memory 

                
                if done:
                    break
        ### agent postprocess

        ###
        loss = agent.loss()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





    # import wandb
    # import torch.optim as optim
    # import torch.nn as nn
    # import numpy as np
    # from omegaconf import OmegaConf
    # import gymnasium as gym
    # import ale_py

    # gym.register_envs(ale_py)


    # # cfg = OmegaConf.load("rl_project/configs/base.yaml")

    #     # Create the environment
    # env = gym.make('ALE/Pacman-v5', render_mode="human")  # render_mode="human" shows the game

    
    # num_episodes = 5  # number of episodes to run
    # max_steps = 1000  # max steps per episode

    # for episode in range(num_episodes):
    #     obs, info = env.reset()  # reset env at start of each episode
    #     total_reward = 0

    #     for step in range(max_steps):
    #         action = env.action_space.sample()  # take a random action
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         total_reward += reward

    #         if terminated or truncated:  # end episode if done
    #             break

    #     print(f"Episode {episode+1}: Total Reward = {total_reward}")

    # env.close()
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n

    # num_episodes = cfg.training.num_episodes
    # batch_size = cfg.training.batch_size
    # gamma = cfg.network.gamma
    # learning_rate = cfg.training.learning_rate
    # target_update_frequency = cfg.training.target_update_frequency
    # hidden_dim = cfg.network.hidden_dim
    # capacity = cfg.network.replay_buffer.capacity

    # if cfg.logging.use_wandb:
    #     wandb.init(
    #         project=cfg.logging.project,
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #     )

    # replay_buffer = ReplayBuffer(capacity, batch_size)

    # agent = DQNAgent(
    #     state_dim=state_dim,
    #     hidden_dim=hidden_dim,
    #     action_dim=action_dim,
    #     tau=cfg.network.tau,
    # )

    # optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    # loss_fn = nn.MSELoss() 
    # total_steps = 0  # Global tracker

    # episode_rewards = []

    # for episode in range(num_episodes):
    #     state, _ = env.reset()
    #     episode_reward = 0
    #     done = False
    #     episode_losses = []
    #     frames = [] 

    #     while not done:
    #         # 1. Select Action
    #         action, epsilon = agent.select_action(state)

    #         # 2. Interact with Environment
    #         next_state, reward, terminated, truncated, _ = env.step(action)
            
    #         # Optional: Only render if needed to save memory/time
    #         if cfg.env.render_mode == "rgb_array":
    #             frames.append(env.render())

    #         done = terminated or truncated
    #         episode_reward += reward
            
    #         # 3. Store transition in Buffer
    #         replay_buffer.store(state, action, reward, next_state, done)
            
    #         # Move to next state and increment counters
    #         state = next_state
    #         total_steps += 1
    #         agent.total_steps = total_steps # Sync internal agent steps for epsilon/updates

    #         # 4. Training Step
    #         if replay_buffer.check_length():
    #             # Sample mini-batch
    #             b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample()
                
    #             # Use the learn method with the batch
    #             predicted_q, target_q = agent.learn(
    #                 b_states, b_actions, b_rewards, b_next_states, b_dones
    #             )

    #             loss = loss_fn(predicted_q, target_q)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             if cfg.logging.use_wandb:
    #                 episode_losses.append(loss.item())

    #     # --- End of Episode Logging ---
    #     episode_rewards.append(episode_reward)
    #     print(f"Episode {episode + 1}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")

    #     if cfg.logging.use_wandb:
    #         log_dict = {
    #             "episode": episode,
    #             "episode_reward": episode_reward,
    #             "mean_loss": np.mean(episode_losses) if episode_losses else 0,
    #             "epsilon": epsilon,
    #             "total_steps": total_steps
    #         }
    #         # Log the last frame to see progress
    #         if frames:
    #             log_dict["last_frame"] = wandb.Image(frames[-1])
            
    #         wandb.log(log_dict)

    # env.close()