from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
from rl_project_new.agents.base import PolicyRLAgent
from rl_project_new.algorithms.ACnet import ActorCriticNetwork
import ale_py, torch
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate

from wandb_logger import WandBLogger



def setup_training_env(cfg):
    """Programmatically choose between Sync and Single env based on config."""
    gym.register_envs(ale_py)
    # A function to create one instance of the game
    def make_single_env():
        env = gym.make(cfg.env.id, render_mode=cfg.env.render_mode)  # render_mode="human" shows the game
        env = GrayscaleObservation(env, keep_dim=False) # Result: (210, 160)
        env = ResizeObservation(env, (84, 84))          # Result: (84, 84)
        env = FrameStackObservation(env, stack_size=cfg.env.frame_stack)
        return env

    if cfg.training.num_envs > 1:
        print(f"--- INITIALIZING SYNCHRONOUS MODE: {cfg.training.num_envs} ENVS ---")
        # Creates multiple environments running in parallel
        env = gym.vector.SyncVectorEnv([lambda: make_single_env() for _ in range(cfg.training.num_envs)])
        state_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.n
    else:
        print("--- INITIALIZING SINGLE-INSTANCE MODE ---")
        env = make_single_env()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
    return env, action_dim, state_dim, cfg.training.num_envs

def set_up_agent(cfg, env):
    return instantiate(
        cfg,
        in_channels=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        _convert_="none",
    )


if __name__ == "__main__":

    cfg = OmegaConf.load("rl_project_new/configs/ACnet/base.yaml")
    
    ## device
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    env, action_dim, state_dim, num_envs = setup_training_env(cfg)
 
    print(f"Environment ID: {cfg.env.id} Action Dimension: {action_dim} State Dimension: {state_dim}")

    #### agent prep
    agent = instantiate(cfg.agent)  

    #### the loop
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=cfg.training.learning_rate)

    max_episodes = cfg.training.max_episodes
    n_steps = cfg.training.n_steps
    print_every = cfg.training.print_every

    logger = WandBLogger(
        project_name="Pacman-RL", 
        run_name="A2C-NStep-Run", 
        config=OmegaConf.to_container(cfg)
    )
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done :
            for _ in range(n_steps):
                state_tensor = torch.as_tensor(state).float().to(device) / 255.0
                if state_tensor.ndim == 3: state_tensor = state_tensor.unsqueeze(0)

                ### agent preprocess
                action, value, dist = agent.get_action(state_tensor)
                if num_envs > 1:
                    # action is already a batch [num_envs]
                    next_state, reward, term, trunc, info = env.step(action.cpu().numpy())
                    # For Synch env, reward/done are arrays
                    done_tensor = torch.as_tensor(term | trunc).to(device)
                    reward_tensor = torch.as_tensor(reward).to(device)
                else:
                    # action is a single item
                    next_state, reward, term, trunc, info = env.step(action.item())
                    done_tensor = torch.tensor([term or trunc]).to(device)
                    reward_tensor = torch.tensor([reward]).to(device)
                    
                next_state, reward, term, trunc, _ = env.step(action.item())
                # Agent stores this transition in its local memory
                agent.store_transition((value, reward, action, dist))
                
                if episode % cfg.training.print_every == 0:
                    # Use the 'original' rgb frame from info if available, or current state
                    logger.store_frame(env.render())
                    
                episode_reward += reward
                state = next_state
                done = term or trunc
                if done: break
            
            ### agent memory 
            state_tensor = torch.as_tensor(state).float().to(device) / 255.0
            if state_tensor.ndim == 3: state_tensor = state_tensor.unsqueeze(0)

            loss = agent.compute_n_step_loss(state_tensor, done)
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
            optimizer.step()

            agent.memory_clear()

            logger.log_step(
                reward=reward, 
                loss=loss.item(), 
                extra_info={"grad_norm": grad_norm.item()}
            )
        
        logger.log_episode(episode_reward)

        if episode % print_every == 0:
            print(f"Episode {episode} | Reward {episode_reward}")
           

    env.close()



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