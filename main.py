from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
import ale_py, torch
import numpy as np
import argparse
from omegaconf import OmegaConf
from hydra.utils import instantiate

from wandb_logger import WandBLogger
from gymnasium import spaces


def setup_training_env(cfg):
    """Programmatically choose between Sync and Single env based on config."""
    gym.register_envs(ale_py)
    # A function to create one instance of the game
    def make_single_env():
        env = gym.make(cfg.env.id, render_mode=cfg.env.render_mode)  # render_mode="human" shows the game
        obs_shape = env.observation_space.shape
        # Image-based envs: (H, W, C)
        if obs_shape is not None and len(obs_shape) == 3:
            env = GrayscaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, (84, 84))
            env = FrameStackObservation(env, stack_size=cfg.env.frame_stack)

        return env

    if cfg.training.num_envs > 1:
        print(f"--- INITIALIZING SYNCHRONOUS MODE: {cfg.training.num_envs} ENVS ---")
        # Creates multiple environments running in parallel
        env = gym.vector.SyncVectorEnv([lambda: make_single_env() for _ in range(cfg.training.num_envs)])
        state_dim = env.single_observation_space.shape[0]
    
    else:
        print("--- INITIALIZING SINGLE-INSTANCE MODE ---")
        env = make_single_env()
        state_dim = env.observation_space.shape[0]
    
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        action_dim = action_space.n
    elif isinstance(action_space, spaces.Box):
        action_dim = action_space.shape[0]
    return env, action_dim, state_dim, cfg.training.num_envs

def set_up_agent(cfg, env):
    return instantiate(
        cfg,
        in_channels=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        _convert_="none",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="RL Training Script")
    parser.add_argument("--config_file", type=str, default="rl_project_new/configs/ACnet/base.yaml")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render_mode", type=str, default="rgb_array")

    # Always define both learning rates
    
    parser.add_argument("--device", type=str, default="cpu")
    
    args, unknown = parser.parse_known_args()

    it = iter(unknown)
    extra_args = dict(zip(it, it))
    return args, extra_args


if __name__ == "__main__":

    args, extra_args = parse_args()
    
    cfg = OmegaConf.load(args.config_file)
    
    ## device
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    env, action_dim, state_dim, num_envs = setup_training_env(cfg)
 
    print(f"Environment ID: {cfg.env.id} Action Dimension: {action_dim} State Dimension: {state_dim}")


    #### agent prep
    agent = instantiate(
        cfg.agent,
        action_space= env.action_space,
        network={
        "in_channels": state_dim,
        "action_dim": action_dim
        }
    )

    #### the loop
    optimizers = agent.setup_network()

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
        if num_envs == 1:
            state = np.expand_dims(state, 0)

        episode_reward = np.zeros(num_envs, dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.bool_)

        while not done.all():
            for _ in range(n_steps):

                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device) / 255.0
                if state_tensor.ndim == 3:
                    state_tensor = state_tensor.unsqueeze(0)

                ### agent preprocess
                action_tensor, extra = agent.get_action(state_tensor)
                action = agent.format_action(action_tensor)

                if num_envs == 1:
                    next_state, reward, term, trunc, info = env.step(action)
                    reward = np.array([reward], dtype=np.float32)
                    term   = np.array([term], dtype=np.bool_)
                    trunc  = np.array([trunc], dtype=np.bool_)
                    next_state = np.expand_dims(next_state, 0)
                else:
                    next_state, reward, term, trunc, info = env.step(action)
                    reward = reward.astype(np.float32)
                    term   = term.astype(np.bool_)
                    trunc  = trunc.astype(np.bool_)

                done = np.logical_or(term, trunc)   # shape: [num_envs]

                    
                # Agent stores this transition in its local memory
                agent.store_transition(state, action, reward, next_state, done, extra)

                # if episode % cfg.training.print_every == 0:
                    # Use the 'original' rgb frame from info if available, or current state
                    # logger.store_frame(env.render())
                    
                episode_reward += reward.mean()
                state = next_state
                # For single env, break if done
                if num_envs == 1 and done[0]:
                    break

                # For multi-env, optionally reset finished envs individually
                if num_envs > 1:
                    for i, d in enumerate(done):
                        if d:
                            state[i], _ = env.envs[i].reset()
                            episode_reward[i] = 0 
                
            # Convert last state to tensor for bootstrap
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device) / 255.0
            if state_tensor.ndim == 3:
                state_tensor = state_tensor.unsqueeze(0)

            # Compute loss from n-step rollout
            loss = agent.compute_n_step_loss(state_tensor)
            if loss is None:
                continue
            
            grad_norms = {}

            if isinstance(optimizers, dict):
                if "critic" in optimizers:
                    optimizers["critic"].zero_grad()
                    loss["critic_loss"].backward()
                    grad_norms["critic_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        agent.network.critic.parameters(), 0.5
                    )
                    optimizers["critic"].step()

                if "actor" in optimizers:
                    optimizers["actor"].zero_grad()
                    loss["actor_loss"].backward()
                    grad_norms["actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        agent.network.actor.parameters(), 0.5
                    )
                    optimizers["actor"].step()

                logger.log_step(
                    reward=reward,
                    loss=loss,
                    extra_info={k: v.item() for k, v in grad_norms.items()}
                )

            


            
        logger.log_episode(episode_reward)

        if episode % print_every == 0:
            print(f"Episode {episode} | Reward {episode_reward}")
           

    env.close()
