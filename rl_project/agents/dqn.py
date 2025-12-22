import torch, collections, random
import torch.nn as nn 
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, 
                state_dim , 
                hidden_dim, 
                action_dim,
                replay_buffer,
                tau
            ):
        nn.Module.__init__(self)
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )
        self.replay_buffer = replay_buffer # ReplayBuffer(capacity , batch_size)

        ### epsilon 
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 10000

        self.target_update_frequency = 100
        self.gamma = 0.99

        self.total_steps = 0
        # self.init_method(epsilon_decay )

        # if soft target updates 
        self.tau = tau

    @torch.no_grad() # Polyak averaging
    def soft_update_target(self):
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data +
                (1.0 - self.tau) * target_param.data
            )


    def init_method(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        loss_fn = nn.MSELoss() 
        return optimizer, loss_fn
        


    def select_action(self, state):
        epsilon = max(
            self.epsilon_end,
            self.epsilon_start -
            (self.epsilon_start - self.epsilon_end)
            * (self.total_steps / self.epsilon_decay_steps)
        )

        if random.random() < epsilon:
            action = random.randrange(self.action_dim)
        else:
            # Exploit: Choose the best action based on Q-network
            with torch.no_grad(): # No gradient calculation needed here
                # Convert state to appropriate tensor format
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                # Select action with the highest Q-value
                action = q_values.argmax().item()
        return action , epsilon

    def target_calculation(self, next_states_tensor, dones_tensor, rewards_tensor):
        with torch.no_grad():
            next_q_values_target = self.target_network(next_states_tensor)
            max_next_q_values = next_q_values_target.max(1)[0]
            # Zero out Q-values for terminal states
            max_next_q_values[dones_tensor] = 0.0
            # Calculate the target Q-value: R + gamma * max_a' Q_target(S', a')
        target_q_values = rewards_tensor + self.gamma * max_next_q_values
        return target_q_values

    def learn(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)
        
        state = next_state
        self.total_steps += 1

        ## Sample 
        if not self.replay_buffer.check_length():
            return None, None

        
        # Sample mini-batch
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.replay_buffer.sample()

        # --- Convert batch to tensors ---
        states_tensor = torch.FloatTensor(states_batch)
        actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1) # Need shape (batch_size, 1) for gather
        rewards_tensor = torch.FloatTensor(rewards_batch)
        next_states_tensor = torch.FloatTensor(next_states_batch)
        dones_tensor = torch.BoolTensor(dones_batch ,dtype=torch.bool) # Use BoolTensor for masking
        
        target_q_values = self.target_calculation(next_states_tensor, dones_tensor, rewards_tensor)

        q_values_pred = self.q_network(states_tensor)
        
        predicted_q_values = q_values_pred.gather(1, actions_tensor).squeeze(1)
        
        if self.tau is None:
            if self.total_steps % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        # SOFT update (tau provided)
        else:
            self.soft_update_target()

        return predicted_q_values, target_q_values
    

if __name__ == "__main__":
    import gymnasium as gym
    import wandb
    import torch.optim as optim
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/dqn_cartpole.yaml")

    env = gym.make(cfg.env.name, render_mode=cfg.env.render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = cfg.training.num_episodes
    batch_size = cfg.training.batch_size
    gamma = cfg.network.gamma
    learning_rate = cfg.training.learning_rate
    target_update_frequency = cfg.training.target_update_frequency
    hidden_dim = cfg.network.hidden_dim
    capacity = cfg.replay_buffer.capacity

    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )


    agent = DQNAgent(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        capacity=capacity,
        batch_size=batch_size,
        tau=cfg.training.tau,
        target_update_frequency=cfg.training.target_update_frequency
    )


    
    optimizer , loss_fn = agent.init_method()
    total_steps  = 0
    episode_rewards = [] 


    for episode in range(num_episodes):

        state, _ = env.reset()
        episode_reward = 0
        done = False

        if cfg.logging.use_wandb:
            episode_losses = []
            frames = [] 
        
        while not done:
            # 1. Select Action
            action, epsilon = agent.select_action(state )

            # 2. Interact with Environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()  # Returns an RGB array

            done = terminated or truncated
            episode_reward += reward
            
            predicted_q_value, target_q_values = agent.learn(
                state, action, reward, next_state, done
            )


            if predicted_q_value is None:
                continue

            loss = loss_fn(predicted_q_value, target_q_values)


            if cfg.logging.use_wandb:
                episode_losses.append(loss.item())
                frames.append(frame)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if done:
                break
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Epsilon = {epsilon:.3f}") # Provide feedback

        mean_loss = np.mean(episode_losses) if episode_losses else 0.0

        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "mean_loss": mean_loss,
            "epsilon": epsilon,
            "total_steps": total_steps,
            "frame": wandb.Image(frames[-1])
        })

    env.close()