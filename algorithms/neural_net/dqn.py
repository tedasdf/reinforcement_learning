import collections
import random
import numpy as np
import gymnasium as gym



# Structure for the Replay Buffer
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

import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.layer1(state))
        x = self.relu(self.layer2(x))
        q_values = self.output_layer(x) # Linear activation for Q-values
        return q_values

state_dim = 4 # CartPole state size
action_dim = 2 # CartPole action size
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)

# Initialize target network weights to match Q-network
target_network.load_state_dict(q_network.state_dict())
target_network.eval() # Set target network to evaluation mode

# Optimizer (e.g., Adam) for the Q-network
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 10000
current_step = 0

def select_action(state, q_network, current_step):
    # Calculate current epsilon based on decay schedule
    epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (current_step / epsilon_decay_steps))

    if random.random() < epsilon:
        # Explore: Choose a random action
        action = env.action_space.sample()
    else:
        # Exploit: Choose the best action based on Q-network
        with torch.no_grad(): # No gradient calculation needed here
            # Convert state to appropriate tensor format
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            # Select action with the highest Q-value
            action = q_values.argmax().item()
    return action , epsilon



if __name__ == "__main__":
    import wandb


    # --- Hyperparameters ---
    num_episodes = 1000
    replay_buffer_capacity = 10000
    batch_size = 64
    gamma = 0.99 # Discount factor
    target_update_frequency = 100 # Update target network every C steps
    learning_rate = 0.001
    # epsilon parameters defined earlier...

    # --- Initialization ---
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    q_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    optimizer = optim.Adam(q_network.parameters(), lr=0.001) # Initialize optimizer (e.g., Adam) for q_network
    loss_fn = nn.MSELoss() # Or Huber loss

    total_steps = 0
    episode_rewards = []

    run = wandb.init(
        project="brick-by-brick",
        # entity="teedsingyau",
        config={
            "env": "CartPole-v1",
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": replay_buffer_capacity,
            "target_update_frequency": target_update_frequency,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
        },
    )


    # --- Training ---
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        episode_losses = []
        frames = [] 
        while not done:
            # 1. Select Action
            action, epsilon = select_action(state, q_network, total_steps)

            # 2. Interact with Environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()  # Returns an RGB array
            frames.append(frame)
            done = terminated or truncated
            episode_reward += reward

            # 3. Store Transition
            replay_buffer.store(state, action, reward, next_state, done)

            # Update current state
            state = next_state
            total_steps += 1

            # 4. Sample and Learn (if buffer has enough samples)
            if len(replay_buffer) > batch_size:
                # Sample mini-batch
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)

                # --- Convert batch to tensors ---
                states_tensor = torch.FloatTensor(states_batch)
                actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1) # Need shape (batch_size, 1) for gather
                rewards_tensor = torch.FloatTensor(rewards_batch)
                next_states_tensor = torch.FloatTensor(next_states_batch)
                dones_tensor = torch.BoolTensor(dones_batch) # Use BoolTensor for masking

                # --- Calculate Target Q-values ---
                with torch.no_grad(): # No gradients needed for target calculation
                    # Get Q-values for next states from the target network
                    next_q_values_target = target_network(next_states_tensor)
                    # Select the best action's Q-value (max over actions)
                    max_next_q_values = next_q_values_target.max(1)[0]
                    # Zero out Q-values for terminal states
                    max_next_q_values[dones_tensor] = 0.0
                    # Calculate the target Q-value: R + gamma * max_a' Q_target(S', a')
                    target_q_values = rewards_tensor + gamma * max_next_q_values

                # --- Calculate Predicted Q-values ---
                # Get Q-values for the current states from the main Q-network
                q_values_pred = q_network(states_tensor)
                # Select the Q-value corresponding to the action actually taken in the batch
                # Use gather() to select Q-values based on actions_tensor indices
                predicted_q_values = q_values_pred.gather(1, actions_tensor).squeeze(1)

                # --- Calculate Loss ---
                loss = loss_fn(predicted_q_values, target_q_values)
                episode_losses.append(loss.item())

                # --- Perform Gradient Descent ---
                optimizer.zero_grad()
                loss.backward()
                # Optional: Clip gradients to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                optimizer.step()

            # 5. Update Target Network periodically
            if total_steps % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

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