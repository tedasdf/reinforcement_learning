
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make('CartPole-v1')

# Get state and action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimensions: {state_dim}")
print(f"Action dimensions: {action_dim}")


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # Output raw scores (logits) for actions
        action_scores = self.fc2(x)
        # Convert scores to probabilities using softmax
        # Use dim=-1 to apply softmax across the action dimension
        return F.softmax(action_scores, dim=-1)

# Instantiate the policy network and optimizer
policy_net = PolicyNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3) # Learning rate


# Define a structure to store trajectory data
SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) # Value here means Return G_t

def select_action(state):
    """Selects an action based on the policy network's output probabilities."""
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy_net(state)
    # Create a categorical distribution over the list of probabilities of actions
    m = torch.distributions.Categorical(probs)
    # Sample an action from the distribution
    action = m.sample()
    # Store the log probability of the sampled action
    policy_net.saved_action = SavedAction(m.log_prob(action), 0) # Placeholder for G_t
    return action.item()

def calculate_returns(rewards, gamma=0.99):
    """Calculates discounted returns for an episode."""
    R = 0
    returns = []
    # Iterate backwards through the rewards
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R) # Prepend to maintain order
    # Normalize returns (optional but often helpful)
    returns = torch.tensor(returns, device=device)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
    return returns

def finish_episode(episode_rewards):
    """Performs the REINFORCE update at the end of an episode."""
    policy_loss = []
    returns = calculate_returns(episode_rewards)

    # Retrieve saved log probabilities and associate them with calculated returns
    for (log_prob, _), G_t in zip(policy_net.all_saved_actions, returns):
         # REINFORCE objective: maximize log_prob * G_t
         # Loss is the negative objective
        policy_loss.append(-log_prob * G_t)

    # Sum the losses for the episode
    optimizer.zero_grad() # Reset gradients
    loss = torch.stack(policy_loss).sum() # Combine losses
    loss.backward() # Compute gradients
    optimizer.step() # Update network weights

    # Clear saved actions for the next episode
    del policy_net.all_saved_actions[:]

# Training parameters
num_episodes = 1000
gamma = 0.99
log_interval = 50 # Print status every 50 episodes
max_steps_per_episode = 1000 # Prevent excessively long episodes

all_episode_rewards = []
episode_durations = []

# Main training loop
for i_episode in range(num_episodes):
    state, _ = env.reset()
    episode_rewards = []
    policy_net.all_saved_actions = [] # Store log_probs for the episode

    for t in range(max_steps_per_episode):
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        policy_net.all_saved_actions.append(policy_net.saved_action)
        episode_rewards.append(reward)
        state = next_state

        if done:
            break

    # Episode finished, update policy
    finish_episode(episode_rewards)

    # Logging
    total_reward = sum(episode_rewards)
    all_episode_rewards.append(total_reward)
    episode_durations.append(t + 1)

    if i_episode % log_interval == 0:
        avg_reward = np.mean(all_episode_rewards[-log_interval:])
        print(f'Episode {i_episode}\tAverage Reward (last {log_interval}): {avg_reward:.2f}\tLast Duration: {t+1}')

    # Optional: Stop training if the problem is solved
    # CartPole-v1 is considered solved if avg reward > 475 over 100 consecutive episodes
    if len(all_episode_rewards) > 100:
         if np.mean(all_episode_rewards[-100:]) > 475:
              print(f'\nSolved in {i_episode} episodes!')
              break

env.close()

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(all_episode_rewards)
plt.title('Episode Rewards over Time (REINFORCE)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Calculate and plot rolling average
rolling_avg = np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid')
plt.plot(np.arange(99, len(all_episode_rewards)), rolling_avg, label='100-episode rolling average', color='orange')
plt.legend()
plt.grid(True)
plt.show()