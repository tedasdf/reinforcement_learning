import torch, collections, random
import torch.nn as nn 
import numpy as np

# Structure for the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        # Use a deque as it automatically handles max size
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

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

    def check_length(self):
        return len(self.buffer) > self.batch_size


class DQNAgent(nn.Module):
    def __init__(self, 
                 state_dim , 
                 hidden_dim, 
                 action_dim, 
                 epsilon_decay,
                 capacity,

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
        self.replay_buffer = ReplayBuffer()

        ### epsilon 
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 10000

        self.target_update_frequency = 100
        
        self.init_method(epsilon_decay )

    def init_method(self, epsilon_decay):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    @staticmethod   
    def QNetwork():
        return 
    

    def forward(self, x):
        return self.q_network(x)

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()
        
    def learn(self):

        ## Sample 
        if len(self.replay_buffer) > 
