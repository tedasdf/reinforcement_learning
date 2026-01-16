
import torch, os
from rl_project_new.algorithms.ACnet import ActorCriticNetwork


class ACnetRLAgent():
    def __init__(self, network, gamma, device):
        self.network = network
        self.device = device
        self.gamma = gamma
        self.memory = []
        
    def get_action(self, state_tensor):
        # Forward pass through the ACnet
        logits, value = self.network(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value, dist
    
    def store_transition(self, transition_tuple):
        # Agent stores this transition in its local memory
        self.memory.append(transition_tuple)

        
    def process_rollout(self, rollout_buffer, next_state, done):
        # 1. Get the 'Guess' for the state we landed in (BOOTSTRAPPING)
        with torch.no_grad():

            # boot strapping
            _, next_value = self.network(next_state)
            
            # If the game ended, the future value is 0
            R = next_value * (1 - done)
            
            targets = []
            # 2. Work backwards to find n-step returns
            for _, reward, _, _ in reversed(rollout_buffer):
                R = reward + self.gamma * R
                targets.append(R)
            
            targets.reverse()
            return torch.tensor(targets) # These are your "Truths"

    def memory_clear(self):
        self.memory = []

    def compute_n_step_loss(self, next_state, done):
        """
        This function:
        1. Bootstraps from next_state.
        2. Calculates N-step targets.
        3. Sums up the losses from the Algorithm.
        """
        # PART 1: Get the 'Truths' (Targets) from your bootstrapping logic
        # This uses the reversed reward summation we built earlier
        targets = self.process_rollout(self.memory, next_state, done)
        
        total_loss = 0
        
        # PART 2: Sum the losses for every step in the rollout
        # We zip the memory (Guesses) with the targets (Truths)
        for i, (value, reward, action, dist) in enumerate(self.memory):
            target = targets[i]
            
            # Call the Pure Math Algorithm (ACnet.a2c_loss)
            # We pass the values the network predicted vs the targets we calculated
            step_loss = self.network.a2c_loss(
                value=value, 
                target=target, 
                action=action, 
                dist=dist
            )
            
            total_loss += step_loss

        # Clear memory for the next interval
        self.memory_clear()
        
        # Return the average loss for this rollout
        return total_loss / len(targets)

    def save_checkpoint(self, episode, reward, path="checkpoints/"):
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_path = os.path.join(path, f"pacman_ep_{episode}.pth")
        torch.save({
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'reward': reward,
        }, file_path)
        print(f"Checkpoint saved: {file_path}")

    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint['episode']