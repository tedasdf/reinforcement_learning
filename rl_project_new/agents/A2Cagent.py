import torch
from rl_project_new.agents.base import ACnetRLAgent        


class ACnetRLAgent(ACnetRLAgent): # Inherits from the base class you defined
    def __init__(self, network, gamma, device):
        # We call the parent's __init__ to set up memory, gamma, and device
        super().__init__(network, gamma, device)
        
    def get_action(self, state_tensor):
        # We override this to add the squeeze(-1) on the value
        logits, value = self.network(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        # Squeezing ensures value is a 1D tensor [batch_size]
        return action, value.squeeze(-1), dist
    
    def process_rollout(self, next_state, done):
        # Logic for calculating the N-step targets
        with torch.no_grad():
            _, next_value = self.network(next_state)
            # R is our bootstrap starting point
            R = next_value.squeeze() * (1 - done.float())

        targets = []
        for _, reward, _, _ in reversed(self.memory):
            R = reward + self.gamma * R
            targets.append(R.detach()) # Truths should not track gradients
        
        targets.reverse()
        return targets
        
    def compute_n_step_loss(self, next_state, done):
        # Calculate targets using the function above
        targets = self.process_rollout(next_state, done)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Zip memory and targets to calculate step-by-step losses
        for (value, _, action, dist), target in zip(self.memory, targets):
            # value: predicted, target: actual reward-sum
            actor_loss, critic_loss, entropy = self.network.a2c_loss(
                value, target, action, dist
            )
            
            total_policy_loss += actor_loss
            total_value_loss += critic_loss
            total_entropy += entropy

        # Combine losses: minimize Policy + 0.5*Value - 0.01*Entropy
        n = len(self.memory)
        loss = (total_policy_loss + 0.5 * total_value_loss - 0.01 * total_entropy) / n
        
        # Important: clear memory after computing loss
        self.memory_clear()
        return loss

    # Note: We do NOT define save_checkpoint or load_checkpoint here.
    # They are inherited automatically from the parent ACnetRLAgent!