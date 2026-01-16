import torch
from rl_project_new.agents.base import ACnetRLAgent        


class A2CnetRLAgent(ACnetRLAgent): # Inherits from the base class you defined
    def __init__(self, network, gamma, device):
        # We call the parent's __init__ to set up memory, gamma, and device
        super().__init__(network, gamma, device)
        
    def get_action(self, state_tensor):
        # We override this to add the squeeze(-1) on the value
        logits, value = self.network(state_tensor)
        
        if logits.ndim == 3:
            # Example: taking the specific actor for the specific batch index
            logits = torch.diagonal(logits, dim1=0, dim2=1).permute(1, 0)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        # Squeezing ensures value is a 1D tensor [batch_size]
        return action, value.squeeze(-1), log_prob, entropy
    
    def process_rollout(self, next_state):
        with torch.no_grad():
            _, next_value = self.network(next_state)
            R = next_value.squeeze(-1)

        targets = []
        for value, reward, log_prob, entropy, done in reversed(self.memory):
            R = reward + self.gamma * R * (1.0 - done.float())
            targets.append(R)

        targets.reverse()
        return targets

        
    def compute_n_step_loss(self, next_state):
        # Calculate targets using the function above
        targets = self.process_rollout(next_state)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Zip memory and targets to calculate step-by-step losses
        for (value, reward, log_prob, entropy, done), target in zip(self.memory, targets):
            # value: predicted, target: actual reward-sum
            actor_loss, critic_loss = self.network.a2c_loss(
                value, target, log_prob
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