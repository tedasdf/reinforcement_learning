import numpy as np
import torch.nn as nn
import torch 

class reinforce(nn.Module):
    def __init__(
            self,
            states,
            actions,
            hidden_dim,
            init_type='', 
            discount_rate = 1,
            alpha= 0.05
            ):
        super().__init__()
        self.discount_rate = discount_rate
        
        self.init_action_state( states, actions, hidden_dim)
        

    def init_action_state(self, state_dim, action_dim, hidden_dim):
        self.pi_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
  

    def select_action(self, state):
        action_probs = self.forward(state)
        m = torch.distributinos.Categorical(action_probs)
        action = m.sample()
        return action.item() , m.log_prob(action)


    def update(self, all_action_probs, rewards ):
        # calculating returns
        G_t = 0
        return_list = [] 
        for reward in reversed(rewards):
            G_t = reward + self.discount_rate * G_t
            return_list.insert(0, G_t)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        policy_loss = []

        for log_prob, G_t in zip(all_action_probs, returns):
            policy_loss.append(-log_prob * G_t)
        
        return torch.stack(policy_loss).sum() # Combine losses
    
    def forward(self, state):
        action_probs = self.pi_policy(state)
        return action_probs
    
