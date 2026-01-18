import torch, os
from rl_project_new.agents.base import BaseAgent

class ACnetRLAgent(BaseAgent):
    def __init__(self, network, gamma, lr, device, action_space):
        super().__init__(device, network, action_space)
        self.gamma = gamma
        self.lambda_ = 0.95
        self.lr = lr
    
    def setup_network(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)
    
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
        return action, {
            'value': value.squeeze(-1),
            'log_prob': log_prob,
            'entropy': entropy
        }
    
    def store_transition(self, state, action, reward, next_state, done, extra):
        # Agent stores this transition in its local memory
        self.memory.append((
            extra['value'],
            reward,
            extra['log_prob'],
            extra['entropy'],
            done
        ))

    def process_gae(self, next_state):
        """
        Computes GAE advantages and value targets.
        Memory stores: (value, reward, log_prob, entropy, done)
        """

        device = next_state.device

        with torch.no_grad():
            _, next_value = self.network(next_state)
            next_value = next_value.squeeze(-1)  # [num_envs]

            gae = torch.zeros_like(next_value)
            advantages = []
            targets = []

            for value, reward, _, _, done in reversed(self.memory):
                reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
                done   = torch.as_tensor(done, dtype=torch.float32, device=device)
                value  = value.squeeze(-1)

                # TD residual
                delta = reward + self.gamma * next_value * (1.0 - done) - value

                # GAE recursion
                gae = delta + self.gamma * self.lambda_ * (1.0 - done) * gae

                advantages.append(gae)
                targets.append(gae + value)

                next_value = value

            advantages.reverse()
            targets.reverse()

            return torch.stack(targets), torch.stack(advantages)

    def process_memory(self, next_state):
        with torch.no_grad():
            # 1. Bootstrapping: get value for next state
            _, next_value = self.network(next_state)
            next_value = next_value.squeeze(-1)  # shape [num_envs]

            # 2. Initial R (bootstrap), done masking
            # Memory stores done per step, shape [num_envs]
            # Convert done to float tensor on correct device
            last_done = torch.zeros_like(next_value)  # default 0 for multi-env
            R = next_value * (1.0 - last_done)

            targets = []
            # 3. Backwards n-step return
            for _, reward, _, _, done in reversed(self.memory):
                # Convert reward and done to tensors on device
                reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                done_tensor   = torch.as_tensor(done, dtype=torch.float32, device=self.device)

                R = reward_tensor + self.gamma * R * (1.0 - done_tensor)
                targets.append(R)

            targets.reverse()
            # Stack into [n_steps, num_envs]
            return torch.stack(targets)

    def memory_clear(self):
        self.memory = []

    def compute_n_step_loss(self, next_state, use_gae=False):
        if use_gae:
            targets, advantages = self.process_gae(next_state)
        else:
            targets = self.process_memory(next_state)
            values = torch.stack([v for v, _, _, _, _ in self.memory])
            advantages = targets - values
        
        total_policy_loss = 0
        total_value_loss  = 0
        total_entropy     = 0

        for (value, _, log_prob, entropy, _), target, adv in zip(
                self.memory, targets, advantages
            ):
            
            actor_loss, critic_loss = self.network.loss(adv, log_prob, target)

            # Entropy regularization
            total_policy_loss += actor_loss
            total_value_loss  += critic_loss
            total_entropy     += entropy

        n = len(self.memory)
        loss = (total_policy_loss + 0.5 * total_value_loss - 0.01 * total_entropy) / n

        self.memory_clear()
        return loss
    

    def update_networks(self, loss, optimizers):

        optimizers.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        optimizers.step()

        self.soft_update()

        return grad_norm