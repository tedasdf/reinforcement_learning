import torch
from rl_project.agents.dqn import DQNAgent


class DDQNAgent(DQNAgent):
    def __init__(self, 
                state_dim , 
                hidden_dim, 
                action_dim,
                capacity,
                batch_size,
                tau):
        super().__init__(
            state_dim , 
            hidden_dim, 
            action_dim,
            capacity,
            batch_size,
            tau
        )


    def target_calculation(self, next_states_tensor, dones_tensor, rewards_tensor):
        with torch.no_grad():
            # 1. Action selection using ONLINE network
            next_actions = self.q_network(next_states_tensor).argmax(dim=1)

            # 2. Action evaluation using TARGET network
            next_q_values_target = self.target_network(next_states_tensor)
            max_next_q_values = next_q_values_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            max_next_q_values[dones_tensor] = 0.0
            target_q_values = rewards_tensor + self.gamma * max_next_q_values
        return target_q_values
