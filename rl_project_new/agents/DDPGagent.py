import torch

from rl_project_new.agents.base import BaseAgent
from rl_project_new.agents.utils import replay_to_tensor
from rl_project_new.buffer.replay_buffer import ReplayBuffer


class DDPGnetRLAgent(BaseAgent):
    def __init__(self, replay_buffer: ReplayBuffer, network, gamma, critic_lr, actor_lr , device, action_space):
        super().__init__(device, network, action_space)
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def setup_network(self):

        return {
            'actor':    torch.optim.Adam(self.network.parameters(), lr=self.actor_lr),
            'critic':   torch.optim.Adam(self.network.parameters(), lr=self.critic_lr),
        }

    
    def get_action(self, state_tensor):
        action, value = self.network(state_tensor)
        return action, {'value': value.squeeze(-1)}

    
    def store_transition(self, state, action, reward, next_state, done, extra=None):
        self.replay_buffer.store(state, action, reward, next_state, done)


    def process_memory(self):
        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = \
    replay_to_tensor(*self.replay_buffer.sample(), device=self.device)

        with torch.no_grad():
            _, next_q_values_target = self.network(next_states_tensor)
            target_q_values = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_values_target

        return target_q_values, states_tensor, actions_tensor

    def memory_clear(self):
        pass  # replay buffer persists; no action needed


    def compute_n_step_loss(self, state_tensor):
        if not self.replay_buffer.check_length():
            return None
        target_q_values, states_tensor, actions_tensor = self.process_memory()

        current_q_values = self.network.critic(states_tensor, actions_tensor)

        critic_loss = self.network.loss(current_q_values, target_q_values)

        value_tensors, _ = self.network(states_tensor)
        actor_loss = -value_tensors.mean()
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        }