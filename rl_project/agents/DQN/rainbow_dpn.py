import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ----------------------------
# 1. Noisy Linear Layer
# ----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)

# ----------------------------
# 2. Dueling Distributional Network
# ----------------------------
class DuelingDQNNetwork(nn.Module):
    def __init__(self, action_dim, atom_dim, q_type="mean"):
        super().__init__()
        self.action_dim = action_dim
        self.atom_dim = atom_dim
        self.q_type = q_type

        # Feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            NoisyLinear(7*7*64, 512),
            nn.ReLU()
        )

        self.val_fc = NoisyLinear(512, atom_dim)
        self.adv_fc = NoisyLinear(512, action_dim*atom_dim)

    def reset_noise(self):
        self.fc[0].reset_noise()
        self.val_fc.reset_noise()
        self.adv_fc.reset_noise()

    def forward(self, x):
        f = self.feature(x)
        f = f.view(f.size(0), -1)
        f = self.fc(f)

        V = self.val_fc(f).view(-1,1,self.atom_dim)
        A = self.adv_fc(f).view(-1,self.action_dim,self.atom_dim)

        if self.q_type=="mean":
            Z = V + A - A.mean(dim=1, keepdim=True)
        else:
            Z = V + A - A.max(dim=1, keepdim=True).values

        Z = F.softmax(Z, dim=2)
        return Z

# ----------------------------
# 3. Prioritized Replay Buffer
# ----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.long)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

# ----------------------------
# 4. C51 projection
# ----------------------------
def projection(next_dist, rewards, dones, gamma=0.99, Vmin=-10, Vmax=10, atom_num=51):
    delta_z = (Vmax - Vmin) / (atom_num - 1)
    support = torch.linspace(Vmin, Vmax, atom_num).unsqueeze(0)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    Tz = rewards + gamma * support * (1 - dones)
    Tz = Tz.clamp(Vmin, Vmax)

    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    m = torch.zeros_like(next_dist)
    for i in range(next_dist.size(0)):
        for j in range(next_dist.size(1)):
            l_idx = l[i,j]
            u_idx = u[i,j]
            m[i,l_idx] += next_dist[i,j] * (u[i,j]-b[i,j])
            m[i,u_idx] += next_dist[i,j] * (b[i,j]-l[i,j])
    return m


class NStepTransitionBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def push(self, transition):
        """
        transition = (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        return self._get_n_step_info()

    def _get_n_step_info(self):
        R = 0
        for idx, (_, _, reward, _, done) in enumerate(self.buffer):
            R += (self.gamma ** idx) * reward
            if done:
                break
        state, action, _, _, _ = self.buffer[0]
        next_state, _, _, _, done = self.buffer[-1]
        return (state, action, R, next_state, done)

    def reset(self):
        self.buffer.clear()


class RainbowAgent:
    def __init__(self, action_dim, atom_dim=51, gamma=0.99, n_step=3, lr=1e-4):
        self.action_dim = action_dim
        self.atom_dim = atom_dim
        self.gamma = gamma
        self.Vmin = -10
        self.Vmax = 10
        self.support = torch.linspace(self.Vmin, self.Vmax, atom_dim)

        self.q_network = DuelingDQNNetwork(action_dim, atom_dim)
        self.target_network = DuelingDQNNetwork(action_dim, atom_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.buffer = PrioritizedReplayBuffer(10000)
        self.beta = 0.4

        # n-step buffer
        self.n_step = n_step
        self.n_step_buffer = NStepTransitionBuffer(n_step, gamma)

    def reset_noise(self):
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def store_transition(self, transition):
        """
        transition = (state, action, reward, next_state, done)
        """
        n_step_transition = self.n_step_buffer.push(transition)
        if n_step_transition is not None:
            self.buffer.push(n_step_transition)
        if transition[4]:  # done
            self.n_step_buffer.reset()

    def learn(self, batch_size=32):
        if len(self.buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(batch_size, self.beta)

        self.q_network.reset_noise()
        self.target_network.reset_noise()

        Z = self.q_network(states)
        Z_pred = Z[range(batch_size), actions]

        with torch.no_grad():
            Z_next_online = self.q_network(next_states)
            Q_next_online = (Z_next_online * self.support).sum(2)
            next_actions = Q_next_online.argmax(1)

            Z_next_target = self.target_network(next_states)
            Z_next = Z_next_target[range(batch_size), next_actions]

            # Use n-step gamma here
            target_dist = projection(
                Z_next,
                rewards,
                dones,
                gamma=self.gamma ** self.n_step,
                Vmin=self.Vmin,
                Vmax=self.Vmax,
                atom_num=self.atom_dim
            )

        loss = -(target_dist * torch.log(Z_pred + 1e-8)).sum(1)
        loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        prios = loss.detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, prios)

        self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32))
        next_state, reward, done, info = env.step(action)

        agent.store_transition((state, action, reward, next_state, done))
        agent.learn(batch_size=32)

        state = next_state
