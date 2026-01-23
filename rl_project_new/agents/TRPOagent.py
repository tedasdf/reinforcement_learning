import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        # Actor
        self.actor_fc1 = nn.Linear(obs_dim, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_logits = nn.Linear(hidden_size, act_dim)

        # Critic
        self.critic_fc1 = nn.Linear(obs_dim, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_value = nn.Linear(hidden_size, 1)

    def actor(self, x):
        h = F.relu(self.actor_fc1(x))
        h = F.relu(self.actor_fc2(h))
        logits = self.actor_logits(h)
        return logits, None  # return None for compatibility

    def critic(self, x):
        h = F.relu(self.critic_fc1(x))
        h = F.relu(self.critic_fc2(h))
        value = self.critic_value(h)
        return value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []  # 1 if not done, 0 if done
        self.log_probs = []

    def store(self, state, action, reward, mask, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.log_probs.append(log_prob)

    def compute_gae(self, agent, gamma=0.99, lam=0.97):
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        masks = torch.tensor(self.masks, dtype=torch.float32)

        values = agent.network.critic(states).detach().squeeze(-1)
        advantages = torch.zeros_like(rewards)
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            gae = delta + gamma * lam * masks[t] * gae
            advantages[t] = gae
            next_value = values[t]

        targets = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, targets, advantages, old_log_probs

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []


# ---------------------------
# Utilities for flatten/unflatten parameters
# ---------------------------

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    prev_idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[prev_idx:prev_idx+numel].view_as(p))
        prev_idx += numel

def flat_grad(y, model, retain_graph=False):
    grads = torch.autograd.grad(y, model.parameters(), retain_graph=retain_graph)
    return torch.cat([g.contiguous().view(-1) for g in grads])

# ---------------------------
# Conjugate Gradient Solver
# ---------------------------
def conjugate_gradient(fvp_fn, b, n_steps=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(n_steps):
        Ap = fvp_fn(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

# ---------------------------
# Line search
# ---------------------------
def line_search(model, loss_fn, prev_params, full_step, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = loss_fn().detach()
    for stepfrac in 0.5**torch.arange(max_backtracks, dtype=torch.float32):
        new_params = prev_params + stepfrac * full_step
        set_flat_params(model, new_params)
        new_fval = loss_fn().detach()
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / (expected_improve + 1e-8)
        if actual_improve > 0 and ratio > accept_ratio:
            return True, new_params
    return False, prev_params


class TRPOAgent(nn.Module):
    def __init__(self, network, gamma=0.99, lam=0.97, max_kl=1e-2, damping=1e-1):
        super().__init__()
        self.network = network  # actor-critic network
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.damping = damping

    def compute_surrogate_loss(self, states, actions, advantages, old_log_probs):
        logits, _ = self.network.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        return -(ratios * advantages).mean()

    def compute_kl(self, states, old_logits):
        new_logits, _ = self.network.actor(states)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        kl = torch.distributions.kl_divergence(old_dist, new_dist)
        return kl.mean()

    def fisher_vector_product(self, v, states, old_logits):
        kl = self.compute_kl(states, old_logits)
        kl_grads = flat_grad(kl, self.network.actor, retain_graph=True)
        kl_v = (kl_grads * v).sum()
        fvp = flat_grad(kl_v, self.network.actor).detach()
        return fvp + self.damping * v

    def update_actor(self, states, actions, advantages, old_log_probs, n_cg_steps=10):
        # Flattened policy gradient
        surrogate_loss = self.compute_surrogate_loss(states, actions, advantages, old_log_probs)
        g = flat_grad(surrogate_loss, self.network.actor).detach()

        # Save old parameters
        prev_params = get_flat_params(self.network.actor)

        # Conjugate gradient to compute step direction
        old_logits, _ = self.network.actor(states)
        step_dir = conjugate_gradient(lambda v: self.fisher_vector_product(v, states, old_logits), g, n_steps=n_cg_steps)

        # Step size scaling
        shs = 0.5 * torch.dot(step_dir, self.fisher_vector_product(step_dir, states, old_logits))
        step_size = torch.sqrt(self.max_kl / shs)
        full_step = step_size * step_dir

        # Expected improvement
        expected_improve = torch.dot(g, full_step)

        # Line search
        def surrogate_fn():
            return self.compute_surrogate_loss(states, actions, advantages, old_log_probs)

        success, new_params = line_search(self.network.actor, surrogate_fn, prev_params, full_step, expected_improve)
        set_flat_params(self.network.actor, new_params)
        return success

    def update_critic(self, states, targets, optimizer, l2_reg=1e-3):
        values = self.network.critic(states).squeeze(-1)
        loss = F.mse_loss(values, targets)
        # L2 regularization
        for p in self.network.critic.parameters():
            loss += l2_reg * p.pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()



if __name__ == "__main__":
    import gym
    import torch.optim as optim

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Actor-critic network
    network = ActorCritic(obs_dim, act_dim)
    agent = TRPOAgent(network)
    critic_optimizer = optim.Adam(network.critic.parameters(), lr=3e-3)

    num_epochs = 200
    steps_per_epoch = 2000

    for epoch in range(num_epochs):
        buffer = RolloutBuffer()
        state = env.reset()
        ep_reward = 0

        for step in range(steps_per_epoch):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = network.actor(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())
            buffer.store(torch.tensor(state, dtype=torch.float32), action, reward, 1 - done, log_prob)

            state = next_state
            ep_reward += reward

            if done:
                state = env.reset()

        # Compute GAE & targets
        states, actions, targets, advantages, old_log_probs = buffer.compute_gae(agent)

        # Update actor (TRPO)
        agent.update_actor(states, actions, advantages, old_log_probs)

        # Update critic
        critic_loss = agent.update_critic(states, targets, critic_optimizer)

        buffer.clear()
        print(f"Epoch {epoch+1} | Episode Reward ~ {ep_reward/steps_per_epoch:.2f} | Critic Loss: {critic_loss:.3f}")
