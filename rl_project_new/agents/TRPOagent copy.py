import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

from rl_project.agents.Policy.ACnet import ActorCriticNetwork
from rl_project_new.agents.ACagent import ACnetRLAgent

# ---------------------------
# Rollout Buffer with GAE
# ---------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []

    def store(self, state, action, reward, mask, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.log_probs.append(log_prob)

    def compute_gae(self, network, gamma=0.99, lam=0.97):
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.log_probs).detach()
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        masks = torch.tensor(self.masks, dtype=torch.float32)

        values = network.critic(states).detach().squeeze(-1)
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
        self.states, self.actions, self.rewards, self.masks, self.log_probs = [], [], [], [], []


def set_flat_params(model, flat_params):
    prev_idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[prev_idx:prev_idx+numel].view_as(p))
        prev_idx += numel

def compute_kl(states, old_logits, network):
    new_logits = network.actor(states)
    old_dist = torch.distributions.Categorical(logits=old_logits)
    new_dist = torch.distributions.Categorical(logits=new_logits)

    kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
    return kl

def fisher_vector_product(network, v, states, old_logits):
    kl = compute_kl(states, old_logits, network)
    grads = torch.autograd.grad(kl, network.actor.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, network.actor.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
    return flat_grad_grad_kl + 0.1 * v  # Damping term


def line_search(model, loss_fn, prev_params, full_step, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = loss_fn().detach()
    for stepfrac in [0.5**i for i in range(max_backtracks)]:
        new_params = prev_params + stepfrac * full_step
        set_flat_params(model, new_params)
        new_fval = loss_fn().detach()
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / (expected_improve + 1e-8)
        if actual_improve > 0 and ratio > accept_ratio:
            return True, new_params
    return False, prev_params

def loss_fn(network, states, actions, advantages, old_log_probs):
    # surrogate loss computation placeholder
    logits = network.actor(states)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    ratios = torch.exp(log_probs - old_log_probs)
    surr = -(ratios * advantages).mean()
    return surr


class TRPOAgent(ACnetRLAgent):
    def __init__(self, network, gamma, lam, max_kl, damping):
        super().__init__(network, gamma)
        self.network = network
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.damping = damping

    def setup_network(self):
        pass

    def get_action(self, state_tensor):
        pass

    def store_transition(self, state, action, reward, next_state, done, extra):
        pass

    def process_memory(self, next_state=None):
        pass
# ---------------------------
# Training loop
# ---------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    network = ActorCriticNetwork(obs_dim, 128, act_dim)
    
    critic_optimizer = optim.Adam(network.critic.parameters(), lr=3e-3)

    max_kl = 1e-2
    num_epochs = 100
    steps_per_epoch = 2000
    l2_reg = 1e-3

    for epoch in range(num_epochs):
        buffer = RolloutBuffer()
        state = env.reset()
        ep_reward = 0

        for step in range(steps_per_epoch):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = network.actor(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())
            buffer.store(torch.tensor(state, dtype=torch.float32), action, reward, 1 - done, log_prob)

            state = next_state
            ep_reward += reward
            if done:
                state = env.reset()

        # Compute GAE
        states, actions, targets, advantages, old_log_probs = buffer.compute_gae(network)

        # Update actor
        # agent.update_actor(states, actions, advantages, old_log_probs)
        logits = network.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        surr = -(ratios * advantages).mean()

        # flat grad
        grads = torch.autograd.grad(surr, network.actor.parameters())
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        # prev params
        prev_params = torch.cat([p.data.view(-1) for p in network.actor.parameters()])

        old_logits = network.actor(states)
        
        
        # conjuagate gradient
        x = torch.zeros_like(flat_grad)
        r = flat_grad.clone()
        p = flat_grad.clone()
        rdotr = torch.dot(r, r)

        for _ in range(10):
            Ap = fisher_vector_product(network, p, states, old_logits)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        step_dir = x

        shs = 0.5 * torch.dot(step_dir, fisher_vector_product(network, step_dir, states, old_logits))
        step_size = torch.sqrt(max_kl / shs)
        full_step = step_size * step_dir
        expected_improve = torch.dot(flat_grad, full_step)

        success, new_params = line_search(network.actor, lambda: loss_fn(network, states, actions, advantages, old_log_probs), prev_params, full_step, expected_improve)
        set_flat_params(network.actor, new_params)

        # compute surrogate loss
        # Update critic
        # critic_loss = agent.update_critic(states, targets, critic_optimizer)
        values = network.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(values, targets)
        for p in network.critic.parameters():
            critic_loss += l2_reg * p.pow(2).sum()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()


        buffer.clear()
        print(f"Epoch {epoch+1} | Reward ~ {ep_reward/steps_per_epoch:.2f} | Critic Loss: {critic_loss:.3f}")
