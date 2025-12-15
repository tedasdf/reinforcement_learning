import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        action_probs = self.net(state)
        action_probs = self.forward(state)
        m = torch.distributinos.Categorical(action_probs)
        action = m.sample()
        return action , m.log_prob(action)


class CrticNetwork(nn.Module):
    def __init__(self , state_dim , hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)




# if __name__ == '__main__':
#     act_net = ActorNetwork()
#     critic_net = CrticNetwork()
    
#     gamma = 0.99
#     state = env.step()

#     action, log_prob = act_net(state)

#     value = critic_net(state)
    
#     next_state, = env.step()
#     next_value = critic_net(next_state)


#     delta = reward + gammge * next_value - value


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    gamma = 0.99
    # Assume actor_net and critic_net are nn.Modules
    # Assume optimizer_actor and optimizer_critic are Adam optimizers
    optimizer_critic = torch.optim.Adam()
    optimizer_actor = torch.optim.Adam()

    # Convert states to tensors
    state_tensor = torch.tensor(state, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
    reward_tensor = torch.tensor(reward, dtype=torch.float32)

    # ----------------------
    # 1️⃣ Actor: select action
    # ----------------------
    action_probs = ActorNetwork(state_tensor)
    m = torch.distributions.Categorical(action_probs)
    action = m.sample()
    log_prob = m.log_prob(action)

    # ----------------------
    # 2️⃣ Critic: evaluate value
    # ----------------------
    value = CrticNetwork(state_tensor)
    next_value = CrticNetwork(next_state_tensor)

    # ----------------------
    # 3️⃣ Compute TD error (delta)
    # ----------------------
    delta = reward_tensor + gamma * next_value - value  # shape: [1]

    # ----------------------
    # 4️⃣ Compute losses
    # ----------------------
    # Critic: MSE loss (optional 0.5 factor)
    critic_loss = 0.5 * delta.pow(2)

    # Actor: policy gradient loss (use delta as advantage)
    actor_loss = -log_prob * delta.detach()  # detach so gradient doesn't flow into critic

    # ----------------------
    # 5️⃣ Backprop and optimizer step
    # ----------------------
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()
