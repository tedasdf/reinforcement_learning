import torch

def replay_to_tensor(states, actions, rewards, next_states, dones, device):
    states_tensor = torch.tensor(states, dtype=torch.float, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.float, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float, device=device)
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor