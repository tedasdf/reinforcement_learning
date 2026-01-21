import torch

def replay_to_tensor(states, actions, rewards, next_states, dones, device=None):
    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    dones_tensor = torch.tensor(dones)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float)
    actions_tensor = torch.tensor(actions, dtype=torch.float)
    state = torch.tensor(state, dtype=torch.float)


    if device:
        states_tensor = states_tensor.to(device)
        actions_tensor = actions_tensor.to(device)
        rewards_tensor = rewards_tensor.to(device)
        next_states_tensor = next_states_tensor.to(device)
        dones_tensor = dones_tensor.to(device)
    
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
