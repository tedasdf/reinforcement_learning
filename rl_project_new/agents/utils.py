import torch

def replay_to_tensor(states, actions, rewards, next_states, dones, device=None):
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
    
    if device:
        states_tensor = states_tensor.to(device)
        actions_tensor = actions_tensor.to(device)
        rewards_tensor = rewards_tensor.to(device)
        next_states_tensor = next_states_tensor.to(device)
        dones_tensor = dones_tensor.to(device)
    
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
