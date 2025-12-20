import torch.nn as nn

"""
https://arxiv.org/pdf/1412.7755
"""

def Q_network(state_dim, hidden_dim, action_dim):
    """
    planning to make it more generalised. 
    T_T  been sick for a while 
    """
    
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,action_dim)
    )