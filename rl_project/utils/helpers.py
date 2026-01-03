import torch.nn as nn

"""
https://arxiv.org/pdf/1412.7755
"""

import torch
import torch.nn as nn
import numpy as np

def build_network(
    obs_shape,
    action_dim,
    depth,
    hidden_dim=256,
    net_type="auto",   # "auto", "mlp", "cnn"
):
    """
    Returns a PyTorch network suitable for the observation shape.
    """

    if net_type == "auto":
        if len(obs_shape) == 1:
            net_type = "mlp"
        elif len(obs_shape) == 3:
            net_type = "cnn"
        else:
            raise ValueError(f"Unsupported obs shape: {obs_shape}")

    if net_type == "mlp":
        return build_mlp(np.prod(obs_shape), hidden_dim, action_dim, depth)

    if net_type == "cnn":
        return build_cnn(obs_shape, action_dim)

    raise ValueError(f"Unknown net_type: {net_type}")

def build_mlp(input_dim, hidden_dim, output_dim, depth=2):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())

    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def 