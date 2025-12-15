import torch

x = torch.randn(4)          # random values
probs = torch.softmax(x, dim=0)

m = torch.distributions.Categorical(probs)


action = m.sample()
print(action, m.log_prob(action))