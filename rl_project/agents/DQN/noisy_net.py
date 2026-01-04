import torch
import torch.nn as nn
import torch.nn.functional as F



'''
https://arxiv.org/pdf/1706.10295
'''



class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5, factorised=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised

        # Learnable parameters
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))

        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))

        # Noise buffers
        if self.factorised:
            self.register_buffer("eps_in", torch.empty(in_features))
            self.register_buffer("eps_out", torch.empty(out_features))
        else:
            self.register_buffer("eps_w", torch.empty(out_features, in_features))
            self.register_buffer("eps_b", torch.empty(out_features))

        self.reset_parameters(std_init)
        self.reset_noise()

    def reset_parameters(self, std_init):
        bound = 1 / self.in_features ** 0.5
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_b.data.uniform_(-bound, bound)

        self.sigma_w.data.fill_(std_init / self.in_features ** 0.5)
        self.sigma_b.data.fill_(std_init / self.out_features ** 0.5)

    def reset_noise(self):
        if self.factorised:
            self.eps_in.normal_()
            self.eps_out.normal_()
        else:
            self.eps_w.normal_()
            self.eps_b.normal_()

    def f(self, x):
        # factorised noise transform
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def forward(self, x):
        if self.training:
            if self.factorised:
                eps_w = torch.outer(self.f(self.eps_out), self.f(self.eps_in))
                w = self.mu_w + self.sigma_w * eps_w
                b = self.mu_b + self.sigma_b * self.f(self.eps_out)
            else:
                w = self.mu_w + self.sigma_w * self.eps_w
                b = self.mu_b + self.sigma_b * self.eps_b
        else:
            w = self.mu_w
            b = self.mu_b

        return F.linear(x, w, b)
