import torch
import torch.nn as nn
import torch.nn.functional as F


class dummiNet(nn.Module):
    def __init__(
        self, state_dim, hidden_dim, act_dim, atom_dim,
        gamma, Vmin, Vmax
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim * atom_dim)
        )

        self.act_dim = act_dim
        self.atom_dim = atom_dim
        self.gamma = gamma
        self.Vmin = Vmin
        self.Vmax = Vmax

        # Fixed atom supports
        self.z = torch.linspace(Vmin, Vmax, atom_dim)
        self.delta_z = self.z[1] - self.z[0]

    def forward(self, x):
        logits = self.net(x)
        logits = logits.view(-1, self.act_dim, self.atom_dim)
        return F.softmax(logits, dim=2)  # (batch, action, atom)

    def c51_loss(self, state, action, reward, next_state, done):
        """
        Implements Algorithm 1 from the C51 paper
        """

        batch_size = state.size(0)
        device = state.device
        z = self.z.to(device)

        # -----------------------------
        # 1. Current state distribution
        # -----------------------------
        probs = self(state)  # (B, A, N)
        action = action.unsqueeze(1).unsqueeze(1)
        action = action.expand(-1, 1, self.atom_dim)
        probs_sa = probs.gather(1, action).squeeze(1)  # (B, N)

        # -----------------------------
        # 2. Next-state greedy action
        # -----------------------------
        with torch.no_grad():
            next_probs = self(next_state)  # (B, A, N)
            next_q = (next_probs * z).sum(dim=2)  # expectation
            next_action = next_q.argmax(dim=1)  # (B,)

            next_action = next_action.unsqueeze(1).unsqueeze(1)
            next_action = next_action.expand(-1, 1, self.atom_dim)
            next_probs_sa = next_probs.gather(1, next_action).squeeze(1)  # (B, N)

            # --------------------------------
            # 3. Distributional Bellman update
            # --------------------------------
            Tz = reward.unsqueeze(1) + \
                 (1 - done.unsqueeze(1)) * self.gamma * z.unsqueeze(0)
            Tz = Tz.clamp(self.Vmin, self.Vmax)

            # --------------------------------
            # 4. Projection onto fixed atoms
            # --------------------------------
            b = (Tz - self.Vmin) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros_like(next_probs_sa)

            offset = torch.linspace(
                0, (batch_size - 1) * self.atom_dim, batch_size
            ).long().unsqueeze(1).to(device)

            m.view(-1).index_add_(
                0,
                (l + offset).view(-1),
                (next_probs_sa * (u.float() - b)).view(-1)
            )

            m.view(-1).index_add_(
                0,
                (u + offset).view(-1),
                (next_probs_sa * (b - l.float())).view(-1)
            )

        # -----------------------------
        # 5. Cross-entropy loss
        # -----------------------------
        loss = -(m * torch.log(probs_sa + 1e-8)).sum(dim=1).mean()

        return loss
