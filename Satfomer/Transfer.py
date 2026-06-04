import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferModule(nn.Module):
    """Temporal self-attention bridge from encoder states to decoder states."""

    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        self.history_weight = nn.Parameter(torch.tensor(1.0))
        self.current_weight = nn.Parameter(torch.tensor(1.0))
        self.scale = feature_dim ** -0.5

    def forward(self, x):
        # x: [time, nodes, features]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        time_steps = x.size(0)
        outputs = []

        for t in range(time_steps):
            scores = torch.einsum("nf,knf->nk", q[t], k[: t + 1]) * self.scale
            if t > 0:
                weights = torch.ones(t + 1, dtype=scores.dtype, device=scores.device)
                weights = weights * self.history_weight
                weights[-1] = self.current_weight
                scores = scores * weights[None, :]
            attention = F.softmax(scores, dim=-1)
            context = torch.einsum("nk,knf->nf", attention, v[: t + 1])
            outputs.append(torch.sigmoid(context) * v[t])

        return self.output(torch.stack(outputs, dim=0))


class MLPModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
