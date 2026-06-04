import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @staticmethod
    def normalize_adjacency(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.float()
        eye = torch.eye(
            adjacency_matrix.size(0),
            dtype=adjacency_matrix.dtype,
            device=adjacency_matrix.device,
        )
        a_hat = adjacency_matrix + eye
        degree = a_hat.sum(dim=1).clamp_min(1.0)
        d_inv_sqrt = torch.pow(degree, -0.5)
        return d_inv_sqrt[:, None] * a_hat * d_inv_sqrt[None, :]

    def forward(self, x, adjacency_matrix):
        a_norm = self.normalize_adjacency(adjacency_matrix)
        return a_norm.matmul(self.linear(x))


class TwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.gcn1 = GraphConvolution(in_features, hidden_features)
        self.gcn2 = GraphConvolution(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_matrix):
        x = F.relu(self.gcn1(x, adjacency_matrix))
        x = self.dropout(x)
        return self.gcn2(x, adjacency_matrix)


# Backward-compatible aliases for older imports.
Norm_SpatiotemporalGraphConvolution = GraphConvolution
Norm_SpatiotemporalGraphDeConvolution = GraphConvolution
