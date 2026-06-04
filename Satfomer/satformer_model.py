import math

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
        return self.normalize_adjacency(adjacency_matrix).matmul(self.linear(x))


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


class ASSIT(nn.Module):
    def __init__(self, region_size, center_window, heads, head_dim):
        super().__init__()
        self.region_size = region_size
        self.center_window = center_window
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(1, self.inner_dim)
        self.key = nn.Linear(1, self.inner_dim)
        self.value = nn.Linear(1, self.inner_dim)
        self.output = nn.Linear(self.inner_dim, 1)
        self.sparse_regularizer = nn.Parameter(torch.ones(heads, 1, 1))
        self.sparse_scale = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        height, width = x.shape
        patches, padded_shape = self._split_regions(x)
        outputs = [self._attend_region(patch) for patch in patches]
        y = self._merge_regions(outputs, padded_shape, x.device, x.dtype)
        return y[:height, :width]

    def _split_regions(self, x):
        region = self.region_size
        height, width = x.shape
        pad_h = (region - height % region) % region
        pad_w = (region - width % region) % region
        padded = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = padded.shape
        patches = []
        for row in range(0, padded_h, region):
            for col in range(0, padded_w, region):
                patches.append(padded[row : row + region, col : col + region])
        return patches, (padded_h, padded_w)

    def _attend_region(self, patch):
        region = self.region_size
        tokens = patch.reshape(region * region, 1)
        q = self.query(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)
        k = self.key(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)
        v = self.value(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        center_mask = self._center_window_mask(region, patch.device)
        scores = scores.masked_fill(~center_mask[None, None, :], -1e9)
        attention = F.softmax(scores, dim=-1)
        sparse_gate = F.relu(1.0 - self.sparse_regularizer * attention)
        sparse_attention = F.softmax(self.sparse_scale * sparse_gate * attention, dim=-1)

        output = torch.matmul(sparse_attention, v)
        output = output.transpose(0, 1).reshape(region * region, self.inner_dim)
        return self.output(output).reshape(region, region)

    def _center_window_mask(self, region, device):
        window = min(self.center_window, region)
        start = (region - window) // 2
        end = start + window
        mask = torch.zeros(region, region, dtype=torch.bool, device=device)
        mask[start:end, start:end] = True
        return mask.reshape(region * region)

    def _merge_regions(self, patches, padded_shape, device, dtype):
        region = self.region_size
        padded_h, padded_w = padded_shape
        output = torch.zeros(padded_h, padded_w, device=device, dtype=dtype)
        index = 0
        for row in range(0, padded_h, region):
            for col in range(0, padded_w, region):
                output[row : row + region, col : col + region] = patches[index]
                index += 1
        return output


class SatFormerBlock(nn.Module):
    def __init__(self, feature_dim, region_size, center_window, heads, mlp_ratio=2.0):
        super().__init__()
        head_dim = max(1, math.ceil(feature_dim / heads))
        self.norm1 = nn.LayerNorm(feature_dim)
        self.assit = ASSIT(region_size, center_window, heads, head_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        x = x + self.assit(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioTemporalModule(nn.Module):
    def __init__(self, feature_dim, gcn_hidden_dim, region_size, center_window, heads, dropout=0.0):
        super().__init__()
        self.norm_gcn = nn.LayerNorm(feature_dim)
        self.gcn = TwoLayerGCN(feature_dim, gcn_hidden_dim, feature_dim, dropout=dropout)
        self.satformer = SatFormerBlock(feature_dim, region_size, center_window, heads)

    def forward(self, x, adjacency_matrix):
        x = x + self.gcn(self.norm_gcn(x), adjacency_matrix)
        return self.satformer(x)


class TransferModule(nn.Module):
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
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        outputs = []
        for t in range(x.size(0)):
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


class SatFormer(nn.Module):
    def __init__(
        self,
        num_nodes,
        feature_dim=128,
        gcn_hidden_dim=128,
        num_modules=10,
        region_size=16,
        center_window=16,
        heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.input_projection = nn.Linear(num_nodes, feature_dim)
        self.encoder = nn.ModuleList(
            [
                SpatioTemporalModule(
                    feature_dim,
                    gcn_hidden_dim,
                    region_size,
                    center_window,
                    heads,
                    dropout=dropout,
                )
                for _ in range(num_modules)
            ]
        )
        self.transfer = TransferModule(feature_dim)
        self.decoder = nn.ModuleList(
            [
                SpatioTemporalModule(
                    feature_dim,
                    gcn_hidden_dim,
                    region_size,
                    center_window,
                    heads,
                    dropout=dropout,
                )
                for _ in range(num_modules)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_nodes),
            nn.ReLU(),
        )

    def forward(self, x, adjacency_matrices):
        time_outputs = []
        for time_step in range(x.size(-1)):
            h = self.input_projection(x[:, :, time_step])
            adjacency = adjacency_matrices[:, :, time_step]
            for module in self.encoder:
                h = module(h, adjacency)
            time_outputs.append(h)

        h = self.transfer(torch.stack(time_outputs, dim=0))

        decoded = []
        for time_step in range(h.size(0)):
            step_h = h[time_step]
            adjacency = adjacency_matrices[:, :, time_step]
            for module in self.decoder:
                step_h = module(step_h, adjacency)
            decoded.append(self.output_projection(step_h).unsqueeze(-1))
        return torch.cat(decoded, dim=-1)
