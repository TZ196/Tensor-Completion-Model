import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ODGraphConvolution(nn.Module):
    """Graph convolution over both source and destination dimensions of OD flow."""

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
        source_context = torch.einsum("ab,bjc->ajc", a_norm, x)
        destination_context = torch.einsum("jb,ibc->ijc", a_norm, x)
        topology_context = 0.5 * (source_context + destination_context)
        return self.linear(topology_context)


class TwoLayerODGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.gcn1 = ODGraphConvolution(in_features, hidden_features)
        self.gcn2 = ODGraphConvolution(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_matrix):
        x = F.relu(self.gcn1(x, adjacency_matrix))
        x = self.dropout(x)
        return self.gcn2(x, adjacency_matrix)


class ASSIT(nn.Module):
    """Adaptive Sparse Spatio-Temporal Attention on local OD regions."""

    def __init__(self, feature_dim, region_size, center_window, heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.region_size = region_size
        self.center_window = center_window
        self.heads = heads
        self.head_dim = max(1, math.ceil(feature_dim / heads))
        self.inner_dim = heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(feature_dim, self.inner_dim)
        self.key = nn.Linear(feature_dim, self.inner_dim)
        self.value = nn.Linear(feature_dim, self.inner_dim)
        self.output = nn.Linear(self.inner_dim, feature_dim)
        self.sparse_regularizer = nn.Parameter(torch.ones(heads, 1, 1))
        self.sparse_scale = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        squeeze_time = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_time = True
        if x.dim() != 4:
            raise ValueError("ASSIT expects [H,W,C] or [T,H,W,C], got %s" % (tuple(x.shape),))

        time_steps, height, width, channels = x.shape
        patches, valid_masks, padded_shape = self._split_regions(x)
        outputs = [
            self._attend_region(patch, valid_mask)
            for patch, valid_mask in zip(patches, valid_masks)
        ]
        y = self._merge_regions(outputs, padded_shape, channels, x.device, x.dtype)
        y = y[:, :height, :width, :]
        if squeeze_time:
            return y.squeeze(0)
        return y

    def _split_regions(self, x):
        region = self.region_size
        time_steps, height, width, channels = x.shape
        pad_h = (region - height % region) % region
        pad_w = (region - width % region) % region
        padded_h = height + pad_h
        padded_w = width + pad_w
        padded = x.new_zeros(time_steps, padded_h, padded_w, channels)
        padded[:, :height, :width, :] = x
        valid = torch.ones(height, width, dtype=torch.bool, device=x.device)
        valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)

        patches = []
        valid_masks = []
        for row in range(0, padded_h, region):
            for col in range(0, padded_w, region):
                patches.append(padded[:, row : row + region, col : col + region, :])
                valid_masks.append(valid[row : row + region, col : col + region])
        return patches, valid_masks, (padded_h, padded_w)

    def _attend_region(self, patch, valid_mask):
        region = self.region_size
        time_steps = patch.size(0)
        tokens = patch.reshape(time_steps * region * region, self.feature_dim)
        q = self.query(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)
        k = self.key(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)
        v = self.value(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        center_mask = self._center_window_mask(region, patch.device)
        valid_key_mask = valid_mask.reshape(region * region)
        spatial_key_mask = center_mask & valid_key_mask
        key_mask = spatial_key_mask.repeat(time_steps)
        if not torch.any(key_mask):
            key_mask = valid_key_mask.repeat(time_steps)
        scores = scores.masked_fill(~key_mask[None, None, :], -65504.0)

        attention = F.softmax(scores, dim=-1)
        sparse_gate = F.relu(1.0 - self.sparse_regularizer * attention)
        sparse_attention = F.softmax(self.sparse_scale * sparse_gate * attention, dim=-1)

        output = torch.matmul(sparse_attention, v)
        output = output.transpose(0, 1).reshape(time_steps * region * region, self.inner_dim)
        return self.output(output).reshape(time_steps, region, region, self.feature_dim)

    def _center_window_mask(self, region, device):
        window = min(self.center_window, region)
        start = (region - window) // 2
        end = start + window
        mask = torch.zeros(region, region, dtype=torch.bool, device=device)
        mask[start:end, start:end] = True
        return mask.reshape(region * region)

    def _merge_regions(self, patches, padded_shape, channels, device, dtype):
        region = self.region_size
        padded_h, padded_w = padded_shape
        time_steps = patches[0].size(0)
        output = torch.zeros(time_steps, padded_h, padded_w, channels, device=device, dtype=dtype)
        index = 0
        for row in range(0, padded_h, region):
            for col in range(0, padded_w, region):
                output[:, row : row + region, col : col + region, :] = patches[index]
                index += 1
        return output


class SatFormerBlock(nn.Module):
    def __init__(self, feature_dim, region_size, center_window, heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.assit = ASSIT(feature_dim, region_size, center_window, heads)
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
        self.gcn = TwoLayerODGCN(feature_dim, gcn_hidden_dim, feature_dim, dropout=dropout)
        self.satformer = SatFormerBlock(feature_dim, region_size, center_window, heads)

    def forward(self, x, adjacency_matrix):
        x = x + self.gcn(self.norm_gcn(x), adjacency_matrix)
        return self.satformer(x)

    def forward_sequence(self, x, adjacency_matrices):
        outputs = []
        for time_step in range(x.size(0)):
            x_t = x[time_step]
            adjacency_t = adjacency_matrices[:, :, time_step]
            outputs.append(x_t + self.gcn(self.norm_gcn(x_t), adjacency_t))
        return self.satformer(torch.stack(outputs, dim=0))


class TransferModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        self.p = nn.Parameter(torch.tensor(1.0))
        self.q = nn.Parameter(torch.tensor(1.0))
        self.scale = feature_dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        outputs = []
        for t in range(x.size(0)):
            scores = torch.einsum("ijc,kijc->ijk", q[t], k[: t + 1]) * self.scale
            attention = F.softmax(scores, dim=-1)
            if t > 0:
                past_context = torch.einsum("ijk,kijc->ijc", attention[:, :, :t], v[:t])
            else:
                past_context = torch.zeros_like(v[t])
            current_context = attention[:, :, -1].unsqueeze(-1) * v[t]
            d_t = torch.sigmoid(self.p * past_context + self.q * current_context)
            outputs.append(d_t)
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
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.input_projection = nn.Linear(2, feature_dim)
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
            nn.Linear(feature_dim, 1),
        )

    def _run_module(self, module, h, adjacency):
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, h, adjacency, use_reentrant=False)
        return module(h, adjacency)

    def _run_module_sequence(self, module, h, adjacency_matrices):
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                lambda hidden, adjacency: module.forward_sequence(hidden, adjacency),
                h,
                adjacency_matrices,
                use_reentrant=False,
            )
        return module.forward_sequence(h, adjacency_matrices)

    def _run_transfer(self, h):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self.transfer, h, use_reentrant=False)
        return self.transfer(h)

    def _encode_time_step(self, x_t, mask_t, adjacency):
        x_inp = torch.stack((x_t, mask_t), dim=-1)
        h = self.input_projection(x_inp)
        for module in self.encoder:
            h = self._run_module(module, h, adjacency)
        return h

    def _encode_sequence(self, x, observed_mask, adjacency_matrices):
        projected = []
        for time_step in range(x.size(-1)):
            x_inp = torch.stack(
                (x[:, :, time_step], observed_mask[:, :, time_step]),
                dim=-1,
            )
            projected.append(self.input_projection(x_inp))
        h = torch.stack(projected, dim=0)
        for module in self.encoder:
            h = self._run_module_sequence(module, h, adjacency_matrices)
        return h

    def _decode_time_step(self, h_t, adjacency):
        for module in self.decoder:
            h_t = self._run_module(module, h_t, adjacency)
        return self.output_projection(h_t).squeeze(-1)

    def _decode_sequence(self, h, adjacency_matrices):
        for module in self.decoder:
            h = self._run_module_sequence(module, h, adjacency_matrices)
        decoded = []
        for time_step in range(h.size(0)):
            decoded.append(self.output_projection(h[time_step]).squeeze(-1).unsqueeze(-1))
        return torch.cat(decoded, dim=-1)

    def forward(self, x, adjacency_matrices, observed_mask=None):
        if observed_mask is None:
            observed_mask = (x != 0).to(dtype=x.dtype)
        encoded = self._encode_sequence(x, observed_mask, adjacency_matrices)
        transferred = self._run_transfer(encoded)
        return self._decode_sequence(transferred, adjacency_matrices)

    def forward_time_window(
        self,
        x_window,
        adjacency_window,
        mask_window=None,
        target_offset=-1,
    ):
        if mask_window is None:
            mask_window = (x_window != 0).to(dtype=x_window.dtype)
        encoded = self._encode_sequence(x_window, mask_window, adjacency_window)
        transferred = self._run_transfer(encoded)
        decoded = self._decode_sequence(transferred, adjacency_window)
        return decoded[:, :, target_offset]
