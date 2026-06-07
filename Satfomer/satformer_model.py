import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GraphConvolution(nn.Module):
    """Standard graph convolution as described in the SatFormer paper.

    H^{l+1} = sigma(D^{-1/2} A_tilde D^{-1/2} H^l W^l)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @staticmethod
    def normalize_adjacency(adjacency_matrix):
        if getattr(adjacency_matrix, "_satformer_normalized", False):
            return adjacency_matrix
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
        # x: [N, N, C_in]  — OD hidden representation
        # adjacency: [N, N]  — satellite connectivity
        a_norm = self.normalize_adjacency(adjacency_matrix)
        # Standard GCN: propagate along source-satellite dimension only
        support = torch.matmul(a_norm, x)  # [N, N, C_in]
        return self.linear(support)        # [N, N, C_out]


class TwoLayerGCN(nn.Module):
    """Two-layer GCN as described in the SatFormer paper.

    Z = sigma( D^{-1/2} A_tilde D^{-1/2}
               ReLU( D^{-1/2} A_tilde D^{-1/2} X W^(0) )
               W^(1) )
    """

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
    """Adaptive Sparse Spatial Attention on local OD regions.

    Paper description:
      - Split the 2-D spatial tensor into D×D local regions.
      - Compute multi-head self-attention within each region.
      - Apply a centre-window mask Ψ so attention is restricted
        to a local neighbourhood.
      - Adaptive sparsity: ReLU(1 - Wr * alpha) followed by
        softmax(Ws * sparse_gate * alpha).
    """

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
        self._center_mask_cache = {}

    def forward(self, x):
        """x: [H, W, C] — a single 2-D spatial slice (one time step)."""
        if x.dim() != 3:
            raise ValueError(
                "ASSIT expects [H,W,C] (single time step), got %s" % (tuple(x.shape),)
            )

        height, width, channels = x.shape
        patches, valid_masks, padded_shape = self._split_regions(x)
        outputs = [
            self._attend_region(patch, valid_mask)
            for patch, valid_mask in zip(patches, valid_masks)
        ]
        y = self._merge_regions(outputs, padded_shape, channels, x.device, x.dtype)
        return y[:height, :width, :]

    def _split_regions(self, x):
        region = self.region_size
        height, width, channels = x.shape
        pad_h = (region - height % region) % region
        pad_w = (region - width % region) % region
        padded_h = height + pad_h
        padded_w = width + pad_w
        padded = x.new_zeros(padded_h, padded_w, channels)
        padded[:height, :width, :] = x
        valid = torch.ones(height, width, dtype=torch.bool, device=x.device)
        valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)

        patches = []
        valid_masks = []
        for row in range(0, padded_h, region):
            for col in range(0, padded_w, region):
                patches.append(padded[row: row + region, col: col + region, :])
                valid_masks.append(valid[row: row + region, col: col + region])
        return patches, valid_masks, (padded_h, padded_w)

    def _attend_region(self, patch, valid_mask):
        """Attend over spatial positions within a single D×D region.

        patch:   [region, region, C]
        valid_mask: [region, region]  bool
        """
        region = self.region_size
        # Spatial tokens only — no time dimension
        tokens = patch.reshape(region * region, self.feature_dim)

        q = self.query(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)
        k = self.key(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)
        v = self.value(tokens).view(-1, self.heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Centre-window mask Ψ
        center_mask = self._center_window_mask(region, patch.device)
        valid_key_mask = valid_mask.reshape(region * region)
        spatial_key_mask = center_mask & valid_key_mask
        if not torch.any(spatial_key_mask):
            spatial_key_mask = valid_key_mask
        scores = scores.masked_fill(~spatial_key_mask[None, None, :], -65504.0)

        attention = F.softmax(scores, dim=-1)

        # Adaptive sparse gating
        sparse_gate = F.relu(1.0 - self.sparse_regularizer * attention)
        sparse_attention = F.softmax(
            self.sparse_scale * sparse_gate * attention, dim=-1
        )

        output = torch.matmul(sparse_attention, v)
        output = output.transpose(0, 1).reshape(
            region * region, self.inner_dim
        )
        return self.output(output).reshape(region, region, self.feature_dim)

    def _center_window_mask(self, region, device):
        cache_key = (region, str(device))
        if cache_key in self._center_mask_cache:
            return self._center_mask_cache[cache_key]
        window = min(self.center_window, region)
        start = (region - window) // 2
        end = start + window
        mask = torch.zeros(region, region, dtype=torch.bool, device=device)
        mask[start:end, start:end] = True
        mask = mask.reshape(region * region)
        self._center_mask_cache[cache_key] = mask
        return mask

    def _merge_regions(self, patches, padded_shape, channels, device, dtype):
        region = self.region_size
        padded_h, padded_w = padded_shape
        output = torch.zeros(padded_h, padded_w, channels, device=device, dtype=dtype)
        index = 0
        for row in range(0, padded_h, region):
            for col in range(0, padded_w, region):
                output[row: row + region, col: col + region, :] = patches[index]
                index += 1
        return output


class SatFormerBlock(nn.Module):
    """SatFormer block as shown in paper Fig. 1(b).

    LN -> ASSIT -> residual -> LN -> MLP -> residual
    """

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
    """One Spatio-Temporal Module from the paper.

    Contains:
      1. Graph Embedding Module (two-layer GCN)
      2. SatFormer Block (ASSIT + MLP)

    Residual connections are added between modules
    (see SatFormer._encode_sequence / _decode_sequence).
    """

    def __init__(
        self, feature_dim, gcn_hidden_dim, region_size, center_window, heads, dropout=0.0
    ):
        super().__init__()
        self.gcn = TwoLayerGCN(feature_dim, gcn_hidden_dim, feature_dim, dropout=dropout)
        self.satformer = SatFormerBlock(feature_dim, region_size, center_window, heads)

    def forward(self, x, adjacency_matrix):
        """x: [N, N, C] — single time step."""
        # Graph Embedding: standard GCN (no LayerNorm — paper does not mention LN before GCN)
        x = x + self.gcn(x, adjacency_matrix)
        # SatFormer Block
        return self.satformer(x)


class TransferModule(nn.Module):
    """Transfer Module between encoder and decoder.

    Paper description:
      Encoder outputs per-time-step feature vectors E = {e1, ..., eT}.
      Transfer Module uses temporal self-attention across time steps,
      with learnable scalars p (past) and q (current) to gate the
      influence of historical vs. current information.
    """

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
        """x: [T, N, N, C] — encoder output for all time steps.

        Returns: [T, N, N, C] — temporally fused features.
        """
        T, N, _, C = x.shape

        # Pool each time step to a single feature vector e_t  (paper Eq.)
        e = x.mean(dim=(1, 2))  # [T, C]

        q = self.query(e)  # [T, C]
        k = self.key(e)    # [T, C]
        v = self.value(e)  # [T, C]

        outputs = []
        for t in range(T):
            # Dot-product attention between q_t and all k_{0..t}
            scores = torch.einsum("c,ic->i", q[t], k[: t + 1]) * self.scale  # [t+1]
            attention = F.softmax(scores, dim=-1)  # [t+1]

            if t > 0:
                past_context = torch.einsum(
                    "i,ic->c", attention[:t], v[:t]
                )  # [C]
            else:
                past_context = torch.zeros(C, device=x.device, dtype=x.dtype)

            current_context = attention[t].unsqueeze(-1) * v[t]  # [C]

            # p gates past influence, q gates current influence
            d_t = torch.sigmoid(self.p * past_context + self.q * current_context)
            outputs.append(d_t)

        d = torch.stack(outputs, dim=0)  # [T, C]
        d = self.output(d)               # [T, C]

        # Broadcast temporal context back to every OD position
        return x + d[:, None, None, :]  # [T, N, N, C]


class SatFormer(nn.Module):
    """SatFormer: GCN + ASSIT for satellite traffic tensor completion.

    Paper architecture:
        Input X ∈ R^{N×N×T}
          → Encoder (L SpatioTemporalModules, per time step)
          → Transfer Module (temporal attention)
          → Decoder (L SpatioTemporalModules, per time step)
          → Output X_hat ∈ R^{N×N×T}
    """

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
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Embed scalar traffic value into feature_dim  (paper: first GCN layer W^(0))
        self.input_projection = nn.Linear(1, feature_dim)

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

    # ------------------------------------------------------------------
    #  Per-time-step helpers
    # ------------------------------------------------------------------

    def _run_module(self, module, h, adjacency):
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, h, adjacency, use_reentrant=False)
        return module(h, adjacency)

    def _run_transfer(self, h):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self.transfer, h, use_reentrant=False)
        return self.transfer(h)

    # ------------------------------------------------------------------
    #  Encode / decode a full sequence
    # ------------------------------------------------------------------

    def _encode_sequence(self, x, adjacency_matrices):
        """Encode each time step independently through all encoder modules.

        x:                 [N, N, T]
        adjacency_matrices: [N, N, T]
        returns:            [T, N, N, C]
        """
        T = x.size(-1)
        encoded = []
        for t in range(T):
            # x_t: [N, N, 1] -> embed -> [N, N, C]
            h = self.input_projection(x[:, :, t].unsqueeze(-1))
            adj_t = adjacency_matrices[:, :, t]
            for module in self.encoder:
                h = h + self._run_module(module, h, adj_t)
            encoded.append(h)
        return torch.stack(encoded, dim=0)

    def _decode_sequence(self, h, adjacency_matrices):
        """Decode each time step independently through all decoder modules.

        h:                 [T, N, N, C]
        adjacency_matrices: [N, N, T]
        returns:            [N, N, T]
        """
        T = h.size(0)
        decoded = []
        for t in range(T):
            d = h[t]
            adj_t = adjacency_matrices[:, :, t]
            for module in self.decoder:
                d = d + self._run_module(module, d, adj_t)
            y = self.output_projection(d).squeeze(-1)  # [N, N]
            decoded.append(y.unsqueeze(-1))
        return torch.cat(decoded, dim=-1)  # [N, N, T]

    # ------------------------------------------------------------------
    #  Public forward
    # ------------------------------------------------------------------

    def forward(self, x, adjacency_matrices):
        """Full-tensor forward pass.

        x:                 [N, N, T]  — partially observed traffic tensor
        adjacency_matrices: [N, N, T] — dynamic topology
        returns:            [N, N, T]  — completed traffic tensor
        """
        encoded = self._encode_sequence(x, adjacency_matrices)
        transferred = self._run_transfer(encoded)
        return self._decode_sequence(transferred, adjacency_matrices)

    def forward_time_window(
        self,
        x_window,
        adjacency_window,
        mask_window=None,
        target_offset=-1,
    ):
        """Sliding-window forward (used during training / step-by-step eval).

        x_window:         [N, N, window_size]
        adjacency_window: [N, N, window_size]
        mask_window:       ignored (paper does not use explicit mask channel)
        target_offset:     index of the target time step within the window
        returns:           [N, N]  — predicted traffic matrix at target time
        """
        encoded = self._encode_sequence(x_window, adjacency_window)
        transferred = self._run_transfer(encoded)
        decoded = self._decode_sequence(transferred, adjacency_window)
        return decoded[:, :, target_offset]
