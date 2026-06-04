import math

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class ASSIT(nn.Module):
    """Adaptive sparse spatio-temporal attention over local 2-D regions."""

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


# Backward-compatible names for older imports.
SatFormer_Block = SatFormerBlock


def process_local_regions(x, block, num_heads=None, region_size=None):
    return block(x)
