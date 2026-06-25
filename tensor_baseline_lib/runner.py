import argparse
import json
import math
import os
import random
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_tensor(path):
    tensor = np.load(path)
    if tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape %s" % (tensor.shape,))
    return tensor.astype("float32")


def nonzero_finite_entries(tensor):
    finite_mask = np.isfinite(tensor)
    mask = finite_mask & (tensor != 0)
    indices = np.argwhere(mask).astype("int64")
    values = tensor[mask].astype("float32")
    return indices, values, {
        "total_entries": int(tensor.size),
        "finite_entries": int(np.sum(finite_mask)),
        "nonzero_finite_entries": int(np.sum(mask)),
        "zero_finite_entries": int(np.sum(finite_mask & (tensor == 0))),
        "nonfinite_entries": int(tensor.size - np.sum(finite_mask)),
    }


def create_random_completion_split(tensor_path, split_path, observed_ratio, val_ratio, seed):
    if split_path and os.path.exists(split_path):
        data = np.load(split_path)
        return (
            np.asarray(data["shape"]).astype("int32"),
            data["train_indices"].astype("int64"),
            data["train_values"].astype("float32"),
            data["val_indices"].astype("int64"),
            data["val_values"].astype("float32"),
            data["test_indices"].astype("int64"),
            data["test_values"].astype("float32"),
            {
                "observed_ratio": float(data["observed_ratio"]),
                "missing_rate": float(data["missing_rate"]),
                "val_ratio": float(data["val_ratio"]),
                "seed": int(data["seed"]),
                "total_entries": int(data["total_entries"]),
                "finite_entries": int(data["finite_entries"]),
                "nonzero_finite_entries": int(data["nonzero_finite_entries"]),
                "zero_finite_entries": int(data["zero_finite_entries"]),
                "nonfinite_entries": int(data["nonfinite_entries"]),
                "loaded_existing_split": True,
            },
        )

    tensor = load_tensor(tensor_path)
    if not 0.0 < observed_ratio < 1.0:
        raise ValueError("--observed-ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    indices, values, data_stats = nonzero_finite_entries(tensor)
    if indices.shape[0] < 3:
        raise ValueError("Need at least 3 non-zero finite entries to split")

    rng = np.random.RandomState(seed)
    order = rng.permutation(indices.shape[0])
    train_size = int(round(indices.shape[0] * observed_ratio))
    train_size = max(1, min(train_size, indices.shape[0] - 2))
    remaining_size = indices.shape[0] - train_size
    val_size = int(round(remaining_size * val_ratio))
    val_size = max(1, min(val_size, remaining_size - 1))

    train_order = order[:train_size]
    val_order = order[train_size:train_size + val_size]
    test_order = order[train_size + val_size:]
    split = (
        np.asarray(tensor.shape).astype("int32"),
        indices[train_order],
        values[train_order],
        indices[val_order],
        values[val_order],
        indices[test_order],
        values[test_order],
    )

    if split_path:
        split_dir = os.path.dirname(split_path)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
        np.savez(
            split_path,
            shape=split[0],
            train_indices=split[1],
            train_values=split[2],
            val_indices=split[3],
            val_values=split[4],
            test_indices=split[5],
            test_values=split[6],
            observed_ratio=np.array(observed_ratio).astype("float32"),
            missing_rate=np.array(1.0 - observed_ratio).astype("float32"),
            val_ratio=np.array(val_ratio).astype("float32"),
            seed=np.array(seed).astype("int32"),
            total_entries=np.array(data_stats["total_entries"]).astype("int64"),
            finite_entries=np.array(data_stats["finite_entries"]).astype("int64"),
            nonzero_finite_entries=np.array(data_stats["nonzero_finite_entries"]).astype("int64"),
            zero_finite_entries=np.array(data_stats["zero_finite_entries"]).astype("int64"),
            nonfinite_entries=np.array(data_stats["nonfinite_entries"]).astype("int64"),
        )

    stats = dict(data_stats)
    stats.update({
        "observed_ratio": float(observed_ratio),
        "missing_rate": float(1.0 - observed_ratio),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "loaded_existing_split": False,
    })
    return (*split, stats)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_target_scale(values, target_normalization):
    if target_normalization == "none":
        return 1.0
    if target_normalization == "max":
        scale = float(np.max(values))
        if scale <= 0.0:
            raise ValueError("Cannot use max normalization with non-positive max")
        return scale
    raise ValueError("Unsupported target normalization: %s" % target_normalization)


def make_mlp(input_dim, hidden_dim, output_dim=1, depth=2, dropout=0.1):
    layers = []
    dim = input_dim
    for _ in range(max(1, depth)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class NTCModel(nn.Module):
    """Neural Tensor Completion style embedding interaction model."""

    def __init__(self, shape, rank=64, hidden_dim=128, depth=2, dropout=0.1):
        super().__init__()
        self.src = nn.Embedding(shape[0], rank)
        self.dst = nn.Embedding(shape[1], rank)
        self.time = nn.Embedding(shape[2], rank)
        feature_dim = rank * 9
        self.head = make_mlp(feature_dim, hidden_dim, 1, depth, dropout)

    def forward(self, indices):
        s = self.src(indices[:, 0])
        d = self.dst(indices[:, 1])
        t = self.time(indices[:, 2])
        x = torch.cat([s, d, t, s * d, s * t, d * t, torch.abs(s - d), torch.abs(s - t), torch.abs(d - t)], dim=-1)
        return self.head(x).squeeze(-1)


class NTFModel(nn.Module):
    """Neural Tensor Factorization with CP-style product plus nonlinear residual."""

    def __init__(self, shape, rank=64, hidden_dim=128, depth=2, dropout=0.1):
        super().__init__()
        self.src = nn.Embedding(shape[0], rank)
        self.dst = nn.Embedding(shape[1], rank)
        self.time = nn.Embedding(shape[2], rank)
        self.bias_s = nn.Embedding(shape[0], 1)
        self.bias_d = nn.Embedding(shape[1], 1)
        self.bias_t = nn.Embedding(shape[2], 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.residual = make_mlp(rank * 4, hidden_dim, 1, depth, dropout)

    def forward(self, indices):
        s = self.src(indices[:, 0])
        d = self.dst(indices[:, 1])
        t = self.time(indices[:, 2])
        cp = torch.sum(s * d * t, dim=-1)
        bias = self.bias_s(indices[:, 0]).squeeze(-1) + self.bias_d(indices[:, 1]).squeeze(-1) + self.bias_t(indices[:, 2]).squeeze(-1)
        residual = self.residual(torch.cat([s, d, t, s * d * t], dim=-1)).squeeze(-1)
        return cp + bias + self.global_bias + residual


class NTMModel(nn.Module):
    """Nonlinear Tensor Machine style bilinear tensor interaction."""

    def __init__(self, shape, rank=64, tensor_rank=32, hidden_dim=128, depth=2, dropout=0.1):
        super().__init__()
        self.src = nn.Embedding(shape[0], rank)
        self.dst = nn.Embedding(shape[1], rank)
        self.time = nn.Embedding(shape[2], rank)
        self.tensor = nn.Parameter(torch.empty(tensor_rank, rank, rank))
        nn.init.xavier_uniform_(self.tensor)
        self.head = make_mlp(tensor_rank + rank * 3, hidden_dim, 1, depth, dropout)

    def forward(self, indices):
        s = self.src(indices[:, 0])
        d = self.dst(indices[:, 1])
        t = self.time(indices[:, 2])
        bilinear = torch.einsum("br,krs,bs->bk", s, self.tensor, d)
        x = torch.cat([bilinear, s, d, t], dim=-1)
        return self.head(x).squeeze(-1)


class SAITSLikeModel(nn.Module):
    """Self-attention imputation baseline adapted to sparse tensor entries."""

    def __init__(self, shape, rank=64, hidden_dim=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.src = nn.Embedding(shape[0], rank)
        self.dst = nn.Embedding(shape[1], rank)
        self.time = nn.Embedding(shape[2], rank)
        self.role = nn.Embedding(3, rank)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rank,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = make_mlp(rank * 4, hidden_dim, 1, 1, dropout)

    def forward(self, indices):
        tokens = torch.stack([
            self.src(indices[:, 0]) + self.role.weight[0],
            self.dst(indices[:, 1]) + self.role.weight[1],
            self.time(indices[:, 2]) + self.role.weight[2],
        ], dim=1)
        z = self.encoder(tokens)
        pooled = torch.cat([z.reshape(z.shape[0], -1), z.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)


class CSDILikeModel(nn.Module):
    """Lightweight conditional denoising baseline inspired by CSDI."""

    def __init__(self, shape, rank=64, hidden_dim=128, depth=3, diffusion_steps=16, dropout=0.1):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.src = nn.Embedding(shape[0], rank)
        self.dst = nn.Embedding(shape[1], rank)
        self.time = nn.Embedding(shape[2], rank)
        self.step = nn.Embedding(self.diffusion_steps, rank)
        self.denoiser = make_mlp(rank * 4 + 1, hidden_dim, 1, depth, dropout)

    def forward(self, indices, noisy_value=None, step=None):
        if step is None:
            step = torch.zeros(indices.shape[0], dtype=torch.long, device=indices.device)
        if noisy_value is None:
            noisy_value = torch.zeros(indices.shape[0], dtype=torch.float32, device=indices.device)
        s = self.src(indices[:, 0])
        d = self.dst(indices[:, 1])
        t = self.time(indices[:, 2])
        q = self.step(step)
        x = torch.cat([s, d, t, q, noisy_value[:, None]], dim=-1)
        return self.denoiser(x).squeeze(-1)


class PriSTILikeModel(nn.Module):
    """Spatiotemporal imputation baseline with source/destination structural priors."""

    def __init__(self, shape, rank=64, hidden_dim=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.src = nn.Embedding(shape[0], rank)
        self.dst = nn.Embedding(shape[1], rank)
        self.time = nn.Embedding(shape[2], rank)
        self.src_context = nn.Embedding(shape[0], rank)
        self.dst_context = nn.Embedding(shape[1], rank)
        self.role = nn.Embedding(5, rank)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rank,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = make_mlp(rank * 6, hidden_dim, 1, 1, dropout)

    def forward(self, indices):
        tokens = torch.stack([
            self.src(indices[:, 0]) + self.role.weight[0],
            self.dst(indices[:, 1]) + self.role.weight[1],
            self.time(indices[:, 2]) + self.role.weight[2],
            self.src_context(indices[:, 0]) + self.role.weight[3],
            self.dst_context(indices[:, 1]) + self.role.weight[4],
        ], dim=1)
        z = self.encoder(tokens)
        pooled = torch.cat([z.reshape(z.shape[0], -1), z.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)


def build_model(model_name, shape, args):
    model_name = model_name.lower()
    if model_name == "ntc":
        return NTCModel(shape, args.rank, args.hidden_dim, args.mlp_depth, args.dropout)
    if model_name == "ntf":
        return NTFModel(shape, args.rank, args.hidden_dim, args.mlp_depth, args.dropout)
    if model_name == "ntm":
        return NTMModel(shape, args.rank, args.tensor_rank, args.hidden_dim, args.mlp_depth, args.dropout)
    if model_name == "saits":
        return SAITSLikeModel(shape, args.rank, args.hidden_dim, args.layers, args.heads, args.dropout)
    if model_name == "csdi":
        return CSDILikeModel(shape, args.rank, args.hidden_dim, args.mlp_depth, args.diffusion_steps, args.dropout)
    if model_name == "pristi":
        return PriSTILikeModel(shape, args.rank, args.hidden_dim, args.layers, args.heads, args.dropout)
    raise ValueError("Unsupported model name: %s" % model_name)


def model_forward(model, model_name, indices, targets=None):
    if model_name.lower() != "csdi" or targets is None:
        return model(indices)
    batch = indices.shape[0]
    step = torch.randint(0, model.diffusion_steps, (batch,), device=indices.device)
    noise_scale = (step.float() + 1.0) / float(model.diffusion_steps)
    noise = torch.randn_like(targets) * noise_scale
    noisy = targets + noise
    return model(indices, noisy, step)


@torch.no_grad()
def predict(model, model_name, indices, batch_size, device):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(indices).long()), batch_size=batch_size, shuffle=False)
    for (idx,) in loader:
        idx = idx.to(device)
        pred = model(idx)
        outputs.append(pred.detach().cpu())
    return torch.cat(outputs, dim=0).numpy().astype("float32")


def metrics(y_true, y_pred):
    y_true = y_true.astype("float32")
    y_pred = y_pred.astype("float32")
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    nmae = float(np.sum(np.abs(error)) / max(np.sum(np.abs(y_true)), 1e-8))
    nrmse = float(np.sqrt(np.sum(error ** 2) / max(np.sum(y_true ** 2), 1e-8)))
    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "nrmse": nrmse,
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
        "y_pred_mean": float(np.mean(y_pred)),
    }


def parse_args(default_model):
    parser = argparse.ArgumentParser("Sparse tensor completion baseline")
    parser.add_argument("--model-name", default=default_model)
    parser.add_argument("--tensor-path", default="data/sat_path_bytes_mb_tensor.npy")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--target-normalization", choices=["max", "none"], default="max")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--tensor-rank", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    return parser.parse_args()


def run(default_model):
    args = parse_args(default_model)
    set_seed(args.seed)
    observed_ratio = args.observed_ratio
    if observed_ratio is None:
        observed_ratio = 1.0 - args.missing_rate
    if args.split_path is None:
        observed_percent = int(round(observed_ratio * 100))
        args.split_path = os.path.join("splits", "random_observed%d_val10_seed_%d.npz" % (observed_percent, args.seed))
    if args.metrics_path is None:
        observed_percent = int(round(observed_ratio * 100))
        args.metrics_path = os.path.join("results", "vis%d_%s_seed%d.json" % (observed_percent, args.model_name.lower(), args.seed))

    shape, train_idx, train_values, val_idx, val_values, test_idx, test_values, data_stats = create_random_completion_split(
        args.tensor_path,
        args.split_path,
        observed_ratio,
        args.val_ratio,
        args.seed,
    )
    target_scale = get_target_scale(train_values, args.target_normalization)
    train_targets = (train_values / target_scale).astype("float32")
    val_targets = (val_values / target_scale).astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    model = build_model(args.model_name, [int(v) for v in shape], args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_idx).long(), torch.from_numpy(train_targets).float()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_val = math.inf
    wait = 0
    history = []
    print("===== Tensor Baseline Experiment =====")
    print("Model:", args.model_name)
    print("Tensor shape:", shape.tolist())
    print("Observed ratio:", observed_ratio)
    print("Train/val/test entries:", len(train_idx), len(val_idx), len(test_idx))
    print("Device:", device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for idx, target in train_loader:
            idx = idx.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model_forward(model, args.model_name, idx, target)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_pred = predict(model, args.model_name, val_idx, args.eval_batch_size, device) * target_scale
        val_metric = metrics(val_values, val_pred)
        val_loss = float(np.mean(((val_values - val_pred) / target_scale) ** 2))
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_nmae": val_metric["nmae"], "val_nrmse": val_metric["nrmse"]})
        if epoch == 1 or epoch % 10 == 0:
            print("epoch %d/%d - train_loss=%.6f - val_loss=%.6f - val_nmae=%.6f - val_nrmse=%.6f" % (epoch, args.epochs, train_loss, val_loss, val_metric["nmae"], val_metric["nrmse"]))

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping at epoch:", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    train_pred = predict(model, args.model_name, train_idx, args.eval_batch_size, device) * target_scale
    val_pred = predict(model, args.model_name, val_idx, args.eval_batch_size, device) * target_scale
    test_pred = predict(model, args.model_name, test_idx, args.eval_batch_size, device) * target_scale
    payload = {
        "config": vars(args),
        "shape": [int(v) for v in shape],
        "data_stats": data_stats,
        "target_scale": float(target_scale),
        "history": history,
        "train": metrics(train_values, train_pred),
        "validation": metrics(val_values, val_pred),
        "test": metrics(test_values, test_pred),
    }
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("Test metrics:")
    pprint(payload["test"])
    print("Saved metrics to:", args.metrics_path)
