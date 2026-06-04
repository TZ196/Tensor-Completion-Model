import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn

from satformer_model import SatFormer


def load_tensor(path):
    tensor = np.load(path)
    if tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape %s" % (tensor.shape,))
    return tensor.astype("float32")


def load_npz_array(path, preferred_key="sat_connectivity"):
    data = np.load(path)
    if preferred_key in data.files:
        return data[preferred_key].astype("float32")
    if len(data.files) == 1:
        return data[data.files[0]].astype("float32")
    raise ValueError(
        "%s contains multiple arrays %s, but key %r was not found"
        % (path, data.files, preferred_key)
    )


def nonzero_finite_entries(tensor):
    finite_mask = np.isfinite(tensor)
    mask = finite_mask & (tensor != 0)
    indices = np.argwhere(mask).astype("int64")
    values = tensor[mask].astype("float32")
    stats = {
        "total_entries": int(tensor.size),
        "finite_entries": int(np.sum(finite_mask)),
        "nonzero_finite_entries": int(np.sum(mask)),
        "zero_finite_entries": int(np.sum(finite_mask & (tensor == 0))),
        "nonfinite_entries": int(tensor.size - np.sum(finite_mask)),
    }
    return indices, values, stats


def create_random_completion_split(tensor_path, split_path, observed_ratio, val_ratio, seed):
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
    val_order = order[train_size : train_size + val_size]
    test_order = order[train_size + val_size :]

    split_dir = os.path.dirname(split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)

    np.savez(
        split_path,
        shape=np.array(tensor.shape).astype("int32"),
        train_indices=indices[train_order],
        train_values=values[train_order],
        val_indices=indices[val_order],
        val_values=values[val_order],
        test_indices=indices[test_order],
        test_values=values[test_order],
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
    return (
        np.array(tensor.shape).astype("int32"),
        indices[train_order],
        values[train_order],
        indices[val_order],
        values[val_order],
        indices[test_order],
        values[test_order],
        data_stats,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SatFormer tensor completion on sat_path_bytes_tensor.npy."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument(
        "--adjacency-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--gcn-hidden-dim", type=int, default=128)
    parser.add_argument("--num-modules", type=int, default=10)
    parser.add_argument("--region-size", type=int, default=16)
    parser.add_argument("--center-window", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--target-normalization",
        choices=["max", "none"],
        default="max",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


def default_output_path(output_dir, observed_ratio, val_ratio, seed, feature_dim, num_modules, target_normalization):
    name = "random_observed%d_val%d_seed%d_dim%d_layers%d_norm_%s.json" % (
        int(round(observed_ratio * 100)),
        int(round(val_ratio * 100)),
        seed,
        feature_dim,
        num_modules,
        target_normalization,
    )
    return os.path.join(output_dir, name)


def get_target_scale(values, target_normalization):
    if target_normalization == "none":
        return 1.0
    scale = float(np.max(values))
    if scale <= 0.0:
        raise ValueError("Cannot use max normalization with non-positive max")
    return scale


def configure_torch(cpu_only=False, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not cpu_only:
        torch.cuda.manual_seed_all(seed)
        return torch.device("cuda")
    return torch.device("cpu")


def build_observed_tensor(shape, indices, values, target_scale, device):
    observed = torch.zeros(tuple(shape.tolist()), dtype=torch.float32, device=device)
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    vals = torch.as_tensor(values / target_scale, dtype=torch.float32, device=device)
    observed[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
    return observed


def gather_entries(tensor, indices):
    idx = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor[idx[:, 0], idx[:, 1], idx[:, 2]]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def nmae(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))
    if denominator == 0.0:
        raise ValueError("Cannot compute NMAE: sum(abs(y_true)) is 0")
    return np.sum(np.abs(y_true - y_pred)) / denominator


def nrmse(y_true, y_pred):
    denominator = np.sum(np.square(y_true))
    if denominator == 0.0:
        raise ValueError("Cannot compute NRMSE: sum(square(y_true)) is 0")
    return np.sqrt(np.sum(np.square(y_true - y_pred)) / denominator)


def evaluate_satformer(model, input_tensor, adjacency_tensor, indices, values, target_scale):
    model.eval()
    with torch.no_grad():
        pred = gather_entries(model(input_tensor, adjacency_tensor), indices)
        pred = pred.detach().cpu().numpy().astype("float32") * target_scale
    pred = np.maximum(pred, 0.0)
    return {
        "rmse": float(rmse(values, pred)),
        "mae": float(mae(values, pred)),
        "nmae": float(nmae(values, pred)),
        "nrmse": float(nrmse(values, pred)),
        "y_true_min": float(np.min(values)),
        "y_true_max": float(np.max(values)),
        "y_true_mean": float(np.mean(values)),
        "y_pred_min": float(np.min(pred)),
        "y_pred_max": float(np.max(pred)),
        "y_pred_mean": float(np.mean(pred)),
    }


def main():
    args = parse_args()
    observed_ratio = args.observed_ratio
    if observed_ratio is None:
        if not 0.0 < args.missing_rate < 1.0:
            raise ValueError("--missing-rate must be between 0 and 1")
        observed_ratio = 1.0 - args.missing_rate
    elif not 0.0 < observed_ratio < 1.0:
        raise ValueError("--observed-ratio must be between 0 and 1")

    if args.split_path is None:
        args.split_path = os.path.join(
            "splits",
            "random_observed%d_val%d_seed_%d.npz" % (
                int(round(observed_ratio * 100)),
                int(round(args.val_ratio * 100)),
                args.seed,
            ),
        )

    if args.metrics_path is None:
        args.metrics_path = default_output_path(
            "results",
            observed_ratio,
            args.val_ratio,
            args.seed,
            args.feature_dim,
            args.num_modules,
            args.target_normalization,
        )

    device = configure_torch(cpu_only=args.cpu_only, seed=args.seed)
    (
        shape,
        train_indices,
        train_values,
        val_indices,
        val_values,
        test_indices,
        test_values,
        data_stats,
    ) = create_random_completion_split(
        args.tensor_path,
        args.split_path,
        observed_ratio,
        args.val_ratio,
        args.seed,
    )

    adjacency = load_npz_array(args.adjacency_path)
    if tuple(adjacency.shape) != tuple(shape.tolist()):
        raise ValueError("Adjacency shape %s does not match tensor shape %s" % (adjacency.shape, shape))
    adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32, device=device)

    target_scale = get_target_scale(train_values, args.target_normalization)
    input_tensor = build_observed_tensor(shape, train_indices, train_values, target_scale, device)
    train_targets = torch.as_tensor(train_values / target_scale, dtype=torch.float32, device=device)
    val_targets = torch.as_tensor(val_values / target_scale, dtype=torch.float32, device=device)

    model = SatFormer(
        num_nodes=int(shape[0]),
        feature_dim=args.feature_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        num_modules=args.num_modules,
        region_size=args.region_size,
        center_window=args.center_window,
        heads=args.heads,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print("Tensor shape:", shape.tolist())
    print("Topology shape:", list(adjacency.shape))
    print("Split mode: random transductive completion")
    print("Non-zero finite entries:", data_stats["nonzero_finite_entries"])
    print("Observed ratio:", observed_ratio)
    print("Missing rate:", 1.0 - observed_ratio)
    print("Validation ratio within unobserved entries:", args.val_ratio)
    print("Train entries:", train_indices.shape[0])
    print("Validation entries:", val_indices.shape[0])
    print("Test entries:", test_indices.shape[0])
    print("Device:", device)
    print("Split path:", args.split_path)
    print("Metrics path:", args.metrics_path)
    print("Target normalization:", args.target_normalization)
    print("Target scale:", target_scale)
    print("Gradient checkpointing:", not args.no_gradient_checkpointing)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(input_tensor, adjacency_tensor)
        train_pred = gather_entries(output, train_indices)
        train_loss = criterion(train_pred, train_targets)
        train_loss.backward()
        optimizer.step()
        train_loss_value = train_loss.item()
        del output, train_pred, train_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            output = model(input_tensor, adjacency_tensor)
            val_pred = gather_entries(output, val_indices)
            val_loss = criterion(val_pred, val_targets)
            val_loss_value = val_loss.item()
        del output, val_pred, val_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            "Epoch [%d/%d] loss: %.6f val_loss: %.6f"
            % (epoch + 1, args.epochs, train_loss_value, val_loss_value)
        )

        if val_loss_value < best_val:
            best_val = val_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping at epoch %d" % (epoch + 1))
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    results = {
        "config": {
            "tensor_path": args.tensor_path,
            "adjacency_path": args.adjacency_path,
            "split_mode": "random_transductive_completion",
            "sample_filter": "finite_and_nonzero",
            "observed_ratio": observed_ratio,
            "missing_rate": 1.0 - observed_ratio,
            "val_ratio_within_unobserved": args.val_ratio,
            "train_entries": int(train_indices.shape[0]),
            "val_entries": int(val_indices.shape[0]),
            "test_entries": int(test_indices.shape[0]),
            "feature_dim": args.feature_dim,
            "gcn_hidden_dim": args.gcn_hidden_dim,
            "num_modules": args.num_modules,
            "region_size": args.region_size,
            "center_window": args.center_window,
            "heads": args.heads,
            "dropout": args.dropout,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "target_normalization": args.target_normalization,
            "target_scale": target_scale,
            "metrics_scale": "original",
            "seed": args.seed,
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
        },
        "train": evaluate_satformer(model, input_tensor, adjacency_tensor, train_indices, train_values, target_scale),
        "val": evaluate_satformer(model, input_tensor, adjacency_tensor, val_indices, val_values, target_scale),
        "test": evaluate_satformer(model, input_tensor, adjacency_tensor, test_indices, test_values, target_scale),
    }

    pprint(results)
    metrics_dir = os.path.dirname(args.metrics_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    with open(args.metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()
