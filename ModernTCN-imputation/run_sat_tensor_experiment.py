import argparse
import json
import os
import random
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from models.ModernTCN import Model


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
    stats = {
        "total_entries": int(tensor.size),
        "finite_entries": int(np.sum(finite_mask)),
        "nonzero_finite_entries": int(np.sum(mask)),
        "zero_finite_entries": int(np.sum(finite_mask & (tensor == 0))),
        "nonfinite_entries": int(tensor.size - np.sum(finite_mask)),
    }
    return indices, values, stats


def create_random_completion_split(tensor_path, split_path, observed_ratio,
                                   val_ratio, seed):
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

    split_dir = os.path.dirname(split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)

    split = {
        "shape": np.array(tensor.shape).astype("int32"),
        "train_indices": indices[train_order],
        "train_values": values[train_order],
        "val_indices": indices[val_order],
        "val_values": values[val_order],
        "test_indices": indices[test_order],
        "test_values": values[test_order],
    }
    np.savez(
        split_path,
        **split,
        observed_ratio=np.array(observed_ratio).astype("float32"),
        missing_rate=np.array(1.0 - observed_ratio).astype("float32"),
        val_ratio=np.array(val_ratio).astype("float32"),
        seed=np.array(seed).astype("int32"),
        total_entries=np.array(data_stats["total_entries"]).astype("int64"),
        finite_entries=np.array(data_stats["finite_entries"]).astype("int64"),
        nonzero_finite_entries=np.array(
            data_stats["nonzero_finite_entries"]
        ).astype("int64"),
        zero_finite_entries=np.array(
            data_stats["zero_finite_entries"]
        ).astype("int64"),
        nonfinite_entries=np.array(data_stats["nonfinite_entries"]).astype("int64"),
    )
    return tensor, split, data_stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target_scale(values, target_normalization):
    if target_normalization == "none":
        return 1.0
    if target_normalization == "max":
        scale = float(np.max(values))
        if scale <= 0.0:
            raise ValueError("Cannot use max normalization with non-positive max")
        return scale
    raise ValueError("Unsupported target normalization: %s" % target_normalization)


def flatten_time_major(tensor):
    return np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1)


def flat_positions(indices, dst_count):
    src = indices[:, 0]
    dst = indices[:, 1]
    time = indices[:, 2]
    variable = src * dst_count + dst
    return time.astype("int64"), variable.astype("int64")


def parse_int_list(value):
    if isinstance(value, list):
        return value
    return [int(item) for item in value.replace(",", " ").split()]


def expand_stage_dims(values, min_len=4):
    if len(values) == 1:
        return values * min_len
    if len(values) < min_len:
        return values + [values[-1]] * (min_len - len(values))
    return values


def build_moderntcn_configs(args, input_dim, seq_len):
    num_blocks = parse_int_list(args.num_blocks)
    large_size = parse_int_list(args.large_size)
    small_size = parse_int_list(args.small_size)
    stage_lengths = {len(num_blocks), len(large_size), len(small_size)}
    if len(stage_lengths) != 1:
        raise ValueError(
            "--num-blocks, --large-size, and --small-size must have the same "
            "number of stages"
        )
    dims = expand_stage_dims(parse_int_list(args.dims))
    dw_dims = expand_stage_dims(parse_int_list(args.dw_dims))
    return SimpleNamespace(
        task_name="imputation",
        seq_len=seq_len,
        pred_len=0,
        enc_in=input_dim,
        stem_ratio=args.stem_ratio,
        downsample_ratio=args.downsample_ratio,
        ffn_ratio=args.ffn_ratio,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_blocks=num_blocks,
        large_size=large_size,
        small_size=small_size,
        dims=dims,
        dw_dims=dw_dims,
        small_kernel_merged=False,
        use_multi_scale=args.use_multi_scale,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        revin=bool(args.revin),
        affine=bool(args.affine),
        subtract_last=bool(args.subtract_last),
        freq="h",
        individual=bool(args.individual),
        kernel_size=args.kernel_size,
        decomposition=0,
    )


def make_window_starts(seq_len, window_len, window_stride):
    if window_len <= 0:
        raise ValueError("--window-len must be positive")
    if window_stride <= 0:
        raise ValueError("--window-stride must be positive")
    if window_len > seq_len:
        window_len = seq_len
    last_start = seq_len - window_len
    starts = list(range(0, last_start + 1, window_stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return np.asarray(starts, dtype="int64"), int(window_len)


def filter_window_starts_by_mask(starts, mask, window_len):
    selected = [
        int(start)
        for start in starts
        if float(mask[start:start + window_len].sum()) > 0.0
    ]
    if not selected:
        raise ValueError("No windows contain entries for the requested mask")
    return np.asarray(selected, dtype="int64")


def make_window_batch(values, input_mask, loss_mask, starts, window_len, device):
    value_windows = np.stack(
        [values[start:start + window_len] for start in starts],
        axis=0,
    ).astype("float32")
    input_windows = np.stack(
        [input_mask[start:start + window_len] for start in starts],
        axis=0,
    ).astype("float32")
    loss_windows = np.stack(
        [loss_mask[start:start + window_len] for start in starts],
        axis=0,
    ).astype("float32")

    # ModernTCN imputation normalization expects each variable to have at
    # least one observed point in a window. Zero anchors avoid NaNs without
    # contributing to the supervised loss.
    model_mask = input_windows.copy()
    empty_variables = model_mask.sum(axis=1) == 0.0
    if np.any(empty_variables):
        batch_indices, variable_indices = np.where(empty_variables)
        model_mask[batch_indices, 0, variable_indices] = 1.0

    x_enc = value_windows * input_windows
    return (
        torch.from_numpy(x_enc).float().to(device),
        torch.from_numpy(model_mask).float().to(device),
        torch.from_numpy(loss_windows).float().to(device),
        torch.from_numpy(value_windows).float().to(device),
    )


def iter_start_batches(starts, batch_size):
    for offset in range(0, starts.shape[0], batch_size):
        yield starts[offset:offset + batch_size]


def train_one_epoch(model, optimizer, values, train_mask, train_starts,
                    window_len, batch_size, device, rng):
    model.train()
    shuffled = train_starts.copy()
    rng.shuffle(shuffled)
    total_loss = 0.0
    total_count = 0.0

    for batch_starts in iter_start_batches(shuffled, batch_size):
        x_enc, model_mask, loss_mask, target = make_window_batch(
            values,
            train_mask,
            train_mask,
            batch_starts,
            window_len,
            device,
        )
        count = float(loss_mask.sum().detach().cpu())
        if count <= 0.0:
            continue

        optimizer.zero_grad()
        output = model(x_enc, None, None, None, model_mask)
        loss_sum = F.l1_loss(output * loss_mask, target * loss_mask,
                             reduction="sum")
        loss = loss_sum / torch.clamp(loss_mask.sum(), min=1.0)
        loss.backward()
        optimizer.step()

        total_loss += float(loss_sum.detach().cpu())
        total_count += count

    if total_count <= 0.0:
        raise ValueError("No training entries were used in this epoch")
    return total_loss / total_count


@torch.no_grad()
def evaluate_window_mae(model, values, train_mask, eval_mask, eval_starts,
                        window_len, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_count = 0.0
    for batch_starts in iter_start_batches(eval_starts, batch_size):
        x_enc, model_mask, loss_mask, target = make_window_batch(
            values,
            train_mask,
            eval_mask,
            batch_starts,
            window_len,
            device,
        )
        count = float(loss_mask.sum().detach().cpu())
        if count <= 0.0:
            continue
        output = model(x_enc, None, None, None, model_mask)
        loss_sum = F.l1_loss(output * loss_mask, target * loss_mask,
                             reduction="sum")
        total_loss += float(loss_sum.detach().cpu())
        total_count += count

    if total_count <= 0.0:
        raise ValueError("No evaluation entries were used")
    return total_loss / total_count


@torch.no_grad()
def predict_with_window_average(model, values, train_mask, window_starts,
                                window_len, batch_size, device):
    model.eval()
    pred_sum = np.zeros_like(values, dtype="float32")
    pred_count = np.zeros_like(values, dtype="float32")
    zero_mask = np.zeros_like(train_mask, dtype="float32")

    for batch_starts in iter_start_batches(window_starts, batch_size):
        x_enc, model_mask, _, _ = make_window_batch(
            values,
            train_mask,
            zero_mask,
            batch_starts,
            window_len,
            device,
        )
        output = model(x_enc, None, None, None, model_mask)
        output_np = output.detach().cpu().numpy().astype("float32")
        for batch_index, start in enumerate(batch_starts):
            end = int(start) + window_len
            pred_sum[start:end] += output_np[batch_index]
            pred_count[start:end] += 1.0

    uncovered = pred_count == 0.0
    if np.any(uncovered):
        pred_count[uncovered] = 1.0
        pred_sum[uncovered] = values[uncovered]
    return pred_sum / pred_count


def evaluate_predictions(pred, indices, values, dst_count):
    time, variable = flat_positions(indices, dst_count)
    y_pred = pred[time, variable]
    y_true = values.astype("float32")
    error = y_pred - y_true
    abs_denom = float(np.sum(np.abs(y_true)))
    sq_denom = float(np.sum(np.square(y_true)))
    if abs_denom == 0.0 or sq_denom == 0.0:
        raise ValueError("Metric denominator is zero")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "nmae": float(np.sum(np.abs(error)) / abs_denom),
        "nrmse": float(np.sqrt(np.sum(np.square(error)) / sq_denom)),
        "entries": int(values.shape[0]),
    }


def default_output_path(output_dir, observed_ratio, val_ratio, seed,
                        target_normalization):
    name = "moderntcn_random_observed%d_val%d_seed%d_norm_%s.json" % (
        int(round(observed_ratio * 100)),
        int(round(val_ratio * 100)),
        seed,
        target_normalization,
    )
    return os.path.join(output_dir, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ModernTCN tensor completion on satellite path traffic."
    )
    parser.add_argument(
        "--tensor-path",
        default=os.path.join("data", "sat_path_bytes_mb_tensor.npy"),
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--target-normalization",
        choices=["max", "none"],
        default="max",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--stem-ratio", type=int, default=6)
    parser.add_argument("--downsample-ratio", type=int, default=2)
    parser.add_argument("--ffn-ratio", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-stride", type=int, default=1)
    parser.add_argument("--num-blocks", default="1")
    parser.add_argument("--large-size", default="31")
    parser.add_argument("--small-size", default="5")
    parser.add_argument("--dims", default="64")
    parser.add_argument("--dw-dims", default="64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--use-multi-scale", action="store_true")
    parser.add_argument("--revin", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--subtract-last", type=int, default=0)
    parser.add_argument("--individual", type=int, default=0)
    parser.add_argument("--kernel-size", type=int, default=25)
    parser.add_argument(
        "--window-len",
        type=int,
        default=96,
        help="Temporal sliding-window length for full-feature ModernTCN training.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=48,
        help="Temporal stride between adjacent training/evaluation windows.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    return parser.parse_args()


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
            args.target_normalization,
        )

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"
    )

    tensor, split, data_stats = create_random_completion_split(
        args.tensor_path,
        args.split_path,
        observed_ratio,
        args.val_ratio,
        args.seed,
    )
    shape = tuple(int(v) for v in split["shape"])
    seq_len = shape[2]
    input_dim = shape[0] * shape[1]
    target_scale = get_target_scale(split["train_values"], args.target_normalization)

    values = flatten_time_major(np.nan_to_num(tensor, nan=0.0)) / target_scale
    train_mask = np.zeros_like(values, dtype="float32")
    train_time, train_var = flat_positions(split["train_indices"], shape[1])
    train_mask[train_time, train_var] = 1.0

    val_mask = np.zeros_like(values, dtype="float32")
    val_time, val_var = flat_positions(split["val_indices"], shape[1])
    val_mask[val_time, val_var] = 1.0

    window_starts, actual_window_len = make_window_starts(
        seq_len,
        args.window_len,
        args.window_stride,
    )
    train_window_starts = filter_window_starts_by_mask(
        window_starts,
        train_mask,
        actual_window_len,
    )
    val_window_starts = filter_window_starts_by_mask(
        window_starts,
        val_mask,
        actual_window_len,
    )

    configs = build_moderntcn_configs(args, input_dim, actual_window_len)
    model = Model(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_rng = np.random.RandomState(args.seed + 2027)

    print("Tensor shape:", list(shape))
    print("Split mode: random transductive completion")
    print("Training mode: full-feature sliding-window batches")
    print("Total entries:", data_stats["total_entries"])
    print("Finite entries:", data_stats["finite_entries"])
    print("Non-zero finite entries:", data_stats["nonzero_finite_entries"])
    print("Excluded zero finite entries:", data_stats["zero_finite_entries"])
    print("Excluded non-finite entries:", data_stats["nonfinite_entries"])
    print("Observed ratio:", observed_ratio)
    print("Missing rate:", 1.0 - observed_ratio)
    print("Validation ratio within unobserved entries:", args.val_ratio)
    print("Train entries:", int(split["train_indices"].shape[0]))
    print("Validation entries:", int(split["val_indices"].shape[0]))
    print("Test entries:", int(split["test_indices"].shape[0]))
    print("NMAE denominator: sum(abs(true_values))")
    print("NRMSE denominator: sqrt(sum(square(error)) / sum(square(true_values)))")
    print("Split path:", args.split_path)
    print("Metrics path:", args.metrics_path)
    print("Target normalization:", args.target_normalization)
    print("Target scale:", target_scale)
    print("Window len/stride/count:", actual_window_len, args.window_stride,
          int(window_starts.shape[0]))
    print("Train/val windows:", int(train_window_starts.shape[0]),
          int(val_window_starts.shape[0]))
    print("Batch/eval batch:", args.batch_size, args.eval_batch_size)
    print("Device:", device)

    best_state = None
    best_val = float("inf")
    wait = 0
    for epoch in range(1, args.epochs + 1):
        train_value = train_one_epoch(
            model,
            optimizer,
            values,
            train_mask,
            train_window_starts,
            actual_window_len,
            args.batch_size,
            device,
            epoch_rng,
        )
        val_value = evaluate_window_mae(
            model,
            values,
            train_mask,
            val_mask,
            val_window_starts,
            actual_window_len,
            args.eval_batch_size,
            device,
        )
        print(
            "Epoch %03d/%03d - train_mae %.6f - val_mae %.6f" %
            (epoch, args.epochs, train_value, val_value)
        )

        if val_value < best_val:
            best_val = val_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping at epoch:", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    pred_np = predict_with_window_average(
        model,
        values,
        train_mask,
        window_starts,
        actual_window_len,
        args.eval_batch_size,
        device,
    ) * target_scale

    config_dict = vars(args).copy()
    config_dict.update({
        "tensor_path": args.tensor_path,
        "split_mode": "random_transductive_completion",
        "sample_filter": "finite_and_nonzero",
        "total_entries": data_stats["total_entries"],
        "finite_entries": data_stats["finite_entries"],
        "nonzero_finite_entries": data_stats["nonzero_finite_entries"],
        "zero_finite_entries_excluded": data_stats["zero_finite_entries"],
        "nonfinite_entries_excluded": data_stats["nonfinite_entries"],
        "observed_ratio": observed_ratio,
        "missing_rate": 1.0 - observed_ratio,
        "val_ratio_within_unobserved": args.val_ratio,
        "train_entries": int(split["train_indices"].shape[0]),
        "val_entries": int(split["val_indices"].shape[0]),
        "test_entries": int(split["test_indices"].shape[0]),
        "split_path": args.split_path,
        "metrics_path": args.metrics_path,
        "training_mode": "full_feature_sliding_window",
        "window_len": actual_window_len,
        "window_stride": args.window_stride,
        "window_count": int(window_starts.shape[0]),
        "train_window_count": int(train_window_starts.shape[0]),
        "val_window_count": int(val_window_starts.shape[0]),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "prediction_aggregation": "average_over_overlapping_windows",
        "target_scale": target_scale,
        "metrics_scale": "original",
        "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
        "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
    })

    results = {
        "config": config_dict,
        "train": evaluate_predictions(
            pred_np, split["train_indices"], split["train_values"], shape[1]
        ),
        "val": evaluate_predictions(
            pred_np, split["val_indices"], split["val_values"], shape[1]
        ),
        "test": evaluate_predictions(
            pred_np, split["test_indices"], split["test_values"], shape[1]
        ),
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
