import argparse
import json
import os
import time
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
        description="Run SatFormer tensor completion on sat_path_bytes_mb_tensor.npy."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_mb_tensor.npy")
    parser.add_argument(
        "--adjacency-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--gcn-hidden-dim", type=int, default=64)
    parser.add_argument("--num-modules", type=int, default=4)
    parser.add_argument("--region-size", type=int, default=16)
    parser.add_argument("--center-window", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history-window", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-log-every", type=int, default=10)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--eval-splits",
        choices=["none", "test", "val-test", "all"],
        default="test",
        help="Final metric splits to evaluate after training. Validation loss is still used every epoch.",
    )
    parser.add_argument(
        "--target-normalization",
        choices=["max", "none"],
        default="max",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


def default_output_path(
    output_dir,
    observed_ratio,
    val_ratio,
    seed,
    feature_dim,
    num_modules,
    batch_size,
    history_window,
    target_normalization,
):
    history_label = "full" if history_window <= 0 else str(history_window)
    name = "random_observed%d_val%d_seed%d_dim%d_layers%d_batch%d_hist%s_norm_%s.json" % (
        int(round(observed_ratio * 100)),
        int(round(val_ratio * 100)),
        seed,
        feature_dim,
        num_modules,
        batch_size,
        history_label,
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


def format_duration(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return "%dh %02dm %05.2fs" % (hours, minutes, secs)
    if minutes > 0:
        return "%dm %05.2fs" % (minutes, secs)
    return "%.2fs" % secs


def build_observed_tensor(shape, indices, values, target_scale, device):
    observed = torch.zeros(tuple(shape.tolist()), dtype=torch.float32, device=device)
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    vals = torch.as_tensor(values / target_scale, dtype=torch.float32, device=device)
    observed[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
    return observed


def gather_entries(tensor, indices):
    idx = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor[idx[:, 0], idx[:, 1], idx[:, 2]]


def iter_time_entry_batches(indices, values, batch_size, rng):
    time_steps = np.unique(indices[:, 2])
    rng.shuffle(time_steps)
    for time_step in time_steps:
        time_positions = np.flatnonzero(indices[:, 2] == time_step)
        time_positions = time_positions[rng.permutation(time_positions.shape[0])]
        for start in range(0, time_positions.shape[0], batch_size):
            positions = time_positions[start : start + batch_size]
            yield int(time_step), indices[positions], values[positions]


def iter_time_groups(indices, values, rng):
    time_steps = np.unique(indices[:, 2])
    rng.shuffle(time_steps)
    for time_step in time_steps:
        positions = np.flatnonzero(indices[:, 2] == time_step)
        positions = positions[rng.permutation(positions.shape[0])]
        yield int(time_step), indices[positions], values[positions]


def time_window_bounds(time_step, history_window):
    if history_window <= 0:
        return 0, time_step
    return max(0, time_step - history_window + 1), time_step


def predict_entries_for_time(
    model,
    input_tensor,
    adjacency_tensor,
    time_step,
    entry_indices,
    history_window,
):
    start, end = time_window_bounds(time_step, history_window)
    target_offset = time_step - start
    prediction_matrix = model.forward_time_window(
        input_tensor[:, :, start : end + 1],
        adjacency_tensor[:, :, start : end + 1],
        target_offset=target_offset,
    )
    rows = torch.as_tensor(entry_indices[:, 0], dtype=torch.long, device=input_tensor.device)
    cols = torch.as_tensor(entry_indices[:, 1], dtype=torch.long, device=input_tensor.device)
    return prediction_matrix[rows, cols]


def predict_time_matrix(
    model,
    input_tensor,
    adjacency_tensor,
    time_step,
    history_window,
):
    start, end = time_window_bounds(time_step, history_window)
    target_offset = time_step - start
    return model.forward_time_window(
        input_tensor[:, :, start : end + 1],
        adjacency_tensor[:, :, start : end + 1],
        target_offset=target_offset,
    )


def gather_matrix_entries(prediction_matrix, entry_indices):
    rows = torch.as_tensor(entry_indices[:, 0], dtype=torch.long, device=prediction_matrix.device)
    cols = torch.as_tensor(entry_indices[:, 1], dtype=torch.long, device=prediction_matrix.device)
    return prediction_matrix[rows, cols]


def set_warmup_lr(optimizer, base_lr, epoch, warmup_epochs):
    if warmup_epochs <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, float(epoch + 1) / float(warmup_epochs))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def evaluate_mse_loss(
    model,
    input_tensor,
    adjacency_tensor,
    indices,
    values,
    target_scale,
    history_window,
    device,
):
    model.eval()
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for time_step in np.unique(indices[:, 2]):
            positions = np.flatnonzero(indices[:, 2] == time_step)
            batch_indices = indices[positions]
            targets = torch.as_tensor(
                values[positions] / target_scale,
                dtype=torch.float32,
                device=device,
            )
            predictions = predict_entries_for_time(
                model,
                input_tensor,
                adjacency_tensor,
                int(time_step),
                batch_indices,
                history_window,
            )
            squared_error += torch.sum((predictions - targets) ** 2).item()
            count += targets.numel()
    return squared_error / max(1, count)


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


def evaluate_satformer(
    model,
    input_tensor,
    adjacency_tensor,
    indices,
    values,
    target_scale,
    history_window,
    split_name="eval",
    log_every=10,
):
    model.eval()
    predictions = np.empty(values.shape[0], dtype="float32")
    time_steps = np.unique(indices[:, 2])
    eval_start_time = time.perf_counter()
    print(
        "Final %s evaluation starting: %d entries across %d time step(s)"
        % (split_name, values.shape[0], time_steps.shape[0]),
        flush=True,
    )
    with torch.no_grad():
        for step_index, time_step in enumerate(time_steps, start=1):
            step_start_time = time.perf_counter()
            positions = np.flatnonzero(indices[:, 2] == time_step)
            batch_indices = indices[positions]
            pred = predict_entries_for_time(
                model,
                input_tensor,
                adjacency_tensor,
                int(time_step),
                batch_indices,
                history_window,
            )
            predictions[positions] = pred.detach().cpu().numpy().astype("float32")
            should_log = log_every > 0 and (
                step_index == 1
                or step_index == time_steps.shape[0]
                or step_index % log_every == 0
            )
            if should_log:
                print(
                    "Final %s evaluation step %d/%d target_time %d entries %d step_time: %s"
                    % (
                        split_name,
                        step_index,
                        time_steps.shape[0],
                        int(time_step),
                        positions.shape[0],
                        format_duration(time.perf_counter() - step_start_time),
                    ),
                    flush=True,
                )
    pred = predictions * target_scale
    pred = np.maximum(pred, 0.0)
    metrics = {
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
    print(
        "Final %s evaluation finished in %s: NMAE %.6f, NRMSE %.6f"
        % (
            split_name,
            format_duration(time.perf_counter() - eval_start_time),
            metrics["nmae"],
            metrics["nrmse"],
        ),
        flush=True,
    )
    return metrics


def selected_eval_splits(eval_splits):
    if eval_splits == "none":
        return []
    if eval_splits == "test":
        return ["test"]
    if eval_splits == "val-test":
        return ["val", "test"]
    if eval_splits == "all":
        return ["train", "val", "test"]
    raise ValueError("Unsupported --eval-splits value: %s" % eval_splits)


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
            args.batch_size,
            args.history_window,
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
    print("Batch size:", args.batch_size)
    print("Training objective: masked tensor window -> one target-time traffic matrix")
    print("Training update unit: one target time step per optimizer step")
    print("History window:", "full" if args.history_window <= 0 else args.history_window)
    print("Warmup epochs:", args.warmup_epochs)
    print("Log every steps:", args.log_every)
    print("Final eval splits:", args.eval_splits)
    print("Final eval log every time steps:", args.eval_log_every)
    print(
        "Max train steps per epoch:",
        "all" if args.max_train_steps_per_epoch <= 0 else args.max_train_steps_per_epoch,
    )
    print("Gradient checkpointing:", not args.no_gradient_checkpointing)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    rng = np.random.RandomState(args.seed)
    completed_epochs = 0
    train_start_time = time.perf_counter()
    for epoch in range(args.epochs):
        current_lr = set_warmup_lr(optimizer, args.lr, epoch, args.warmup_epochs)
        model.train()
        epoch_loss_sum = 0.0
        epoch_entry_count = 0
        epoch_step_count = 0
        epoch_time_steps = np.unique(train_indices[:, 2]).shape[0]
        print(
            "Epoch [%d/%d] starting %d time-step update(s)"
            % (epoch + 1, args.epochs, epoch_time_steps),
            flush=True,
        )
        for time_step, batch_indices, batch_values in iter_time_groups(
            train_indices,
            train_values,
            rng,
        ):
            next_step = epoch_step_count + 1
            should_log_step = args.log_every > 0 and (
                next_step == 1 or next_step % args.log_every == 0
            )
            if should_log_step:
                print(
                    "Epoch [%d/%d] step %d/%d target_time %d starting"
                    % (epoch + 1, args.epochs, next_step, epoch_time_steps, time_step),
                    flush=True,
                )
            step_start_time = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction_matrix = predict_time_matrix(
                model,
                input_tensor,
                adjacency_tensor,
                time_step,
                args.history_window,
            )
            predictions = gather_matrix_entries(prediction_matrix, batch_indices)
            targets = torch.as_tensor(
                batch_values / target_scale,
                dtype=torch.float32,
                device=device,
            )
            train_loss = criterion(predictions, targets)
            train_loss.backward()
            optimizer.step()

            batch_size = targets.numel()
            epoch_loss_sum += train_loss.item() * batch_size
            epoch_entry_count += batch_size
            epoch_step_count += 1

            del prediction_matrix, predictions, targets, train_loss
            if should_log_step:
                running_loss = epoch_loss_sum / max(1, epoch_entry_count)
                print(
                    "Epoch [%d/%d] step %d/%d entries %d running_loss: %.6f step_time: %s"
                    % (
                        epoch + 1,
                        args.epochs,
                        epoch_step_count,
                        epoch_time_steps,
                        epoch_entry_count,
                        running_loss,
                        format_duration(time.perf_counter() - step_start_time),
                    ),
                    flush=True,
                )
            if (
                args.max_train_steps_per_epoch > 0
                and epoch_step_count >= args.max_train_steps_per_epoch
            ):
                print(
                    "Reached --max-train-steps-per-epoch=%d for debug run"
                    % args.max_train_steps_per_epoch,
                    flush=True,
                )
                break

        train_loss_value = epoch_loss_sum / max(1, epoch_entry_count)
        val_loss_value = evaluate_mse_loss(
            model,
            input_tensor,
            adjacency_tensor,
            val_indices,
            val_values,
            target_scale,
            args.history_window,
            device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            "Epoch [%d/%d] lr: %.6g steps: %d loss: %.6f val_loss: %.6f"
            % (
                epoch + 1,
                args.epochs,
                current_lr,
                epoch_step_count,
                train_loss_value,
                val_loss_value,
            )
        )
        completed_epochs = epoch + 1

        if val_loss_value < best_val:
            best_val = val_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping at epoch %d" % (epoch + 1))
                break

    training_elapsed_seconds = time.perf_counter() - train_start_time
    print(
        "Training finished in %s (%.2f seconds) over %d epoch(s)"
        % (
            format_duration(training_elapsed_seconds),
            training_elapsed_seconds,
            completed_epochs,
        ),
        flush=True,
    )

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
            "completed_epochs": completed_epochs,
            "batch_size": args.batch_size,
            "history_window": args.history_window,
            "warmup_epochs": args.warmup_epochs,
            "log_every": args.log_every,
            "eval_log_every": args.eval_log_every,
            "eval_splits": args.eval_splits,
            "max_train_steps_per_epoch": args.max_train_steps_per_epoch,
            "patience": args.patience,
            "target_normalization": args.target_normalization,
            "target_scale": target_scale,
            "metrics_scale": "original",
            "seed": args.seed,
            "training_time_seconds": training_elapsed_seconds,
            "training_time": format_duration(training_elapsed_seconds),
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
        },
    }
    eval_data = {
        "train": (train_indices, train_values),
        "val": (val_indices, val_values),
        "test": (test_indices, test_values),
    }
    for split_name in selected_eval_splits(args.eval_splits):
        split_indices, split_values = eval_data[split_name]
        results[split_name] = evaluate_satformer(
            model,
            input_tensor,
            adjacency_tensor,
            split_indices,
            split_values,
            target_scale,
            args.history_window,
            split_name=split_name,
            log_every=args.eval_log_every,
        )

    pprint(results)
    metrics_dir = os.path.dirname(args.metrics_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    with open(args.metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()
