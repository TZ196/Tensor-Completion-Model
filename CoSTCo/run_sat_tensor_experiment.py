import argparse
import json
import os
from pprint import pprint

import numpy as np
from tensorflow import keras as k

from costco_model import (
    compile_costco,
    configure_tensorflow,
    create_costco,
    evaluate_costco,
    transform_indices,
)


def load_tensor(path):
    tensor = np.load(path)
    if tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape %s" % (tensor.shape,))
    return tensor.astype("float32")


def nonzero_finite_entries(tensor):
    finite_mask = np.isfinite(tensor)
    mask = finite_mask & (tensor != 0)
    indices = np.argwhere(mask).astype("int32")
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

    train_indices = indices[train_order]
    train_values = values[train_order]
    val_indices = indices[val_order]
    val_values = values[val_order]
    test_indices = indices[test_order]
    test_values = values[test_order]

    split_dir = os.path.dirname(split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)

    np.savez(
        split_path,
        shape=np.array(tensor.shape).astype("int32"),
        train_indices=train_indices,
        train_values=train_values,
        val_indices=val_indices,
        val_values=val_values,
        test_indices=test_indices,
        test_values=test_values,
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
        nonfinite_entries=np.array(
            data_stats["nonfinite_entries"]
        ).astype("int64"),
    )

    return (
        np.array(tensor.shape).astype("int32"),
        train_indices,
        train_values,
        val_indices,
        val_values,
        test_indices,
        test_values,
        data_stats,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CoSTCo tensor completion on sat_path_bytes_tensor.npy."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--nc", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


def default_output_path(output_dir, observed_ratio, val_ratio, seed, rank, nc):
    name = "random_observed%d_val%d_seed%d_rank%d_nc%d.json" % (
        int(round(observed_ratio * 100)),
        int(round(val_ratio * 100)),
        seed,
        rank,
        nc,
    )
    return os.path.join(output_dir, name)


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
            )
        )

    if args.nc is None:
        nc = args.rank
    else:
        nc = args.nc

    if args.metrics_path is None:
        args.metrics_path = default_output_path(
            "results",
            observed_ratio,
            args.val_ratio,
            args.seed,
            args.rank,
            nc,
        )

    configure_tensorflow(cpu_only=args.cpu_only, seed=args.seed)

    (
        shape,
        train_indices,
        train_values,
        val_indices,
        val_values,
        test_indices,
        test_values,
        data_stats,
    ) = (
        create_random_completion_split(
            args.tensor_path,
            args.split_path,
            observed_ratio,
            args.val_ratio,
            args.seed,
        )
    )

    print("Tensor shape:", shape.tolist())
    print("Split mode: random transductive completion")
    print("Total entries:", data_stats["total_entries"])
    print("Finite entries:", data_stats["finite_entries"])
    print("Non-zero finite entries:", data_stats["nonzero_finite_entries"])
    print("Excluded zero finite entries:", data_stats["zero_finite_entries"])
    print("Excluded non-finite entries:", data_stats["nonfinite_entries"])
    print("Observed ratio:", observed_ratio)
    print("Missing rate:", 1.0 - observed_ratio)
    print("Validation ratio within unobserved entries:", args.val_ratio)
    print("Train entries:", train_indices.shape[0])
    print("Validation entries:", val_indices.shape[0])
    print("Test entries:", test_indices.shape[0])
    print("NMAE denominator: sum(abs(true_values))")
    print("NRMSE denominator: sqrt(sum(square(error)) / sum(square(true_values)))")
    print("Split path:", args.split_path)
    print("Metrics path:", args.metrics_path)

    model = create_costco(shape, rank=args.rank, nc=nc)
    compile_costco(model, lr=args.lr)
    model.fit(
        x=transform_indices(train_indices),
        y=train_values,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            transform_indices(val_indices),
            val_values,
        ),
        callbacks=[
            k.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=10,
                restore_best_weights=True
            )
        ],
    )

    results = {
        "config": {
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
            "train_entries": int(train_indices.shape[0]),
            "val_entries": int(val_indices.shape[0]),
            "test_entries": int(test_indices.shape[0]),
            "split_path": args.split_path,
            "metrics_path": args.metrics_path,
            "rank": args.rank,
            "nc": nc,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
        },
        "train": evaluate_costco(
            model,
            train_indices,
            train_values,
        ),
        "val": evaluate_costco(
            model,
            val_indices,
            val_values,
        ),
        "test": evaluate_costco(
            model,
            test_indices,
            test_values,
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
