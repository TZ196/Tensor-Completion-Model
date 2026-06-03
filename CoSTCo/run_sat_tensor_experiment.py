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


def finite_entries_in_time_range(tensor, start_t, end_t):
    block = tensor[:, :, start_t:end_t]
    mask = np.isfinite(block)
    indices = np.argwhere(mask).astype("int32")
    indices[:, 2] += start_t
    values = block[mask].astype("float32")
    return indices, values


def split_observed_missing(indices, values, observed_ratio, seed):
    if not 0.0 < observed_ratio < 1.0:
        raise ValueError("--observed-ratio must be between 0 and 1")
    rng = np.random.RandomState(seed)
    order = rng.permutation(indices.shape[0])
    observed_size = int(round(indices.shape[0] * observed_ratio))
    observed_size = max(1, observed_size)
    observed_size = min(observed_size, indices.shape[0] - 1)
    observed_order = order[:observed_size]
    missing_order = order[observed_size:]
    return (
        indices[observed_order],
        values[observed_order],
        indices[missing_order],
        values[missing_order],
    )


def create_temporal_split(tensor_path, split_path, train_ratio, val_ratio,
                          observed_ratio, seed):
    tensor = load_tensor(tensor_path)
    time_steps = tensor.shape[2]
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio >= 1.0:
        raise ValueError("--train-ratio must leave time slices for testing")

    train_val_end = int(round(time_steps * train_ratio))
    val_size = int(round(train_val_end * val_ratio))
    train_val_end = max(2, min(train_val_end, time_steps - 1))
    val_size = max(1, min(val_size, train_val_end - 1))
    train_end = train_val_end - val_size

    train_indices, train_values = finite_entries_in_time_range(
        tensor, 0, train_end
    )
    (
        train_observed_indices,
        train_observed_values,
        train_missing_indices,
        train_missing_values,
    ) = split_observed_missing(
        train_indices,
        train_values,
        observed_ratio,
        seed,
    )
    val_indices, val_values = finite_entries_in_time_range(
        tensor, train_end, train_val_end
    )
    (
        val_observed_indices,
        val_observed_values,
        val_missing_indices,
        val_missing_values,
    ) = split_observed_missing(
        val_indices,
        val_values,
        observed_ratio,
        seed + 1,
    )
    test_indices, test_values = finite_entries_in_time_range(
        tensor, train_val_end, time_steps
    )
    (
        test_observed_indices,
        test_observed_values,
        test_missing_indices,
        test_missing_values,
    ) = split_observed_missing(
        test_indices,
        test_values,
        observed_ratio,
        seed + 2,
    )

    split_dir = os.path.dirname(split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)

    np.savez(
        split_path,
        shape=np.array(tensor.shape).astype("int32"),
        train_observed_indices=train_observed_indices,
        train_observed_values=train_observed_values,
        train_missing_indices=train_missing_indices,
        train_missing_values=train_missing_values,
        val_observed_indices=val_observed_indices,
        val_observed_values=val_observed_values,
        val_missing_indices=val_missing_indices,
        val_missing_values=val_missing_values,
        test_observed_indices=test_observed_indices,
        test_observed_values=test_observed_values,
        test_missing_indices=test_missing_indices,
        test_missing_values=test_missing_values,
        train_time_range=np.array([0, train_end]).astype("int32"),
        val_time_range=np.array([train_end, train_val_end]).astype("int32"),
        test_time_range=np.array([train_val_end, time_steps]).astype("int32"),
        observed_ratio=np.array(observed_ratio).astype("float32"),
        missing_rate=np.array(1.0 - observed_ratio).astype("float32"),
        seed=np.array(seed).astype("int32"),
    )

    return (
        np.array(tensor.shape).astype("int32"),
        train_observed_indices,
        train_observed_values,
        train_missing_indices,
        train_missing_values,
        val_observed_indices,
        val_observed_values,
        val_missing_indices,
        val_missing_values,
        test_observed_indices,
        test_observed_values,
        test_missing_indices,
        test_missing_values,
        (0, train_end),
        (train_end, train_val_end),
        (train_val_end, time_steps),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CoSTCo tensor completion on sat_path_bytes_tensor.npy."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--nc", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default="results_sat_costco.json")
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
            "temporal_train%d_val%d_observed%d_seed_%d.npz" % (
                int(round(args.train_ratio * 100)),
                int(round(args.val_ratio * 100)),
                int(round(observed_ratio * 100)),
                args.seed,
            )
        )

    configure_tensorflow(cpu_only=args.cpu_only, seed=args.seed)

    (
        shape,
        train_observed_indices,
        train_observed_values,
        train_missing_indices,
        train_missing_values,
        val_observed_indices,
        val_observed_values,
        val_missing_indices,
        val_missing_values,
        test_observed_indices,
        test_observed_values,
        test_missing_indices,
        test_missing_values,
        train_time_range,
        val_time_range,
        test_time_range,
    ) = (
        create_temporal_split(
            args.tensor_path,
            args.split_path,
            args.train_ratio,
            args.val_ratio,
            observed_ratio,
            args.seed,
        )
    )

    print("Tensor shape:", shape.tolist())
    print("Observed ratio:", observed_ratio)
    print("Missing rate:", 1.0 - observed_ratio)
    print("Train observed entries:", train_observed_indices.shape[0])
    print("Train missing entries:", train_missing_indices.shape[0])
    print("Validation observed entries:", val_observed_indices.shape[0])
    print("Validation missing entries:", val_missing_indices.shape[0])
    print("Test observed entries:", test_observed_indices.shape[0])
    print("Test missing entries:", test_missing_indices.shape[0])
    print("Train time range:", train_time_range)
    print("Validation time range:", val_time_range)
    print("Test time range:", test_time_range)
    print("NMAE denominator: sum(abs(true_values))")
    print("NRMSE denominator: sqrt(sum(square(error)) / sum(square(true_values)))")
    print("Split path:", args.split_path)

    model = create_costco(shape, rank=args.rank, nc=args.nc)
    compile_costco(model, lr=args.lr)
    model.fit(
        x=transform_indices(train_observed_indices),
        y=train_observed_values,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            transform_indices(val_missing_indices),
            val_missing_values,
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
            "split_mode": "temporal",
            "train_val_time_ratio": args.train_ratio,
            "val_ratio_within_train": args.val_ratio,
            "test_time_ratio": 1.0 - args.train_ratio,
            "observed_ratio": observed_ratio,
            "missing_rate": 1.0 - observed_ratio,
            "train_observed_entries": int(train_observed_indices.shape[0]),
            "train_missing_entries": int(train_missing_indices.shape[0]),
            "val_observed_entries": int(val_observed_indices.shape[0]),
            "val_missing_entries": int(val_missing_indices.shape[0]),
            "test_observed_entries": int(test_observed_indices.shape[0]),
            "test_missing_entries": int(test_missing_indices.shape[0]),
            "train_time_range": train_time_range,
            "val_time_range": val_time_range,
            "test_time_range": test_time_range,
            "split_path": args.split_path,
            "rank": args.rank,
            "nc": args.nc if args.nc is not None else args.rank,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
        },
        "train_observed": evaluate_costco(
            model,
            train_observed_indices,
            train_observed_values,
        ),
        "train_missing": evaluate_costco(
            model,
            train_missing_indices,
            train_missing_values,
        ),
        "val_missing": evaluate_costco(
            model,
            val_missing_indices,
            val_missing_values,
        ),
        "test_missing": evaluate_costco(
            model,
            test_missing_indices,
            test_missing_values,
        ),
    }

    pprint(results)
    with open(args.metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()
