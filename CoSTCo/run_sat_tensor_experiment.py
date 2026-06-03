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


def create_temporal_split(tensor_path, split_path, train_ratio, val_ratio):
    tensor = load_tensor(tensor_path)
    time_steps = tensor.shape[2]
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")

    train_end = int(round(time_steps * train_ratio))
    val_end = train_end + int(round(time_steps * val_ratio))
    train_end = max(1, min(train_end, time_steps - 2))
    val_end = max(train_end + 1, min(val_end, time_steps - 1))

    train_indices, train_values = finite_entries_in_time_range(
        tensor, 0, train_end
    )
    val_indices, val_values = finite_entries_in_time_range(
        tensor, train_end, val_end
    )
    test_indices, test_values = finite_entries_in_time_range(
        tensor, val_end, time_steps
    )

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
        train_time_range=np.array([0, train_end]).astype("int32"),
        val_time_range=np.array([train_end, val_end]).astype("int32"),
        test_time_range=np.array([val_end, time_steps]).astype("int32"),
    )

    return (
        np.array(tensor.shape).astype("int32"),
        train_indices,
        train_values,
        val_indices,
        val_values,
        test_indices,
        test_values,
        (0, train_end),
        (train_end, val_end),
        (val_end, time_steps),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CoSTCo tensor completion on sat_path_bytes_tensor.npy."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
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
    if args.split_path is None:
        args.split_path = os.path.join(
            "splits",
            "temporal_train%d_val%d_seed_%d.npz" % (
                int(args.train_ratio * 100),
                int(args.val_ratio * 100),
                args.seed,
            )
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
        train_time_range,
        val_time_range,
        test_time_range,
    ) = (
        create_temporal_split(
            args.tensor_path,
            args.split_path,
            args.train_ratio,
            args.val_ratio,
        )
    )

    print("Tensor shape:", shape.tolist())
    print("Train entries:", train_indices.shape[0])
    print("Validation entries:", val_indices.shape[0])
    print("Test entries:", test_indices.shape[0])
    print("Train time range:", train_time_range)
    print("Validation time range:", val_time_range)
    print("Test time range:", test_time_range)
    print("NMAE denominator: sum(abs(true_values))")
    print("NRMSE denominator: sqrt(sum(square(error)) / sum(square(true_values)))")
    print("Split path:", args.split_path)

    model = create_costco(shape, rank=args.rank, nc=args.nc)
    compile_costco(model, lr=args.lr)
    model.fit(
        x=transform_indices(train_indices),
        y=train_values,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(transform_indices(val_indices), val_values),
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
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
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
        "train": evaluate_costco(model, train_indices, train_values),
        "val": evaluate_costco(model, val_indices, val_values),
        "test": evaluate_costco(model, test_indices, test_values),
    }

    pprint(results)
    with open(args.metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()
