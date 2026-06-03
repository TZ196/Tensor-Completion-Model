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


def finite_entries(tensor):
    mask = np.isfinite(tensor)
    indices = np.argwhere(mask).astype("int32")
    values = tensor[mask].astype("float32")
    return indices, values


def split_entries(indices, values, missing_rate, seed):
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("--missing-rate must be between 0 and 1")

    rng = np.random.RandomState(seed)
    order = rng.permutation(indices.shape[0])
    train_size = int(round(indices.shape[0] * (1.0 - missing_rate)))
    train_order = order[:train_size]
    test_order = order[train_size:]

    return (
        indices[train_order],
        values[train_order],
        indices[test_order],
        values[test_order],
    )


def get_or_create_split(tensor_path, split_path, missing_rate, seed):
    if os.path.exists(split_path):
        split = np.load(split_path)
        return (
            split["shape"],
            split["train_indices"],
            split["train_values"],
            split["test_indices"],
            split["test_values"],
        )

    tensor = load_tensor(tensor_path)
    indices, values = finite_entries(tensor)
    train_indices, train_values, test_indices, test_values = split_entries(
        indices, values, missing_rate, seed
    )

    split_dir = os.path.dirname(split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)

    np.savez(
        split_path,
        shape=np.array(tensor.shape).astype("int32"),
        train_indices=train_indices,
        train_values=train_values,
        test_indices=test_indices,
        test_values=test_values,
        missing_rate=np.array(missing_rate).astype("float32"),
        seed=np.array(seed).astype("int32"),
    )

    return (
        np.array(tensor.shape).astype("int32"),
        train_indices,
        train_values,
        test_indices,
        test_values,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CoSTCo tensor completion on sat_path_bytes_tensor.npy."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--split-path", default=None)
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
            "missing_%d_seed_%d.npz" % (int(args.missing_rate * 100), args.seed)
        )

    configure_tensorflow(cpu_only=args.cpu_only, seed=args.seed)

    shape, train_indices, train_values, test_indices, test_values = (
        get_or_create_split(
            args.tensor_path,
            args.split_path,
            args.missing_rate,
            args.seed,
        )
    )

    print("Tensor shape:", shape.tolist())
    print("Train entries:", train_indices.shape[0])
    print("Test entries:", test_indices.shape[0])
    print("Missing rate:", args.missing_rate)
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
        validation_split=0.1,
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
            "missing_rate": args.missing_rate,
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
        "test": evaluate_costco(model, test_indices, test_values),
    }

    pprint(results)
    with open(args.metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()
