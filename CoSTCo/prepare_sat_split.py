from __future__ import print_function

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible train/test split."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--split-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split_path is None:
        args.split_path = os.path.join(
            "splits",
            "missing_%d_seed_%d.npz" % (int(args.missing_rate * 100), args.seed)
        )

    tensor = np.load(args.tensor_path).astype("float32")
    mask = np.isfinite(tensor)
    indices = np.argwhere(mask).astype("int32")
    values = tensor[mask].astype("float32")

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(indices.shape[0])
    train_size = int(round(indices.shape[0] * (1.0 - args.missing_rate)))
    train_order = order[:train_size]
    test_order = order[train_size:]

    split_dir = os.path.dirname(args.split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)

    np.savez(
        args.split_path,
        shape=np.array(tensor.shape).astype("int32"),
        train_indices=indices[train_order],
        train_values=values[train_order],
        test_indices=indices[test_order],
        test_values=values[test_order],
        missing_rate=np.array(args.missing_rate).astype("float32"),
        seed=np.array(args.seed).astype("int32"),
    )

    print("Saved split:", args.split_path)
    print("Tensor shape:", tensor.shape)
    print("Train entries:", train_order.shape[0])
    print("Test entries:", test_order.shape[0])


if __name__ == "__main__":
    main()
