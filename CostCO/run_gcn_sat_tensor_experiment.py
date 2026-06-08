import argparse
import json
import os
from pprint import pprint

from tensorflow import keras as k

from gcn_costco_model import (
    compile_gcn_costco,
    create_gcn_costco,
    evaluate_gcn_costco,
)
from run_sat_tensor_experiment import (
    configure_tensorflow,
    create_random_completion_split,
    default_output_path,
    get_target_scale,
    load_connectivity_tensor,
    transform_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GCN-CoSTCo tensor completion with adjacency fusion."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_tensor.npy")
    parser.add_argument(
        "--topology-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=32)
    parser.add_argument("--gcn-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--target-normalization",
        choices=["max", "none"],
        default="max",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--metrics-path", default=None)
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
            args.rank,
            args.nc,
            args.target_normalization,
            model_name="gcn_costco",
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
    ) = create_random_completion_split(
        args.tensor_path,
        args.split_path,
        observed_ratio,
        args.val_ratio,
        args.seed,
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

    topology = load_connectivity_tensor(args.topology_path, shape)
    print("Topology shape:", topology.shape)
    print("Topology branch: temporal GCN")

    target_scale = get_target_scale(train_values, args.target_normalization)
    train_targets = train_values / target_scale
    val_targets = val_values / target_scale
    print("Target normalization:", args.target_normalization)
    print("Target scale:", target_scale)
    print("Metrics scale: original target values")

    model = create_gcn_costco(
        shape,
        topology,
        rank=args.rank,
        nc=args.nc,
        node_dim=args.node_dim,
        gcn_dim=args.gcn_dim,
    )
    compile_gcn_costco(model, lr=args.lr)
    model.fit(
        x=transform_indices(train_indices),
        y=train_targets,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            transform_indices(val_indices),
            val_targets,
        ),
        callbacks=[
            k.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    results = {
        "config": {
            "tensor_path": args.tensor_path,
            "topology_path": args.topology_path,
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
            "model": "gcn_costco",
            "fusion": "costco_kpi_branch_plus_temporal_gcn_branch",
            "topology_input": "sat_connectivity_adjacency",
            "topology_shape": [int(v) for v in topology.shape],
            "rank": args.rank,
            "nc": args.nc,
            "node_dim": args.node_dim,
            "gcn_dim": args.gcn_dim,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_normalization": args.target_normalization,
            "target_scale": target_scale,
            "metrics_scale": "original",
            "seed": args.seed,
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
        },
        "train": evaluate_gcn_costco(
            model,
            train_indices,
            train_values,
            target_scale=target_scale,
        ),
        "val": evaluate_gcn_costco(
            model,
            val_indices,
            val_values,
            target_scale=target_scale,
        ),
        "test": evaluate_gcn_costco(
            model,
            test_indices,
            test_values,
            target_scale=target_scale,
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
