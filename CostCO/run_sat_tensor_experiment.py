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
    create_topo_costco,
    evaluate_costco,
    evaluate_topo_costco,
    transform_indices,
)


def load_tensor(path):
    tensor = np.load(path)
    if tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape %s" % (tensor.shape,))
    return tensor.astype("float32")


def load_connectivity_tensor(path, traffic_shape):
    data = np.load(path)
    print("Topology npz keys:", data.files)

    if "arr_0" in data.files:
        topo = data["arr_0"]
    elif "connectivity" in data.files:
        topo = data["connectivity"]
    elif "adjacency" in data.files:
        topo = data["adjacency"]
    else:
        raise ValueError(
            "Cannot find topology tensor in %s, keys=%s" %
            (path, data.files)
        )

    topo = topo.astype("float32")
    if topo.ndim != 3:
        raise ValueError("Expected a 3-D topology tensor, got %s" %
                         (topo.shape,))

    traffic_shape = tuple(int(v) for v in traffic_shape)
    if topo.shape == (traffic_shape[2], traffic_shape[0], traffic_shape[1]):
        pass
    elif topo.shape == traffic_shape:
        topo = np.transpose(topo, (2, 0, 1))
    elif topo.shape[0] == traffic_shape[2]:
        pass
    elif topo.shape[-1] == traffic_shape[2]:
        topo = np.transpose(topo, (2, 0, 1))
    else:
        raise ValueError(
            "Unexpected topology shape %s for traffic shape %s" %
            (topo.shape, traffic_shape)
        )

    if topo.shape[0] != traffic_shape[2]:
        raise ValueError(
            "Topology time length %d does not match traffic time length %d" %
            (topo.shape[0], traffic_shape[2])
        )
    if topo.shape[1] != traffic_shape[0] or topo.shape[2] != traffic_shape[1]:
        raise ValueError(
            "Topology node size %s does not match traffic matrix size %s" %
            (topo.shape[1:], traffic_shape[:2])
        )

    topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
    return (topo > 0).astype("float32")


def compute_shortest_hop_features(topo):
    """Compute all-pairs unweighted hop distances for each time slice."""
    time_len, node_count, _ = topo.shape
    unreachable_hops = float(node_count + 1)
    dist_all = np.full(
        (time_len, node_count, node_count),
        unreachable_hops,
        dtype="float32",
    )
    reachable_all = np.zeros(
        (time_len, node_count, node_count),
        dtype="float32",
    )

    for t in range(time_len):
        adjacency = topo[t] > 0
        for src in range(node_count):
            dist = dist_all[t, src]
            dist[src] = 0.0
            queue = [src]
            head = 0
            while head < len(queue):
                current = queue[head]
                head += 1
                neighbors = np.flatnonzero(adjacency[current])
                next_distance = dist[current] + 1.0
                for neighbor in neighbors:
                    if dist[neighbor] == unreachable_hops:
                        dist[neighbor] = next_distance
                        queue.append(int(neighbor))

        reachable_all[t] = (dist_all[t] < unreachable_hops).astype("float32")

    return dist_all, reachable_all


def build_topology_features(indices, topo, dist_all=None, reachable_all=None):
    src = indices[:, 0]
    dst = indices[:, 1]
    time = indices[:, 2]

    degrees = topo.sum(axis=2)
    two_hop_all = np.matmul(topo, topo)

    direct_edge = topo[time, src, dst]
    deg_src = degrees[time, src]
    deg_dst = degrees[time, dst]
    common_neighbors = np.sum(
        topo[time, src, :] * topo[time, dst, :],
        axis=1,
    )
    two_hop_paths = two_hop_all[time, src, dst]

    feature_list = [
        direct_edge,
        deg_src,
        deg_dst,
        common_neighbors,
        two_hop_paths,
    ]

    if dist_all is not None and reachable_all is not None:
        feature_list.extend([
            dist_all[time, src, dst],
            reachable_all[time, src, dst],
        ])

    return np.stack(feature_list, axis=1).astype("float32")


def normalize_topology_features(train_features, val_features, test_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-6
    return (
        (train_features - mean) / std,
        (val_features - mean) / std,
        (test_features - mean) / std,
        mean.flatten().astype("float32"),
        std.flatten().astype("float32"),
    )


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
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--nc", type=int, default=64)
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
    parser.add_argument(
        "--topology-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument("--use-topology", action="store_true")
    parser.add_argument(
        "--skip-topology-shortest-path",
        action="store_true",
        help="Skip shortest_path_hops and reachable topology features.",
    )
    return parser.parse_args()


def default_output_path(output_dir, observed_ratio, val_ratio, seed, rank, nc,
                        target_normalization, model_name="costco"):
    name = "%s_random_observed%d_val%d_seed%d_rank%d_nc%d_norm_%s.json" % (
        model_name,
        int(round(observed_ratio * 100)),
        int(round(val_ratio * 100)),
        seed,
        rank,
        nc,
        target_normalization,
    )
    return os.path.join(output_dir, name)


def get_target_scale(values, target_normalization):
    if target_normalization == "none":
        return 1.0
    if target_normalization == "max":
        scale = float(np.max(values))
        if scale <= 0.0:
            raise ValueError("Cannot use max normalization with non-positive max")
        return scale
    raise ValueError("Unsupported target normalization: %s" % target_normalization)


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
        model_name = "topo_costco" if args.use_topology else "costco"
        args.metrics_path = default_output_path(
            "results",
            observed_ratio,
            args.val_ratio,
            args.seed,
            args.rank,
            nc,
            args.target_normalization,
            model_name=model_name,
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

    target_scale = get_target_scale(train_values, args.target_normalization)
    train_targets = train_values / target_scale
    val_targets = val_values / target_scale
    print("Target normalization:", args.target_normalization)
    print("Target scale:", target_scale)
    print("Metrics scale: original target values")

    train_topo_features = None
    val_topo_features = None
    test_topo_features = None
    topology_config = {"use_topology": bool(args.use_topology)}

    if args.use_topology:
        topo = load_connectivity_tensor(args.topology_path, shape)
        use_shortest_path = not args.skip_topology_shortest_path
        print("Topology shape:", topo.shape)
        print("Topology shortest path:", use_shortest_path)

        dist_all = None
        reachable_all = None
        if use_shortest_path:
            dist_all, reachable_all = compute_shortest_hop_features(topo)

        train_topo_raw = build_topology_features(
            train_indices,
            topo,
            dist_all=dist_all,
            reachable_all=reachable_all,
        )
        val_topo_raw = build_topology_features(
            val_indices,
            topo,
            dist_all=dist_all,
            reachable_all=reachable_all,
        )
        test_topo_raw = build_topology_features(
            test_indices,
            topo,
            dist_all=dist_all,
            reachable_all=reachable_all,
        )
        (
            train_topo_features,
            val_topo_features,
            test_topo_features,
            topo_feature_mean,
            topo_feature_std,
        ) = normalize_topology_features(
            train_topo_raw,
            val_topo_raw,
            test_topo_raw,
        )
        topology_feature_names = [
            "direct_edge",
            "source_degree",
            "destination_degree",
            "common_neighbors",
            "two_hop_paths",
        ]
        if use_shortest_path:
            topology_feature_names.extend([
                "shortest_path_hops",
                "reachable",
            ])
        topology_config = {
            "use_topology": True,
            "topology_path": args.topology_path,
            "topology_shape": [int(v) for v in topo.shape],
            "topology_features": topology_feature_names,
            "topology_feature_dim": int(train_topo_features.shape[1]),
            "topology_feature_normalization": "train_zscore",
            "topology_feature_mean": topo_feature_mean.tolist(),
            "topology_feature_std": topo_feature_std.tolist(),
            "topology_shortest_path": bool(use_shortest_path),
        }

        model = create_topo_costco(
            shape,
            rank=args.rank,
            nc=nc,
            topo_dim=train_topo_features.shape[1],
        )
        train_x = transform_indices(train_indices) + [train_topo_features]
        val_x = transform_indices(val_indices) + [val_topo_features]
    else:
        model = create_costco(shape, rank=args.rank, nc=nc)
        train_x = transform_indices(train_indices)
        val_x = transform_indices(val_indices)

    compile_costco(model, lr=args.lr)
    model.fit(
        x=train_x,
        y=train_targets,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            val_x,
            val_targets,
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
            "target_normalization": args.target_normalization,
            "target_scale": target_scale,
            "metrics_scale": "original",
            "seed": args.seed,
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
            **topology_config,
        },
    }

    if args.use_topology:
        results["train"] = evaluate_topo_costco(
            model,
            train_indices,
            train_values,
            train_topo_features,
            target_scale=target_scale,
        )
        results["val"] = evaluate_topo_costco(
            model,
            val_indices,
            val_values,
            val_topo_features,
            target_scale=target_scale,
        )
        results["test"] = evaluate_topo_costco(
            model,
            test_indices,
            test_values,
            test_topo_features,
            target_scale=target_scale,
        )
    else:
        results["train"] = evaluate_costco(
            model,
            train_indices,
            train_values,
            target_scale=target_scale,
        )
        results["val"] = evaluate_costco(
            model,
            val_indices,
            val_values,
            target_scale=target_scale,
        )
        results["test"] = evaluate_costco(
            model,
            test_indices,
            test_values,
            target_scale=target_scale,
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
