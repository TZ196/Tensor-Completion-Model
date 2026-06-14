import argparse
import json
import os
from pprint import pprint

import numpy as np
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


NODE_GROUPS = {
    "local": [0, 1, 2, 3],
    "global": [4, 5, 6, 7],
    "dynamic_path": [8, 9],
}

TIME_GROUPS = {
    "local": [0, 1, 2, 3],
    "global": [4, 5, 6, 7],
    "dynamic_path": [8, 9],
}


def select_feature_indices(group, groups):
    if group == "full":
        return None
    if group in groups:
        return groups[group]
    if group == "local_global":
        return groups["local"] + groups["global"]
    raise ValueError("Unsupported structural feature group: %s" % group)


def feature_quality_report(node_features, time_features, node_names, time_names):
    node_flat = node_features.reshape(-1, node_features.shape[-1])
    node_std = node_flat.std(axis=0)
    time_std = time_features.std(axis=0)
    return {
        "node_finite": bool(np.isfinite(node_features).all()),
        "time_finite": bool(np.isfinite(time_features).all()),
        "node_std": node_std.astype("float64").tolist(),
        "time_std": time_std.astype("float64").tolist(),
        "near_constant_node_features": [
            node_names[idx] for idx, value in enumerate(node_std)
            if value < 1e-6
        ],
        "near_constant_time_features": [
            time_names[idx] for idx, value in enumerate(time_std)
            if value < 1e-6
        ],
    }


def load_mode_struct_features(path, group, seed, shuffle_features=False):
    if group == "none":
        return None, None, [], [], {}
    data = np.load(path, allow_pickle=True)
    node_features = data["node_features"].astype("float32")
    time_features = data["time_features"].astype("float32")
    node_names = data["node_feature_names"].astype(str).tolist()
    time_names = data["time_feature_names"].astype(str).tolist()

    node_idx = select_feature_indices(group, NODE_GROUPS)
    time_idx = select_feature_indices(group, TIME_GROUPS)
    if node_idx is not None:
        node_features = node_features[:, :, node_idx]
        node_names = [node_names[idx] for idx in node_idx]
    if time_idx is not None:
        time_features = time_features[:, time_idx]
        time_names = [time_names[idx] for idx in time_idx]

    if shuffle_features:
        rng = np.random.RandomState(seed)
        flat = node_features.reshape(-1, node_features.shape[-1])
        order = rng.permutation(flat.shape[0])
        node_features = flat[order].reshape(node_features.shape)
        time_order = rng.permutation(time_features.shape[0])
        time_features = time_features[time_order]

    quality = feature_quality_report(
        node_features,
        time_features,
        node_names,
        time_names,
    )
    if not quality["node_finite"] or not quality["time_finite"]:
        raise ValueError("Structural features contain NaN or Inf values")
    return node_features, time_features, node_names, time_names, quality


def load_mode_text_embeddings(text_dir):
    source_path = os.path.join(text_dir, "source_text_embeddings.npy")
    destination_path = os.path.join(text_dir, "destination_text_embeddings.npy")
    time_path = os.path.join(text_dir, "time_text_embeddings.npy")
    source_numeric_path = os.path.join(
        text_dir,
        "source_text_numeric_features.npy",
    )
    destination_numeric_path = os.path.join(
        text_dir,
        "destination_text_numeric_features.npy",
    )
    time_numeric_path = os.path.join(text_dir, "time_text_numeric_features.npy")
    metadata_path = os.path.join(text_dir, "text_embedding_metadata.json")
    source = np.load(source_path).astype("float32")
    destination = np.load(destination_path).astype("float32")
    time = np.load(time_path).astype("float32")
    if source.ndim != 3:
        raise ValueError("source text embeddings must have shape [time,node,dim]")
    if destination.shape != source.shape:
        raise ValueError("source/destination text embeddings must share shape")
    if time.ndim != 2 or time.shape[0] != source.shape[0]:
        raise ValueError("time text embeddings must have shape [time,dim]")
    if time.shape[1] != source.shape[2]:
        raise ValueError("source/destination/time text embedding dims differ")
    source_numeric = None
    destination_numeric = None
    time_numeric = None
    if (
        os.path.exists(source_numeric_path) or
        os.path.exists(destination_numeric_path) or
        os.path.exists(time_numeric_path)
    ):
        if not (
            os.path.exists(source_numeric_path) and
            os.path.exists(destination_numeric_path) and
            os.path.exists(time_numeric_path)
        ):
            raise ValueError(
                "source, destination, and time text numeric feature files "
                "must be provided together"
            )
        source_numeric = np.load(source_numeric_path).astype("float32")
        destination_numeric = np.load(destination_numeric_path).astype("float32")
        time_numeric = np.load(time_numeric_path).astype("float32")
        if source_numeric.ndim != 3 or source_numeric.shape[:2] != source.shape[:2]:
            raise ValueError(
                "source text numeric features must have shape [time,node,dim]"
            )
        if (
            destination_numeric.ndim != 3 or
            destination_numeric.shape[:2] != source.shape[:2]
        ):
            raise ValueError(
                "destination text numeric features must have shape [time,node,dim]"
            )
        if time_numeric.ndim != 2 or time_numeric.shape[0] != source.shape[0]:
            raise ValueError("time text numeric features must have shape [time,dim]")
        if not (
            np.all(np.isfinite(source_numeric)) and
            np.all(np.isfinite(destination_numeric)) and
            np.all(np.isfinite(time_numeric))
        ):
            raise ValueError("Text numeric features contain NaN or Inf")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    target_start = int(metadata.get("target_start", 0))
    return (
        source,
        destination,
        time,
        source_numeric,
        destination_numeric,
        time_numeric,
        metadata,
        target_start,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GCN-CoSTCo tensor completion with adjacency fusion."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_mb_tensor.npy")
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
    parser.add_argument("--mode-struct-path", default="mode_struct_features.npz")
    parser.add_argument(
        "--struct-feature-group",
        choices=[
            "none",
            "local",
            "global",
            "dynamic_path",
            "local_global",
            "full",
        ],
        default="none",
    )
    parser.add_argument("--shuffle-struct-features", action="store_true")
    parser.add_argument("--time-conditioned-modes", action="store_true")
    parser.add_argument("--structural-hidden-dim", type=int, default=64)
    parser.add_argument("--structural-align-dim", type=int, default=64)
    parser.add_argument("--structural-beta", type=float, default=0.1)
    parser.add_argument("--structural-alpha", type=float, default=0.1)
    parser.add_argument("--source-align-weight", type=float, default=0.0)
    parser.add_argument("--destination-align-weight", type=float, default=0.0)
    parser.add_argument("--time-align-weight", type=float, default=0.0)
    parser.add_argument("--alignment-temperature", type=float, default=0.2)
    parser.add_argument("--temporal-delta", type=int, default=2)
    parser.add_argument("--use-mode-text", action="store_true")
    parser.add_argument("--mode-text-dir", default="mode_text_data")
    parser.add_argument(
        "--text-fusion-mode",
        choices=["concat", "dual"],
        default="concat",
    )
    parser.add_argument("--text-hidden-dim", type=int, default=64)
    parser.add_argument("--text-align-dim", type=int, default=64)
    parser.add_argument("--text-alpha", type=float, default=0.1)
    parser.add_argument("--source-text-align-weight", type=float, default=0.0)
    parser.add_argument("--destination-text-align-weight", type=float, default=0.0)
    parser.add_argument("--time-text-align-weight", type=float, default=0.0)
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
    (
        node_struct_features,
        time_struct_features,
        node_feature_names,
        time_feature_names,
        struct_feature_quality,
    ) = load_mode_struct_features(
        args.mode_struct_path,
        args.struct_feature_group,
        args.seed,
        shuffle_features=args.shuffle_struct_features,
    )
    if node_struct_features is not None:
        print("Mode structural node features:", node_struct_features.shape)
        print("Mode structural time features:", time_struct_features.shape)
        print("Node structural feature names:", node_feature_names)
        print("Time structural feature names:", time_feature_names)
        print("Node structural feature std:", struct_feature_quality["node_std"])
        print("Time structural feature std:", struct_feature_quality["time_std"])
        if struct_feature_quality["near_constant_node_features"]:
            print(
                "Near-constant node structural features:",
                struct_feature_quality["near_constant_node_features"],
            )
        if struct_feature_quality["near_constant_time_features"]:
            print(
                "Near-constant time structural features:",
                struct_feature_quality["near_constant_time_features"],
            )
    else:
        print("Mode structural features: disabled")

    source_text_embeddings = None
    destination_text_embeddings = None
    time_text_embeddings = None
    source_text_numeric_features = None
    destination_text_numeric_features = None
    time_text_numeric_features = None
    text_metadata = {}
    text_target_start = 0
    if args.use_mode_text:
        (
            source_text_embeddings,
            destination_text_embeddings,
            time_text_embeddings,
            source_text_numeric_features,
            destination_text_numeric_features,
            time_text_numeric_features,
            text_metadata,
            text_target_start,
        ) = load_mode_text_embeddings(args.mode_text_dir)
        print("Mode text source embeddings:", source_text_embeddings.shape)
        print("Mode text destination embeddings:", destination_text_embeddings.shape)
        print("Mode text time embeddings:", time_text_embeddings.shape)
        if source_text_numeric_features is not None:
            print(
                "Mode text source numeric features:",
                source_text_numeric_features.shape,
            )
            print(
                "Mode text destination numeric features:",
                destination_text_numeric_features.shape,
            )
            print(
                "Mode text time numeric features:",
                time_text_numeric_features.shape,
            )
        print("Mode text target start:", text_target_start)
    else:
        print("Mode text embeddings: disabled")

    target_scale = get_target_scale(train_values, args.target_normalization)
    train_targets = train_values / target_scale
    val_targets = val_values / target_scale
    output_bias_init = float(np.mean(train_targets))
    print("Target normalization:", args.target_normalization)
    print("Target scale:", target_scale)
    print("Output bias init:", output_bias_init)
    print("Metrics scale: original target values")

    model = create_gcn_costco(
        shape,
        topology,
        rank=args.rank,
        nc=args.nc,
        node_dim=args.node_dim,
        gcn_dim=args.gcn_dim,
        node_struct_features=node_struct_features,
        time_struct_features=time_struct_features,
        structural_hidden_dim=args.structural_hidden_dim,
        structural_align_dim=args.structural_align_dim,
        structural_beta=args.structural_beta,
        structural_alpha=args.structural_alpha,
        source_align_weight=args.source_align_weight,
        destination_align_weight=args.destination_align_weight,
        time_align_weight=args.time_align_weight,
        alignment_temperature=args.alignment_temperature,
        temporal_delta=args.temporal_delta,
        time_conditioned_modes=args.time_conditioned_modes,
        source_text_embeddings=source_text_embeddings,
        destination_text_embeddings=destination_text_embeddings,
        time_text_embeddings=time_text_embeddings,
        source_text_numeric_features=source_text_numeric_features,
        destination_text_numeric_features=destination_text_numeric_features,
        time_text_numeric_features=time_text_numeric_features,
        text_fusion_mode=args.text_fusion_mode,
        text_hidden_dim=args.text_hidden_dim,
        text_align_dim=args.text_align_dim,
        text_alpha=args.text_alpha,
        source_text_align_weight=args.source_text_align_weight,
        destination_text_align_weight=args.destination_text_align_weight,
        time_text_align_weight=args.time_text_align_weight,
        text_target_start=text_target_start,
        output_bias_init=output_bias_init,
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
            "mode_struct_path": args.mode_struct_path,
            "struct_feature_group": args.struct_feature_group,
            "shuffle_struct_features": args.shuffle_struct_features,
            "time_conditioned_modes": args.time_conditioned_modes,
            "node_struct_feature_names": node_feature_names,
            "time_struct_feature_names": time_feature_names,
            "struct_feature_quality": struct_feature_quality,
            "structural_hidden_dim": args.structural_hidden_dim,
            "structural_align_dim": args.structural_align_dim,
            "structural_beta": args.structural_beta,
            "structural_alpha": args.structural_alpha,
            "source_align_weight": args.source_align_weight,
            "destination_align_weight": args.destination_align_weight,
            "time_align_weight": args.time_align_weight,
            "alignment_temperature": args.alignment_temperature,
            "temporal_delta": args.temporal_delta,
            "use_mode_text": args.use_mode_text,
            "mode_text_dir": args.mode_text_dir,
            "text_fusion_mode": args.text_fusion_mode,
            "text_embedding_metadata": text_metadata,
            "use_text_numeric_features": source_text_numeric_features is not None,
            "source_text_numeric_shape": (
                None if source_text_numeric_features is None else
                [int(v) for v in source_text_numeric_features.shape]
            ),
            "destination_text_numeric_shape": (
                None if destination_text_numeric_features is None else
                [int(v) for v in destination_text_numeric_features.shape]
            ),
            "time_text_numeric_shape": (
                None if time_text_numeric_features is None else
                [int(v) for v in time_text_numeric_features.shape]
            ),
            "text_hidden_dim": args.text_hidden_dim,
            "text_align_dim": args.text_align_dim,
            "text_alpha": args.text_alpha,
            "source_text_align_weight": args.source_text_align_weight,
            "destination_text_align_weight": args.destination_text_align_weight,
            "time_text_align_weight": args.time_text_align_weight,
            "rank": args.rank,
            "nc": args.nc,
            "node_dim": args.node_dim,
            "gcn_dim": args.gcn_dim,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_normalization": args.target_normalization,
            "target_scale": target_scale,
            "output_bias_init": output_bias_init,
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
