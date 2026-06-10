import argparse
import json
import os
import sys
from pprint import pprint

import numpy as np
from tensorflow import keras as k

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(SCRIPT_DIR)
MULTIMODAL_DIR = os.path.dirname(MODELS_DIR)
PROJECT_DIR = os.path.dirname(MULTIMODAL_DIR)
SHARED_DIR = os.path.join(MULTIMODAL_DIR, "shared")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from experiment_utils import (  # noqa: E402
    apply_text_ablation,
    default_split_path,
    format_lr,
    load_embeddings,
    load_existing_split,
    resolve_path,
    stage_flags,
)
from gcn_text_costco_model import (  # noqa: E402
    compile_mindtext_gcn_costco,
    create_mindtext_gcn_costco,
    evaluate_mindtext_gcn_costco,
)
from run_sat_tensor_experiment import (  # noqa: E402
    configure_tensorflow,
    create_random_completion_split,
    get_target_scale,
    load_connectivity_tensor,
    transform_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run staged MindText-GCN-CoSTCo tensor completion."
    )
    parser.add_argument(
        "--tensor-path",
        default=os.path.join(PROJECT_DIR, "sat_path_bytes_mb_tensor.npy"),
    )
    parser.add_argument(
        "--topology-path",
        default=os.path.join(
            PROJECT_DIR, "sat_connectivity_tensor_dynamic_60s_1000ms.npz"
        ),
    )
    parser.add_argument(
        "--endo-embedding-path",
        default=os.path.join(
            MULTIMODAL_DIR, "text_data", "endo_text_embeddings.npy"
        ),
    )
    parser.add_argument(
        "--exo-embedding-path",
        default=os.path.join(
            MULTIMODAL_DIR, "text_data", "exo_text_embeddings.npy"
        ),
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--gcn-dim", type=int, default=128)
    parser.add_argument("--text-projection-dim", type=int, default=128)
    parser.add_argument(
        "--text-stage",
        choices=[
            "concat",
            "global_context_concat",
            "global_context_condenser",
            "cross_attention",
            "semantic_gating",
            "segment_condenser",
        ],
        default="global_context_concat",
    )
    parser.add_argument(
        "--text-ablation",
        choices=[
            "real",
            "endo_only",
            "exo_only",
            "shuffle_endo",
            "zero",
            "random",
        ],
        default="real",
    )
    parser.add_argument("--alignment-projection-dim", type=int, default=128)
    parser.add_argument("--alignment-temperature", type=float, default=0.2)
    parser.add_argument("--temporal-delta", type=int, default=2)
    parser.add_argument("--flow-text-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-text-loss-weight", type=float, default=0.0)
    parser.add_argument("--condenser-alpha", type=float, default=0.5)
    parser.add_argument("--condenser-epsilon", type=float, default=0.05)
    parser.add_argument("--condenser-loss-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
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


def default_metrics_path(args, observed_ratio):
    name = (
        "mindtext_gcn_costco_%s_%s_observed%d_seed%d_rank%d_nc%d_"
        "node%d_gcn%d_text%d_lr%s_epoch%d.json"
    ) % (
        args.text_stage,
        args.text_ablation,
        int(round(observed_ratio * 100)),
        args.seed,
        args.rank,
        args.nc,
        args.node_dim,
        args.gcn_dim,
        args.text_projection_dim,
        format_lr(args.lr),
        args.epochs,
    )
    return os.path.join("results", name)


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
        args.split_path = default_split_path(
            observed_ratio, args.val_ratio, args.seed, project_dir=PROJECT_DIR
        )
    args.tensor_path = resolve_path(args.tensor_path, SCRIPT_DIR)
    args.topology_path = resolve_path(args.topology_path, SCRIPT_DIR)
    args.split_path = resolve_path(args.split_path, SCRIPT_DIR)
    args.endo_embedding_path = resolve_path(
        args.endo_embedding_path, SCRIPT_DIR
    )
    args.exo_embedding_path = resolve_path(
        args.exo_embedding_path, SCRIPT_DIR
    )
    if args.metrics_path is None:
        args.metrics_path = default_metrics_path(args, observed_ratio)

    configure_tensorflow(cpu_only=args.cpu_only, seed=args.seed)
    if args.create_split:
        split = create_random_completion_split(
            args.tensor_path,
            args.split_path,
            observed_ratio,
            args.val_ratio,
            args.seed,
        )
    else:
        split = load_existing_split(args.split_path)
    (
        shape,
        train_indices,
        train_values,
        val_indices,
        val_values,
        test_indices,
        test_values,
        data_stats,
    ) = split

    topology = load_connectivity_tensor(args.topology_path, shape)
    endo_embeddings = load_embeddings(args.endo_embedding_path)
    exo_embeddings = load_embeddings(args.exo_embedding_path)
    endo_embeddings, exo_embeddings = apply_text_ablation(
        endo_embeddings,
        exo_embeddings,
        args.text_ablation,
        args.seed,
    )

    target_scale = get_target_scale(train_values, args.target_normalization)
    train_targets = train_values / target_scale
    val_targets = val_values / target_scale
    output_bias_init = float(np.mean(train_targets))

    print("Tensor shape:", shape.tolist())
    print("Topology shape:", topology.shape)
    print("Endo embeddings:", endo_embeddings.shape)
    print("Exo embeddings:", exo_embeddings.shape)
    print("Text stage:", args.text_stage)
    print("Text ablation:", args.text_ablation)
    print("Split path:", args.split_path)
    print("Metrics path:", args.metrics_path)
    print("Target scale:", target_scale)
    print("Output bias init:", output_bias_init)

    model = create_mindtext_gcn_costco(
        shape,
        topology,
        endo_embeddings,
        exo_embeddings,
        rank=args.rank,
        nc=args.nc,
        node_dim=args.node_dim,
        gcn_dim=args.gcn_dim,
        text_projection_dim=args.text_projection_dim,
        text_stage=args.text_stage,
        alignment_projection_dim=args.alignment_projection_dim,
        alignment_temperature=args.alignment_temperature,
        temporal_delta=args.temporal_delta,
        flow_text_loss_weight=args.flow_text_loss_weight,
        graph_text_loss_weight=args.graph_text_loss_weight,
        condenser_alpha=args.condenser_alpha,
        condenser_epsilon=args.condenser_epsilon,
        condenser_loss_weight=args.condenser_loss_weight,
        output_bias_init=output_bias_init,
    )
    compile_mindtext_gcn_costco(model, lr=args.lr)
    model.fit(
        x=transform_indices(train_indices),
        y=train_targets,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(transform_indices(val_indices), val_targets),
        callbacks=[
            k.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    flags = stage_flags(
        args.text_stage,
        args.flow_text_loss_weight,
        args.graph_text_loss_weight,
    )
    is_global_context = args.text_stage in [
        "concat",
        "global_context_concat",
        "global_context_condenser",
    ]
    model_family = (
        "MindText-GCN-CoSTCo-GlobalContext"
        if is_global_context else "MindText-GCN-CoSTCo"
    )
    fusion_name = (
        "costco_flow_plus_gcn_topology_plus_time_endo_text_and_global_exo_text"
        if is_global_context else
        "costco_flow_plus_gcn_topology_plus_endo_exo_text"
    )
    results = {
        "config": {
            "model_family": model_family,
            "fusion": fusion_name,
            "tensor_path": args.tensor_path,
            "topology_path": args.topology_path,
            "endo_embedding_path": args.endo_embedding_path,
            "exo_embedding_path": args.exo_embedding_path,
            "text_enabled": True,
            "text_mode": "endo_exo_combined",
            "text_stage": args.text_stage,
            "text_ablation": args.text_ablation,
            "text_projection_dim": args.text_projection_dim,
            "endo_embedding_shape": list(endo_embeddings.shape),
            "exo_embedding_shape": list(exo_embeddings.shape),
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
            "alignment_projection_dim": args.alignment_projection_dim,
            "alignment_temperature": args.alignment_temperature,
            "temporal_delta": args.temporal_delta,
            "flow_text_loss_weight": args.flow_text_loss_weight,
            "graph_text_loss_weight": args.graph_text_loss_weight,
            "condenser_alpha": args.condenser_alpha,
            "condenser_epsilon": args.condenser_epsilon,
            "condenser_loss_weight": args.condenser_loss_weight,
            "nmae": "sum(abs(y_true - y_pred)) / sum(abs(y_true))",
            "nrmse": "sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))",
            **flags,
        },
        "train": evaluate_mindtext_gcn_costco(
            model, train_indices, train_values, target_scale=target_scale
        ),
        "val": evaluate_mindtext_gcn_costco(
            model, val_indices, val_values, target_scale=target_scale
        ),
        "test": evaluate_mindtext_gcn_costco(
            model, test_indices, test_values, target_scale=target_scale
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
