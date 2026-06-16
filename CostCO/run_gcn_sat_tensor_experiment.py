import argparse
import json
import os
from pprint import pprint

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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


class ConciseTrainingLogger(k.callbacks.Callback):
    def __init__(self, log_every=10):
        super().__init__()
        self.log_every = max(1, int(log_every))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        total_epochs = self.params.get("epochs")
        should_log = (
            epoch_num == 1 or
            epoch_num % self.log_every == 0 or
            epoch_num == total_epochs
        )
        if not should_log:
            return
        text_align_total = 0.0
        for name in (
            "source_text_align_weighted_loss",
            "destination_text_align_weighted_loss",
            "time_text_align_weighted_loss",
        ):
            value = logs.get(name)
            if value is not None:
                text_align_total += float(value)
        align_total = text_align_total
        parts = [
            "epoch %d/%s" % (epoch_num, total_epochs),
            "loss=%.6f" % float(logs.get("loss", 0.0)),
            "mae=%.6f" % float(logs.get("mae", 0.0)),
        ]
        if "val_loss" in logs:
            parts.append("val_loss=%.6f" % float(logs["val_loss"]))
        if "val_mae" in logs:
            parts.append("val_mae=%.6f" % float(logs["val_mae"]))
        if text_align_total > 0.0:
            parts.append("text_align=%.6f" % text_align_total)
        if align_total > 0.0 and "loss" in logs:
            pred_loss = max(float(logs["loss"]) - align_total, 0.0)
            parts.append("pred_loss~=%.6f" % pred_loss)
        print(" - ".join(parts))


class TextAlignTargetRatioScheduler(k.callbacks.Callback):
    def __init__(self, target_ratio, epsilon=1e-8):
        super().__init__()
        self.target_ratio = float(target_ratio)
        self.epsilon = float(epsilon)
        self.text_layer = None

    def on_train_begin(self, logs=None):
        if self.target_ratio <= 0.0:
            return
        try:
            self.text_layer = self.model.get_layer("mode_wise_text_alignment")
        except ValueError:
            self.text_layer = None

    def on_epoch_end(self, epoch, logs=None):
        if self.target_ratio <= 0.0 or self.text_layer is None:
            return
        logs = logs or {}
        text_raw = {
            "source": logs.get("source_text_align_loss"),
            "destination": logs.get("destination_text_align_loss"),
            "time": logs.get("time_text_align_loss"),
        }
        active = [
            (name, float(value)) for name, value in text_raw.items()
            if value is not None and float(value) > 0.0
        ]
        if not active or "loss" not in logs:
            return

        weighted_names = (
            "source_text_align_weighted_loss",
            "destination_text_align_weighted_loss",
            "time_text_align_weighted_loss",
        )
        weighted_total = sum(
            float(logs.get(name, 0.0)) for name in weighted_names
        )
        prediction_loss = max(float(logs["loss"]) - weighted_total, self.epsilon)
        target_total = prediction_loss * self.target_ratio
        target_each = target_total / float(len(active))

        assignments = {
            "source": self.text_layer.source_text_align_weight_var,
            "destination": self.text_layer.destination_text_align_weight_var,
            "time": self.text_layer.time_text_align_weight_var,
        }
        for name, raw_loss in active:
            assignments[name].assign(target_each / (raw_loss + self.epsilon))


def default_artifact_path(metrics_path, subdir, suffix):
    stem = os.path.splitext(os.path.basename(metrics_path))[0]
    return os.path.join(subdir, stem + suffix)


def serializable_history(history):
    payload = {}
    for key, values in history.history.items():
        payload[key] = [float(value) for value in values]
    return payload


def od_path_feature_quality_report(features, feature_names):
    flat = features.reshape(-1, features.shape[-1])
    std = flat.std(axis=0)
    return {
        "finite": bool(np.isfinite(features).all()),
        "feature_std": std.astype("float64").tolist(),
        "near_constant_features": [
            feature_names[idx] for idx, value in enumerate(std)
            if value < 1e-6
        ],
    }


def load_od_path_features(path, traffic_shape):
    data = np.load(path, allow_pickle=True)
    if "od_path_features" not in data:
        raise KeyError(
            "OD path feature file must contain 'od_path_features'. "
            "Available keys: %s" % list(data.keys())
        )
    features = data["od_path_features"].astype("float32")
    if features.ndim != 4:
        raise ValueError("od_path_features must have shape [time, source, destination, dim]")
    expected = (traffic_shape[2], traffic_shape[0], traffic_shape[1])
    if features.shape[:3] != expected:
        raise ValueError(
            "OD path feature prefix shape %s does not match expected %s" %
            (features.shape[:3], expected)
        )
    if "od_path_feature_names" in data:
        feature_names = data["od_path_feature_names"].astype(str).tolist()
    else:
        feature_names = [
            "od_path_feature_%d" % idx for idx in range(features.shape[-1])
        ]
    quality = od_path_feature_quality_report(features, feature_names)
    if not quality["finite"]:
        raise ValueError("OD path features contain NaN or Inf values")
    metadata = {}
    for key in ("normalization_mean", "normalization_std"):
        if key in data:
            metadata[key] = data[key].astype("float64").tolist()
    return features, feature_names, quality, metadata


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
    if source.ndim not in (2, 3):
        raise ValueError(
            "source text embeddings must have shape [node,dim] "
            "or [time,node,dim]"
        )
    if destination.shape != source.shape:
        raise ValueError("source/destination text embeddings must share shape")
    if time.ndim != 2:
        raise ValueError("time text embeddings must have shape [time,dim]")
    if source.ndim == 3 and time.shape[0] != source.shape[0]:
        raise ValueError("time length differs between text embedding files")
    if time.shape[1] != source.shape[-1]:
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
        if (
            source_numeric.ndim != source.ndim or
            source_numeric.shape[:-1] != source.shape[:-1]
        ):
            raise ValueError(
                "source text numeric features must match source text prefix shape"
            )
        if (
            destination_numeric.ndim != source.ndim or
            destination_numeric.shape[:-1] != source.shape[:-1]
        ):
            raise ValueError(
                "destination text numeric features must match destination text prefix shape"
            )
        if time_numeric.ndim != 2 or time_numeric.shape[0] != time.shape[0]:
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
    parser.add_argument("--alignment-temperature", type=float, default=0.2)
    parser.add_argument("--temporal-delta", type=int, default=2)
    parser.add_argument("--use-od-path-features", action="store_true")
    parser.add_argument(
        "--od-path-feature-path",
        default="mode_od_path_features.npz",
    )
    parser.add_argument("--od-path-hidden-dim", type=int, default=64)
    parser.add_argument("--od-path-alpha-init", type=float, default=0.05)
    parser.add_argument("--use-mode-text", action="store_true")
    parser.add_argument("--mode-text-dir", default="mode_text_data")
    parser.add_argument(
        "--text-fusion-mode",
        choices=["concat", "gated_numeric"],
        default="concat",
    )
    parser.add_argument("--numeric-alpha-init", type=float, default=0.02)
    parser.add_argument("--text-hidden-dim", type=int, default=64)
    parser.add_argument("--text-align-dim", type=int, default=64)
    parser.add_argument("--text-alpha", type=float, default=0.02)
    parser.add_argument(
        "--text-align-target-ratio",
        type=float,
        default=0.0,
        help=(
            "Automatically scale total text alignment loss to this fraction "
            "of the prediction loss. 0 disables text alignment."
        ),
    )
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
    parser.add_argument(
        "--fit-shuffle",
        action="store_true",
        help="Enable Keras fit shuffle. Default is disabled for reproducibility.",
    )
    parser.add_argument(
        "--fit-verbose",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Keras fit verbosity. Default 0 uses the concise logger.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Epoch interval for concise logging when --fit-verbose is 0.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Train for the full epoch count without EarlyStopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "EarlyStopping patience. Defaults to 20 when any alignment loss "
            "is enabled, otherwise 10."
        ),
    )
    parser.add_argument(
        "--save-run-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save best weights and training history for reproducible reruns.",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--history-path", default=None)
    parser.add_argument(
        "--load-best-checkpoint-for-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the best validation checkpoint when checkpointing is enabled.",
    )
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    alignment_enabled = args.text_align_target_ratio > 0.0
    early_stopping_patience = args.early_stopping_patience
    if early_stopping_patience is None:
        early_stopping_patience = 20 if alignment_enabled else 10
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
    if args.checkpoint_path is None:
        args.checkpoint_path = default_artifact_path(
            args.metrics_path,
            "checkpoints",
            ".best.weights.h5",
        )
    if args.history_path is None:
        args.history_path = default_artifact_path(
            args.metrics_path,
            "histories",
            ".history.json",
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

    od_path_features = None
    od_path_feature_names = []
    od_path_feature_quality = {}
    od_path_feature_metadata = {}
    if args.use_od_path_features:
        (
            od_path_features,
            od_path_feature_names,
            od_path_feature_quality,
            od_path_feature_metadata,
        ) = load_od_path_features(args.od_path_feature_path, shape)
        print("OD path features:", od_path_features.shape)
        print("OD path feature names:", od_path_feature_names)
        print("OD path feature std:", od_path_feature_quality["feature_std"])
        if od_path_feature_quality["near_constant_features"]:
            print(
                "Near-constant OD path features:",
                od_path_feature_quality["near_constant_features"],
            )
    else:
        print("OD path features: disabled")

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

    configure_tensorflow(cpu_only=args.cpu_only, seed=args.seed)
    model = create_gcn_costco(
        shape,
        topology,
        rank=args.rank,
        nc=args.nc,
        node_dim=args.node_dim,
        gcn_dim=args.gcn_dim,
        alignment_temperature=args.alignment_temperature,
        temporal_delta=args.temporal_delta,
        source_text_embeddings=source_text_embeddings,
        destination_text_embeddings=destination_text_embeddings,
        time_text_embeddings=time_text_embeddings,
        source_text_numeric_features=source_text_numeric_features,
        destination_text_numeric_features=destination_text_numeric_features,
        time_text_numeric_features=time_text_numeric_features,
        text_fusion_mode=args.text_fusion_mode,
        numeric_alpha_init=args.numeric_alpha_init,
        text_hidden_dim=args.text_hidden_dim,
        text_align_dim=args.text_align_dim,
        text_alpha=args.text_alpha,
        text_align_target_ratio=args.text_align_target_ratio,
        text_target_start=text_target_start,
        od_path_features=od_path_features,
        od_path_hidden_dim=args.od_path_hidden_dim,
        od_path_alpha_init=args.od_path_alpha_init,
        output_bias_init=output_bias_init,
    )
    compile_gcn_costco(model, lr=args.lr)
    callbacks = []
    if args.text_align_target_ratio > 0.0:
        callbacks.append(TextAlignTargetRatioScheduler(args.text_align_target_ratio))
    if args.fit_verbose == 0:
        callbacks.append(ConciseTrainingLogger(log_every=args.log_every))
    if args.save_run_artifacts:
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        callbacks.append(
            k.callbacks.ModelCheckpoint(
                filepath=args.checkpoint_path,
                monitor="val_mae",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=0,
            )
        )
    if not args.disable_early_stopping:
        callbacks.append(
            k.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        x=transform_indices(train_indices),
        y=train_targets,
        verbose=args.fit_verbose,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=args.fit_shuffle,
        validation_data=(
            transform_indices(val_indices),
            val_targets,
        ),
        callbacks=callbacks,
    )
    if args.save_run_artifacts:
        history_dir = os.path.dirname(args.history_path)
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir)
        history_payload = {
            "config": {
                "seed": args.seed,
                "fit_shuffle": args.fit_shuffle,
                "disable_early_stopping": args.disable_early_stopping,
                "checkpoint_path": args.checkpoint_path,
            },
            "history": serializable_history(history),
        }
        with open(args.history_path, "w") as f:
            json.dump(history_payload, f, indent=2, sort_keys=True)
    if (
        args.save_run_artifacts and
        args.load_best_checkpoint_for_eval and
        os.path.exists(args.checkpoint_path)
    ):
        model.load_weights(args.checkpoint_path)
        print("Loaded best checkpoint for evaluation:", args.checkpoint_path)

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
            "alignment_temperature": args.alignment_temperature,
            "temporal_delta": args.temporal_delta,
            "use_od_path_features": args.use_od_path_features,
            "od_path_feature_path": args.od_path_feature_path,
            "od_path_feature_shape": (
                None if od_path_features is None else
                [int(v) for v in od_path_features.shape]
            ),
            "od_path_feature_names": od_path_feature_names,
            "od_path_feature_quality": od_path_feature_quality,
            "od_path_feature_metadata": od_path_feature_metadata,
            "od_path_hidden_dim": args.od_path_hidden_dim,
            "od_path_alpha_init": args.od_path_alpha_init,
            "use_mode_text": args.use_mode_text,
            "mode_text_dir": args.mode_text_dir,
            "text_fusion_mode": args.text_fusion_mode,
            "numeric_alpha_init": args.numeric_alpha_init,
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
            "text_align_target_ratio": args.text_align_target_ratio,
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
            "fit_shuffle": args.fit_shuffle,
            "fit_verbose": args.fit_verbose,
            "log_every": args.log_every,
            "disable_early_stopping": args.disable_early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "save_run_artifacts": args.save_run_artifacts,
            "checkpoint_path": args.checkpoint_path,
            "history_path": args.history_path,
            "load_best_checkpoint_for_eval": args.load_best_checkpoint_for_eval,
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
