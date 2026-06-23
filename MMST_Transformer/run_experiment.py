import argparse
import json
import os
import random
from pprint import pprint

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras as k

from model import create_gt_mst_model


def configure_tensorflow(seed=3, cpu_only=False, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:
            print("Could not enable TensorFlow op determinism:", exc)
        for setter in (
            tf.config.threading.set_inter_op_parallelism_threads,
            tf.config.threading.set_intra_op_parallelism_threads,
        ):
            try:
                setter(1)
            except RuntimeError as exc:
                print("Could not set TensorFlow thread determinism:", exc)

    gpus = tf.config.list_physical_devices("GPU")
    if cpu_only:
        tf.config.set_visible_devices([], "GPU")
        return []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("Could not set GPU memory growth:", exc)
    return gpus


def load_tensor(path):
    tensor = np.load(path)
    if tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape %s" % (tensor.shape,))
    return tensor.astype("float32")


def load_connectivity_tensor(path, traffic_shape):
    data = np.load(path)
    if "arr_0" in data.files:
        topo = data["arr_0"]
    elif "sat_connectivity" in data.files:
        topo = data["sat_connectivity"]
    elif "connectivity" in data.files:
        topo = data["connectivity"]
    elif "adjacency" in data.files:
        topo = data["adjacency"]
    else:
        raise ValueError("Cannot find topology tensor in %s, keys=%s" % (path, data.files))

    topo = topo.astype("float32")
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
        raise ValueError("Unexpected topology shape %s for traffic shape %s" % (topo.shape, traffic_shape))

    expected = (traffic_shape[2], traffic_shape[0], traffic_shape[1])
    if topo.shape != expected:
        raise ValueError("Topology shape %s does not match expected %s" % (topo.shape, expected))
    topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
    return (topo > 0).astype("float32")


def nonzero_finite_entries(tensor):
    finite = np.isfinite(tensor)
    mask = finite & (tensor != 0)
    indices = np.argwhere(mask).astype("int32")
    values = tensor[mask].astype("float32")
    stats = {
        "total_entries": int(tensor.size),
        "finite_entries": int(np.sum(finite)),
        "nonzero_finite_entries": int(np.sum(mask)),
        "zero_finite_entries": int(np.sum(finite & (tensor == 0))),
        "nonfinite_entries": int(tensor.size - np.sum(finite)),
    }
    return indices, values, stats


def create_random_completion_split(tensor_path, split_path, observed_ratio, val_ratio, seed):
    if os.path.exists(split_path):
        data = np.load(split_path)
        return (
            data["shape"],
            data["train_indices"],
            data["train_values"],
            data["val_indices"],
            data["val_values"],
            data["test_indices"],
            data["test_values"],
            {
                "total_entries": int(data["total_entries"]) if "total_entries" in data else None,
                "finite_entries": int(data["finite_entries"]) if "finite_entries" in data else None,
                "nonzero_finite_entries": int(data["nonzero_finite_entries"]) if "nonzero_finite_entries" in data else None,
                "zero_finite_entries": int(data["zero_finite_entries"]) if "zero_finite_entries" in data else None,
                "nonfinite_entries": int(data["nonfinite_entries"]) if "nonfinite_entries" in data else None,
            },
        )

    tensor = load_tensor(tensor_path)
    if not 0.0 < observed_ratio < 1.0:
        raise ValueError("--observed-ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    indices, values, stats = nonzero_finite_entries(tensor)
    rng = np.random.RandomState(seed)
    order = rng.permutation(indices.shape[0])
    train_size = int(round(indices.shape[0] * observed_ratio))
    train_size = max(1, min(train_size, indices.shape[0] - 2))
    remaining = indices.shape[0] - train_size
    val_size = int(round(remaining * val_ratio))
    val_size = max(1, min(val_size, remaining - 1))

    train_order = order[:train_size]
    val_order = order[train_size:train_size + val_size]
    test_order = order[train_size + val_size:]
    payload = {
        "shape": np.array(tensor.shape).astype("int32"),
        "train_indices": indices[train_order],
        "train_values": values[train_order],
        "val_indices": indices[val_order],
        "val_values": values[val_order],
        "test_indices": indices[test_order],
        "test_values": values[test_order],
        "observed_ratio": np.array(observed_ratio).astype("float32"),
        "missing_rate": np.array(1.0 - observed_ratio).astype("float32"),
        "val_ratio": np.array(val_ratio).astype("float32"),
        "seed": np.array(seed).astype("int32"),
        "total_entries": np.array(stats["total_entries"]).astype("int64"),
        "finite_entries": np.array(stats["finite_entries"]).astype("int64"),
        "nonzero_finite_entries": np.array(stats["nonzero_finite_entries"]).astype("int64"),
        "zero_finite_entries": np.array(stats["zero_finite_entries"]).astype("int64"),
        "nonfinite_entries": np.array(stats["nonfinite_entries"]).astype("int64"),
    }
    split_dir = os.path.dirname(split_path)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
    np.savez(split_path, **payload)
    return (
        payload["shape"],
        payload["train_indices"],
        payload["train_values"],
        payload["val_indices"],
        payload["val_values"],
        payload["test_indices"],
        payload["test_values"],
        stats,
    )


def get_target_scale(values, target_normalization):
    if target_normalization == "none":
        return 1.0
    if target_normalization == "max":
        scale = float(np.max(np.abs(values)))
        if scale <= 0.0:
            raise ValueError("Cannot use max normalization with zero max target")
        return scale
    raise ValueError("Unknown target normalization: %s" % target_normalization)


def transform_indices(indices):
    return [indices[:, i] for i in range(indices.shape[1])]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def nmae(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))
    if denominator == 0.0:
        raise ValueError("Cannot compute NMAE: sum(abs(y_true)) is 0")
    return np.sum(np.abs(y_true - y_pred)) / denominator


def nrmse(y_true, y_pred):
    denominator = np.sum(np.square(y_true))
    if denominator == 0.0:
        raise ValueError("Cannot compute NRMSE: sum(square(y_true)) is 0")
    return np.sqrt(np.sum(np.square(y_true - y_pred)) / denominator)


def evaluate_model(model, indices, values, batch_size=1024, target_scale=1.0, verbose=0):
    pred = model.predict(transform_indices(indices), batch_size=batch_size, verbose=verbose).flatten()
    pred = np.maximum(pred * target_scale, 0.0)
    return {
        "rmse": float(rmse(values, pred)),
        "mae": float(mae(values, pred)),
        "nmae": float(nmae(values, pred)),
        "nrmse": float(nrmse(values, pred)),
        "y_true_min": float(np.min(values)),
        "y_true_max": float(np.max(values)),
        "y_true_mean": float(np.mean(values)),
        "y_pred_min": float(np.min(pred)),
        "y_pred_max": float(np.max(pred)),
        "y_pred_mean": float(np.mean(pred)),
    }


def load_mode_text_data(text_dir):
    def load(name):
        return np.load(os.path.join(text_dir, name)).astype("float32")

    source = load("source_text_embeddings.npy")
    destination = load("destination_text_embeddings.npy")
    time = load("time_text_embeddings.npy")
    source_numeric_path = os.path.join(text_dir, "source_text_numeric_features.npy")
    destination_numeric_path = os.path.join(text_dir, "destination_text_numeric_features.npy")
    time_numeric_path = os.path.join(text_dir, "time_text_numeric_features.npy")
    source_numeric = destination_numeric = time_numeric = None
    if os.path.exists(source_numeric_path) or os.path.exists(destination_numeric_path) or os.path.exists(time_numeric_path):
        if not (os.path.exists(source_numeric_path) and os.path.exists(destination_numeric_path) and os.path.exists(time_numeric_path)):
            raise ValueError("source/destination/time numeric feature files must be provided together")
        source_numeric = np.load(source_numeric_path).astype("float32")
        destination_numeric = np.load(destination_numeric_path).astype("float32")
        time_numeric = np.load(time_numeric_path).astype("float32")
    metadata_path = os.path.join(text_dir, "text_embedding_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    target_start = int(metadata.get("target_start", 0))
    return source, destination, time, source_numeric, destination_numeric, time_numeric, metadata, target_start


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
            self.text_layer = self.model.get_layer("text_injection")
        except ValueError:
            self.text_layer = None

    def on_epoch_end(self, epoch, logs=None):
        if self.target_ratio <= 0.0 or self.text_layer is None:
            return
        logs = logs or {}
        raw = {
            "source": logs.get("source_text_align_loss"),
            "destination": logs.get("destination_text_align_loss"),
            "time": logs.get("time_text_align_loss"),
        }
        active = [(name, float(value)) for name, value in raw.items() if value is not None and float(value) > 0.0]
        if not active or "loss" not in logs:
            return
        weighted_total = sum(
            float(logs.get(name, 0.0))
            for name in (
                "source_text_align_weighted_loss",
                "destination_text_align_weighted_loss",
                "time_text_align_weighted_loss",
            )
        )
        prediction_loss = max(float(logs["loss"]) - weighted_total, self.epsilon)
        target_each = prediction_loss * self.target_ratio / float(len(active))
        mapping = {
            "source": self.text_layer.source_text_align_weight_var,
            "destination": self.text_layer.destination_text_align_weight_var,
            "time": self.text_layer.time_text_align_weight_var,
        }
        for name, raw_loss in active:
            mapping[name].assign(target_each / (raw_loss + self.epsilon))


class ConciseTrainingLogger(k.callbacks.Callback):
    def __init__(self, log_every=10):
        super().__init__()
        self.log_every = max(1, int(log_every))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        total_epochs = self.params.get("epochs")
        if not (epoch_num == 1 or epoch_num % self.log_every == 0 or epoch_num == total_epochs):
            return
        weighted_align = sum(
            float(logs.get(name, 0.0))
            for name in (
                "source_text_align_weighted_loss",
                "destination_text_align_weighted_loss",
                "time_text_align_weighted_loss",
            )
        )
        parts = [
            "epoch %d/%s" % (epoch_num, total_epochs),
            "loss=%.6f" % float(logs.get("loss", 0.0)),
            "mae=%.6f" % float(logs.get("mae", 0.0)),
        ]
        if "val_loss" in logs:
            parts.append("val_loss=%.6f" % float(logs["val_loss"]))
        if "val_mae" in logs:
            parts.append("val_mae=%.6f" % float(logs["val_mae"]))
        if weighted_align > 0.0:
            parts.append("text_align=%.6f" % weighted_align)
        print(" - ".join(parts))


def variant_config(variant):
    mapping = {
        "M0": dict(use_transformer=False, use_graph_token=False, use_mode_text=False, text_mode="text_only", align=False),
        "M1": dict(use_transformer=True, use_graph_token=False, use_mode_text=False, text_mode="text_only", align=False),
        "M2": dict(use_transformer=True, use_graph_token=True, use_mode_text=False, text_mode="text_only", align=False),
        "M3": dict(use_transformer=True, use_graph_token=True, use_mode_text=True, text_mode="text_only", align=False),
        "M4": dict(use_transformer=True, use_graph_token=True, use_mode_text=True, text_mode="numeric_only", align=False),
        "M5": dict(use_transformer=True, use_graph_token=True, use_mode_text=True, text_mode="text_numeric", align=False),
        "M6": dict(use_transformer=True, use_graph_token=True, use_mode_text=True, text_mode="text_numeric", align=True),
    }
    if variant not in mapping:
        raise ValueError("Unsupported variant %s" % variant)
    return mapping[variant]


def serializable_history(history):
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Run GT-MST tensor completion.")
    parser.add_argument("--tensor-path", default="../CostCO/sat_path_bytes_mb_tensor.npy")
    parser.add_argument("--topology-path", default="../CostCO/sat_connectivity_tensor_dynamic_60s_1000ms.npz")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--observed-ratio", type=float, default=0.07)
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4", "M5", "M6"], default="M2")
    parser.add_argument("--mode-text-dir", default="../CostCO/mode_text_numeric_ablation_data/both")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--gcn-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-hidden-dim", type=int, default=128)
    parser.add_argument("--text-alpha", type=float, default=0.02)
    parser.add_argument("--text-align-dim", type=int, default=64)
    parser.add_argument("--text-align-target-ratio", type=float, default=0.01)
    parser.add_argument("--alignment-temperature", type=float, default=0.2)
    parser.add_argument("--temporal-delta", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-normalization", choices=["max", "none"], default="max")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--fit-shuffle", action="store_true")
    parser.add_argument("--fit-verbose", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--history-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = variant_config(args.variant)
    observed_percent = int(round(args.observed_ratio * 100))
    if args.split_path is None:
        args.split_path = os.path.join("splits", "random_observed%d_val10_seed_%d.npz" % (observed_percent, args.seed))
    if args.metrics_path is None:
        args.metrics_path = os.path.join(
            "results",
            "vis%d_%s_gt_mst_seed%d.json" % (observed_percent, args.variant, args.seed),
        )
    if args.checkpoint_path is None:
        stem = os.path.splitext(os.path.basename(args.metrics_path))[0]
        args.checkpoint_path = os.path.join("checkpoints", stem + ".best.weights.h5")
    if args.history_path is None:
        stem = os.path.splitext(os.path.basename(args.metrics_path))[0]
        args.history_path = os.path.join("histories", stem + ".history.json")

    alignment_enabled = bool(cfg["align"] and args.text_align_target_ratio > 0.0)
    patience = args.early_stopping_patience
    if patience is None:
        patience = 20 if alignment_enabled else 10

    configure_tensorflow(seed=args.seed, cpu_only=args.cpu_only)
    (
        shape,
        train_indices,
        train_values,
        val_indices,
        val_values,
        test_indices,
        test_values,
        stats,
    ) = create_random_completion_split(
        args.tensor_path,
        args.split_path,
        args.observed_ratio,
        args.val_ratio,
        args.seed,
    )
    topology = load_connectivity_tensor(args.topology_path, shape)

    text_payload = (None, None, None, None, None, None, {}, 0)
    if cfg["use_mode_text"]:
        text_payload = load_mode_text_data(args.mode_text_dir)

    target_scale = get_target_scale(train_values, args.target_normalization)
    train_targets = train_values / target_scale
    val_targets = val_values / target_scale
    output_bias_init = float(np.mean(train_targets))

    configure_tensorflow(seed=args.seed, cpu_only=args.cpu_only)
    (
        source_text,
        destination_text,
        time_text,
        source_numeric,
        destination_numeric,
        time_numeric,
        text_metadata,
        text_target_start,
    ) = text_payload
    model = create_gt_mst_model(
        shape=shape,
        topology=topology,
        d_model=args.d_model,
        node_dim=args.node_dim,
        gcn_dim=args.gcn_dim,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_graph_token=cfg["use_graph_token"],
        use_transformer=cfg["use_transformer"],
        use_mode_text=cfg["use_mode_text"],
        text_mode=cfg["text_mode"],
        source_text_embeddings=source_text,
        destination_text_embeddings=destination_text,
        time_text_embeddings=time_text,
        source_numeric_features=source_numeric,
        destination_numeric_features=destination_numeric,
        time_numeric_features=time_numeric,
        text_hidden_dim=args.text_hidden_dim,
        text_alpha=args.text_alpha,
        text_align_dim=args.text_align_dim,
        text_align_target_ratio=args.text_align_target_ratio if alignment_enabled else 0.0,
        alignment_temperature=args.alignment_temperature,
        temporal_delta=args.temporal_delta,
        text_target_start=text_target_start,
        output_bias_init=output_bias_init,
    )
    model.compile(k.optimizers.Adam(learning_rate=args.lr), loss="mse", metrics=["mae"])

    callbacks = []
    if alignment_enabled:
        callbacks.append(TextAlignTargetRatioScheduler(args.text_align_target_ratio))
    if args.fit_verbose == 0:
        callbacks.append(ConciseTrainingLogger(log_every=args.log_every))
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
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
                patience=patience,
                mode="min",
                restore_best_weights=True,
                verbose=0,
            )
        )

    print("===== GT-MST Experiment =====")
    print("Variant:", args.variant)
    print("Tensor shape:", shape.tolist())
    print("Observed ratio:", args.observed_ratio)
    print("Split path:", args.split_path)
    print("Metrics path:", args.metrics_path)
    print("Topology shape:", topology.shape)
    print("d_model:", args.d_model)
    print("use_transformer:", cfg["use_transformer"])
    print("use_graph_token:", cfg["use_graph_token"])
    print("use_mode_text:", cfg["use_mode_text"])
    print("text_mode:", cfg["text_mode"])
    print("text_align_target_ratio:", args.text_align_target_ratio if alignment_enabled else 0.0)
    print("Train/val/test entries:", train_indices.shape[0], val_indices.shape[0], test_indices.shape[0])

    history = model.fit(
        transform_indices(train_indices),
        train_targets,
        validation_data=(transform_indices(val_indices), val_targets),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.fit_verbose,
        shuffle=bool(args.fit_shuffle),
        callbacks=callbacks,
    )
    if os.path.exists(args.checkpoint_path):
        model.load_weights(args.checkpoint_path)

    train_metrics = evaluate_model(model, train_indices, train_values, args.batch_size, target_scale)
    val_metrics = evaluate_model(model, val_indices, val_values, args.batch_size, target_scale)
    test_metrics = evaluate_model(model, test_indices, test_values, args.batch_size, target_scale)

    payload = {
        "config": vars(args),
        "variant_config": cfg,
        "data_stats": stats,
        "shape": [int(value) for value in shape],
        "target_scale": float(target_scale),
        "text_metadata": text_metadata,
        "checkpoint_path": args.checkpoint_path,
        "history_path": args.history_path,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)
    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history(history), f, indent=2)

    print("Test metrics:")
    pprint(test_metrics)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()

