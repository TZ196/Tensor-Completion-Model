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

from dense_model import DenseIndependentTextGTMST
from run_experiment import (
    configure_tensorflow,
    create_random_completion_split,
    get_target_scale,
    load_connectivity_tensor,
    load_mode_text_data,
    load_tensor,
    mae,
    nmae,
    nrmse,
    rmse,
)


def build_mask(shape, indices):
    mask = np.zeros(shape, dtype="float32")
    mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return mask


def make_time_chunks(time_count, chunk_len):
    chunks = []
    for start in range(0, int(time_count), int(chunk_len)):
        chunks.append(np.arange(start, min(start + int(chunk_len), int(time_count)), dtype="int32"))
    return chunks


def masked_chunk_loss(model, target_tensor, mask_tensor, time_ids, training):
    pred = model(tf.constant(time_ids[None, :], dtype=tf.int32), training=training)[0]
    y_true = tf.gather(target_tensor, time_ids, axis=2)
    mask = tf.gather(mask_tensor, time_ids, axis=2)
    count = tf.reduce_sum(mask)
    sq = tf.square(pred - y_true) * mask
    pred_loss = tf.cond(
        count > 0.0,
        lambda: tf.reduce_sum(sq) / count,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    total_loss = pred_loss + tf.add_n(model.losses) if model.losses else pred_loss
    return total_loss, pred_loss, count


def predict_full_tensor(model, shape, chunks):
    pred = np.zeros(shape, dtype="float32")
    for time_ids in chunks:
        out = model(tf.constant(time_ids[None, :], dtype=tf.int32), training=False).numpy()[0]
        pred[:, :, time_ids] = out
    return pred


def evaluate_predictions(pred_norm, indices, values, target_scale):
    pred = np.maximum(pred_norm[indices[:, 0], indices[:, 1], indices[:, 2]] * target_scale, 0.0)
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


def variant_config(variant):
    mapping = {
        "M7_dense": dict(use_numeric=False, align=False, label="IndependentTextTokens"),
        "M8_dense": dict(use_numeric=True, align=False, label="IndependentTextTokens+NumericControl"),
        "M9_dense": dict(use_numeric=True, align=True, label="IndependentTextTokens+NumericControl+TextAlign"),
    }
    if variant not in mapping:
        raise ValueError("Unsupported dense variant %s" % variant)
    return mapping[variant]


def parse_args():
    parser = argparse.ArgumentParser(description="Run dense time-block GT-MST tensor completion.")
    parser.add_argument("--tensor-path", default="../CostCO/sat_path_bytes_mb_tensor.npy")
    parser.add_argument("--topology-path", default="../CostCO/sat_connectivity_tensor_dynamic_60s_1000ms.npz")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--observed-ratio", type=float, default=0.07)
    parser.add_argument("--variant", choices=["M7_dense", "M8_dense", "M9_dense"], default="M8_dense")
    parser.add_argument("--mode-text-dir", default="../CostCO/mode_text_numeric_ablation_data/both")
    parser.add_argument("--chunk-len", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--gcn-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-hidden-dim", type=int, default=128)
    parser.add_argument("--text-align-weight", type=float, default=1e-4)
    parser.add_argument("--alignment-temperature", type=float, default=0.2)
    parser.add_argument("--max-graph-attention-bias", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--target-normalization", choices=["max", "none"], default="max")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--shuffle-chunks", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
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
            "vis%d_%s_gt_mst_dense_seed%d.json" % (observed_percent, args.variant, args.seed),
        )
    if args.checkpoint_path is None:
        stem = os.path.splitext(os.path.basename(args.metrics_path))[0]
        args.checkpoint_path = os.path.join("checkpoints", stem + ".best.weights.h5")
    if args.history_path is None:
        stem = os.path.splitext(os.path.basename(args.metrics_path))[0]
        args.history_path = os.path.join("histories", stem + ".history.json")

    configure_tensorflow(seed=args.seed, cpu_only=args.cpu_only)
    tensor = load_tensor(args.tensor_path)
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
    shape = [int(value) for value in shape]
    topology = load_connectivity_tensor(args.topology_path, shape)
    (
        source_text,
        destination_text,
        time_text,
        source_numeric,
        destination_numeric,
        time_numeric,
        text_metadata,
        _text_target_start,
    ) = load_mode_text_data(args.mode_text_dir)
    if not cfg["use_numeric"]:
        source_numeric = destination_numeric = time_numeric = None

    target_scale = get_target_scale(train_values, args.target_normalization)
    target_norm = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).astype("float32") / target_scale
    train_mask = build_mask(shape, train_indices)
    val_mask = build_mask(shape, val_indices)
    test_mask = build_mask(shape, test_indices)
    chunks = make_time_chunks(shape[2], args.chunk_len)

    configure_tensorflow(seed=args.seed, cpu_only=args.cpu_only)
    model = DenseIndependentTextGTMST(
        shape=shape,
        topology=topology,
        source_text_embeddings=source_text,
        destination_text_embeddings=destination_text,
        time_text_embeddings=time_text,
        source_numeric_features=source_numeric,
        destination_numeric_features=destination_numeric,
        time_numeric_features=time_numeric,
        variant=args.variant,
        d_model=args.d_model,
        node_dim=args.node_dim,
        gcn_dim=args.gcn_dim,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        text_hidden_dim=args.text_hidden_dim,
        text_align_weight=args.text_align_weight if cfg["align"] else 0.0,
        alignment_temperature=args.alignment_temperature,
        max_graph_attention_bias=args.max_graph_attention_bias,
        output_bias_init=float(np.mean(train_values / target_scale)),
        name="Dense_Independent_Text_GT_MST",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    target_tensor = tf.constant(target_norm, dtype=tf.float32)
    train_mask_tensor = tf.constant(train_mask, dtype=tf.float32)
    val_mask_tensor = tf.constant(val_mask, dtype=tf.float32)

    # Build variables before checkpointing.
    _ = model(tf.constant(chunks[0][None, :], dtype=tf.int32), training=False)
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    rng = random.Random(args.seed)
    best_val_mae = float("inf")
    best_epoch = 0
    wait = 0
    history = {"loss": [], "prediction_loss": [], "val_loss": [], "val_mae": []}

    print("===== Dense GT-MST Experiment =====")
    print("Variant:", args.variant, cfg["label"])
    print("Tensor shape:", shape)
    print("Observed ratio:", args.observed_ratio)
    print("Chunk length:", args.chunk_len)
    print("Chunks:", len(chunks))
    print("Metrics path:", args.metrics_path)
    print("Train/val/test entries:", train_indices.shape[0], val_indices.shape[0], test_indices.shape[0])

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(chunks)))
        if args.shuffle_chunks:
            rng.shuffle(order)
        total_loss = 0.0
        total_pred_loss = 0.0
        used = 0
        for idx in order:
            time_ids = chunks[idx]
            if np.sum(train_mask[:, :, time_ids]) <= 0.0:
                continue
            with tf.GradientTape() as tape:
                loss, pred_loss, _count = masked_chunk_loss(
                    model,
                    target_tensor,
                    train_mask_tensor,
                    time_ids,
                    training=True,
                )
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients((g, v) for g, v in zip(grads, model.trainable_variables) if g is not None)
            total_loss += float(loss.numpy())
            total_pred_loss += float(pred_loss.numpy())
            used += 1
        train_loss = total_loss / max(used, 1)
        train_pred_loss = total_pred_loss / max(used, 1)

        val_losses = []
        for time_ids in chunks:
            if np.sum(val_mask[:, :, time_ids]) <= 0.0:
                continue
            val_loss, _pred_loss, _count = masked_chunk_loss(
                model,
                target_tensor,
                val_mask_tensor,
                time_ids,
                training=False,
            )
            val_losses.append(float(val_loss.numpy()))
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        pred_norm = predict_full_tensor(model, shape, chunks)
        val_metrics = evaluate_predictions(pred_norm, val_indices, val_values, target_scale)
        val_mae = val_metrics["mae"]

        history["loss"].append(train_loss)
        history["prediction_loss"].append(train_pred_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                "epoch %d/%d - loss=%.6f - pred_loss=%.6f - val_loss=%.6f - val_mae=%.6f"
                % (epoch, args.epochs, train_loss, train_pred_loss, val_loss, val_mae)
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            wait = 0
            model.save_weights(args.checkpoint_path)
        else:
            wait += 1
            if wait >= args.early_stopping_patience:
                print("Early stopping at epoch %d; best epoch %d" % (epoch, best_epoch))
                break

    if os.path.exists(args.checkpoint_path):
        model.load_weights(args.checkpoint_path)
    pred_norm = predict_full_tensor(model, shape, chunks)
    train_metrics = evaluate_predictions(pred_norm, train_indices, train_values, target_scale)
    val_metrics = evaluate_predictions(pred_norm, val_indices, val_values, target_scale)
    test_metrics = evaluate_predictions(pred_norm, test_indices, test_values, target_scale)

    payload = {
        "config": vars(args),
        "variant_config": cfg,
        "data_stats": stats,
        "shape": shape,
        "target_scale": float(target_scale),
        "text_metadata": text_metadata,
        "checkpoint_path": args.checkpoint_path,
        "history_path": args.history_path,
        "best_epoch": int(best_epoch),
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)
    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Test metrics:")
    pprint(test_metrics)
    print("Saved metrics to:", args.metrics_path)


if __name__ == "__main__":
    main()
