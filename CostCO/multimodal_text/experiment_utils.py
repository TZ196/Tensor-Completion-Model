import os

import numpy as np


def load_embeddings(path):
    embeddings = np.load(path).astype("float32")
    if embeddings.ndim != 2:
        raise ValueError("Expected 2-D embeddings at %s" % path)
    return embeddings


def apply_text_ablation(endo_embeddings, exo_embeddings, mode, seed):
    rng = np.random.RandomState(seed)
    endo = np.array(endo_embeddings, copy=True)
    exo = np.array(exo_embeddings, copy=True)
    if mode == "real":
        return endo, exo
    if mode == "shuffle_endo":
        order = rng.permutation(endo.shape[0])
        return endo[order], exo
    if mode == "random":
        endo_random = rng.normal(size=endo.shape).astype("float32")
        exo_random = rng.normal(size=exo.shape).astype("float32")
        endo_random /= np.maximum(
            np.linalg.norm(endo_random, axis=1, keepdims=True),
            1e-8,
        )
        exo_random /= np.maximum(
            np.linalg.norm(exo_random, axis=1, keepdims=True),
            1e-8,
        )
        return endo_random, exo_random
    raise ValueError("Unsupported text ablation mode: %s" % mode)


def load_existing_split(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            "Split file does not exist: %s. Run the preparation pipeline "
            "first, or pass --create-split in the runner." % split_path
        )
    data = np.load(split_path)
    stats = {
        "total_entries": int(data["total_entries"]),
        "finite_entries": int(data["finite_entries"]),
        "nonzero_finite_entries": int(data["nonzero_finite_entries"]),
        "zero_finite_entries": int(data["zero_finite_entries"]),
        "nonfinite_entries": int(data["nonfinite_entries"]),
    }
    return (
        data["shape"].astype("int32"),
        data["train_indices"].astype("int32"),
        data["train_values"].astype("float32"),
        data["val_indices"].astype("int32"),
        data["val_values"].astype("float32"),
        data["test_indices"].astype("int32"),
        data["test_values"].astype("float32"),
        stats,
    )


def default_split_path(observed_ratio, val_ratio, seed):
    return os.path.join(
        "..",
        "splits",
        "random_observed%d_val%d_seed_%d.npz" % (
            int(round(observed_ratio * 100)),
            int(round(val_ratio * 100)),
            seed,
        ),
    )


def format_lr(lr):
    return ("%g" % lr).replace(".", "p").replace("-", "m")


def stage_flags(text_stage, flow_text_weight, graph_text_weight):
    return {
        "cross_view_attention": text_stage in [
            "cross_attention",
            "semantic_gating",
            "segment_condenser",
        ],
        "semantic_gating": text_stage in [
            "semantic_gating",
            "segment_condenser",
        ],
        "segment_condenser": text_stage == "segment_condenser",
        "contrastive_enabled": (
            flow_text_weight > 0.0 or graph_text_weight > 0.0
        ),
    }
