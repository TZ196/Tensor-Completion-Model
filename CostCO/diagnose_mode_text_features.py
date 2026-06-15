import argparse
import json
import os

import numpy as np


EPS = 1e-8


def load_array(text_dir, name, required=True):
    path = os.path.join(text_dir, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return None
    return np.load(path).astype("float32")


def flatten(arr):
    return arr.reshape(-1, arr.shape[-1])


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def cosine_mean(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.mean(np.sum(a * b, axis=-1)))


def variance_report(arr):
    flat = flatten(arr)
    dim_std = flat.std(axis=0)
    return {
        "shape": [int(v) for v in arr.shape],
        "dim_std_mean": float(dim_std.mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std_max": float(dim_std.max()),
        "flat_norm_mean": float(np.linalg.norm(flat, axis=1).mean()),
        "near_collapse": bool(dim_std.mean() < 0.05),
    }


def temporal_report(arr):
    if arr.ndim != 3 or arr.shape[0] < 2:
        return {
            "mean_adjacent_cosine": None,
            "mean_adjacent_l2": None,
            "low_temporal_variation": None,
        }
    prev = arr[:-1].reshape(-1, arr.shape[-1])
    nxt = arr[1:].reshape(-1, arr.shape[-1])
    diff = nxt - prev
    mean_adjacent_cosine = cosine_mean(prev, nxt)
    mean_adjacent_l2 = float(np.linalg.norm(diff, axis=1).mean())
    return {
        "mean_adjacent_cosine": mean_adjacent_cosine,
        "mean_adjacent_l2": mean_adjacent_l2,
        "low_temporal_variation": bool(mean_adjacent_cosine > 0.98),
    }


def asymmetry_report(source, destination):
    if source.shape != destination.shape:
        return {
            "same_shape": False,
            "mean_cosine": None,
            "mean_l2": None,
            "nearly_identical": None,
        }
    src = flatten(source)
    dst = flatten(destination)
    mean_cosine = cosine_mean(src, dst)
    mean_l2 = float(np.linalg.norm(src - dst, axis=1).mean())
    exact_equal = bool(np.array_equal(source, destination))
    return {
        "same_shape": True,
        "mean_cosine": mean_cosine,
        "mean_l2": mean_l2,
        "exact_equal": exact_equal,
        "nearly_identical": bool(exact_equal or mean_cosine > 0.98),
    }


def numeric_report(arr):
    if arr is None:
        return None
    flat = flatten(arr)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    max_abs_mean = float(np.max(np.abs(mean)))
    min_std = float(np.min(std))
    max_std = float(np.max(std))
    std_ratio = float(max_std / (min_std + EPS))
    return {
        "shape": [int(v) for v in arr.shape],
        "max_abs_column_mean": max_abs_mean,
        "min_column_std": min_std,
        "max_column_std": max_std,
        "std_ratio": std_ratio,
        "looks_zscored": bool(max_abs_mean < 0.1 and 0.5 <= min_std and max_std <= 2.0),
        "large_scale_gap": bool(std_ratio > 20.0),
    }


def explained_variance_linear(x, y, sample_size, seed):
    x = flatten(x)
    y = flatten(y)
    if x.shape[0] != y.shape[0]:
        return None
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = x[idx]
        y = y[idx]
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + EPS)
    y = (y - y.mean(axis=0, keepdims=True)) / (y.std(axis=0, keepdims=True) + EPS)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=1e-4)
    pred = x_aug @ coef
    sse = np.sum((y - pred) ** 2)
    sst = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2) + EPS
    return float(1.0 - sse / sst)


def text_numeric_overlap_report(text, numeric, sample_size, seed):
    if numeric is None:
        return None
    r2_num_to_text = explained_variance_linear(
        numeric,
        text,
        sample_size=sample_size,
        seed=seed,
    )
    r2_text_to_num = explained_variance_linear(
        text,
        numeric,
        sample_size=sample_size,
        seed=seed,
    )
    return {
        "linear_r2_numeric_to_text": r2_num_to_text,
        "linear_r2_text_to_numeric": r2_text_to_num,
        "high_overlap": bool(
            (r2_num_to_text is not None and r2_num_to_text > 0.5) or
            (r2_text_to_num is not None and r2_text_to_num > 0.5)
        ),
    }


def print_section(title, rows):
    print(title)
    for key, value in rows.items():
        print("  %s: %s" % (key, value))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether mode text features are likely useful."
    )
    parser.add_argument("--mode-text-dir", default="mode_text_numeric_ablation_data/both")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    source = load_array(args.mode_text_dir, "source_text_embeddings.npy")
    destination = load_array(args.mode_text_dir, "destination_text_embeddings.npy")
    time = load_array(args.mode_text_dir, "time_text_embeddings.npy")
    source_numeric = load_array(
        args.mode_text_dir,
        "source_text_numeric_features.npy",
        required=False,
    )
    destination_numeric = load_array(
        args.mode_text_dir,
        "destination_text_numeric_features.npy",
        required=False,
    )
    time_numeric = load_array(
        args.mode_text_dir,
        "time_text_numeric_features.npy",
        required=False,
    )
    metadata_path = os.path.join(args.mode_text_dir, "text_embedding_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    static_source_destination = (
        metadata.get("source_destination_text_granularity") ==
        "static_node_repeated_by_time"
    )

    report = {
        "mode_text_dir": args.mode_text_dir,
        "metadata": {
            "text_generation": metadata.get("text_generation"),
            "source_destination_text_granularity": metadata.get(
                "source_destination_text_granularity"
            ),
            "time_text_granularity": metadata.get("time_text_granularity"),
        },
        "source_variance": variance_report(source),
        "destination_variance": variance_report(destination),
        "time_variance": variance_report(time),
        "source_temporal": temporal_report(source),
        "destination_temporal": temporal_report(destination),
        "time_temporal": temporal_report(time[:, None, :]),
        "source_destination_asymmetry": asymmetry_report(source, destination),
        "source_numeric": numeric_report(source_numeric),
        "destination_numeric": numeric_report(destination_numeric),
        "time_numeric": numeric_report(time_numeric),
        "source_text_numeric_overlap": text_numeric_overlap_report(
            source,
            source_numeric,
            args.sample_size,
            args.seed,
        ),
        "destination_text_numeric_overlap": text_numeric_overlap_report(
            destination,
            destination_numeric,
            args.sample_size,
            args.seed,
        ),
        "time_text_numeric_overlap": text_numeric_overlap_report(
            time,
            time_numeric,
            args.sample_size,
            args.seed,
        ),
    }

    warnings = []
    for name in ("source_variance", "destination_variance", "time_variance"):
        if report[name]["near_collapse"]:
            warnings.append("%s: embedding variance may be collapsed" % name)
    temporal_names = ["time_temporal"]
    if not static_source_destination:
        temporal_names.extend(["source_temporal", "destination_temporal"])
    for name in temporal_names:
        if report[name]["low_temporal_variation"]:
            warnings.append("%s: temporal variation is very low" % name)
    if report["source_destination_asymmetry"]["nearly_identical"]:
        warnings.append("source/destination embeddings are nearly identical")
    for name in ("source_numeric", "destination_numeric", "time_numeric"):
        item = report[name]
        if item and not item["looks_zscored"]:
            warnings.append("%s: numeric features do not look z-scored" % name)
        if item and item["large_scale_gap"]:
            warnings.append("%s: numeric feature scales differ strongly" % name)
    for name in (
        "source_text_numeric_overlap",
        "destination_text_numeric_overlap",
        "time_text_numeric_overlap",
    ):
        item = report[name]
        if item and item["high_overlap"]:
            warnings.append("%s: text and numeric features may be redundant" % name)
    report["warnings"] = warnings

    print_section("Source Variance", report["source_variance"])
    print_section("Destination Variance", report["destination_variance"])
    print_section("Time Variance", report["time_variance"])
    print_section("Source Temporal", report["source_temporal"])
    print_section("Destination Temporal", report["destination_temporal"])
    print_section("Time Temporal", report["time_temporal"])
    print_section("Source/Destination Asymmetry", report["source_destination_asymmetry"])
    if report["source_numeric"]:
        print_section("Source Numeric", report["source_numeric"])
        print_section("Destination Numeric", report["destination_numeric"])
        print_section("Time Numeric", report["time_numeric"])
    print("Warnings")
    if warnings:
        for warning in warnings:
            print("  - %s" % warning)
    else:
        print("  none")

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.mode_text_dir, "mode_text_diagnostics.json")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print("Saved diagnostics to:", output_path)


if __name__ == "__main__":
    main()
