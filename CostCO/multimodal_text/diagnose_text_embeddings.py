import argparse
import json
import os

import numpy as np


def load_embeddings(path):
    embeddings = np.load(path).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def summarize(values):
    values = np.asarray(values, dtype="float32")
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def diagnose(embeddings, near_delta):
    sim = embeddings @ embeddings.T
    time_ids = np.arange(embeddings.shape[0])
    diff = np.abs(time_ids[:, None] - time_ids[None, :])
    off_diag = diff > 0
    near = (diff <= near_delta) & off_diag
    far = diff > near_delta
    adjacent = diff == 1
    return {
        "shape": list(embeddings.shape),
        "near_delta": int(near_delta),
        "off_diagonal_similarity": summarize(sim[off_diag]),
        "adjacent_similarity": summarize(sim[adjacent]),
        "near_similarity": summarize(sim[near]),
        "far_similarity": summarize(sim[far]),
    }


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether time text embeddings contain variation."
    )
    parser.add_argument(
        "--embedding-path",
        default="text_data/endo_text_embeddings.npy",
    )
    parser.add_argument("--near-delta", type=int, default=2)
    parser.add_argument(
        "--output-path",
        default="text_data/text_embedding_diagnostics.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings = load_embeddings(args.embedding_path)
    report = diagnose(embeddings, args.near_delta)
    report["embedding_path"] = args.embedding_path
    write_json(args.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved diagnostics to:", args.output_path)


if __name__ == "__main__":
    main()
