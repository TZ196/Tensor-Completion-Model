import argparse
import hashlib
import json
import os
import re

import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def stable_hash(token):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def encode_text(text, dim):
    vector = np.zeros((dim,), dtype="float32")
    tokens = tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        hashed = stable_hash(token)
        index = hashed % dim
        sign = 1.0 if ((hashed >> 8) & 1) else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm > 0.0:
        vector = vector / norm
    return vector.astype("float32")


def encode_texts(texts, dim):
    return np.stack([encode_text(text, dim) for text in texts], axis=0)


def write_metadata(path, metadata):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode satellite endogenous/exogenous texts offline."
    )
    parser.add_argument("--text-dir", default="text_data")
    parser.add_argument("--dim", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    endo_path = os.path.join(args.text_dir, "endo_texts.json")
    exo_path = os.path.join(args.text_dir, "exo_text_segments.json")
    endo = load_json(endo_path)
    exo = load_json(exo_path)

    endo_records = sorted(
        endo["texts"],
        key=lambda item: int(item["time_index"]),
    )
    exo_records = exo["segments"]
    endo_texts = [item["text"] for item in endo_records]
    exo_texts = [item["text"] for item in exo_records]

    endo_embeddings = encode_texts(endo_texts, args.dim)
    exo_embeddings = encode_texts(exo_texts, args.dim)

    np.save(
        os.path.join(args.text_dir, "endo_text_embeddings.npy"),
        endo_embeddings,
    )
    np.save(
        os.path.join(args.text_dir, "exo_text_embeddings.npy"),
        exo_embeddings,
    )
    write_metadata(
        os.path.join(args.text_dir, "text_embedding_metadata.json"),
        {
            "encoder_type": "deterministic_hashing_bag_of_words",
            "encoder": "deterministic_hashing_bag_of_words",
            "embedding_dim": args.dim,
            "dim": args.dim,
            "normalize": True,
            "normalization": "l2",
            "text_generation_mode": endo.get(
                "metadata",
                {},
            ).get("text_generation_mode", "template"),
            "endo_text_source": endo.get("metadata", {}).get(
                "source",
                "unknown",
            ),
            "endo_shape": list(endo_embeddings.shape),
            "exo_segment_count": int(exo_embeddings.shape[0]),
            "exo_shape": list(exo_embeddings.shape),
            "endo_source": endo_path,
            "exo_source": exo_path,
            "note": (
                "This no-network baseline is deterministic and intended for "
                "early ablation. It can be replaced by Sentence-BERT or BERT "
                "embeddings later."
            ),
        },
    )

    print("Saved endo embeddings:", endo_embeddings.shape)
    print("Saved exo embeddings:", exo_embeddings.shape)
    print("Text dir:", args.text_dir)


if __name__ == "__main__":
    main()
