import argparse
import json
import os

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-8)


def encode_sentence_transformer(texts, model_name, batch_size, normalize,
                                local_files_only):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for text embedding. "
            "Install it with: pip install -U sentence-transformers"
        ) from exc

    try:
        model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )
    except TypeError:
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SentenceTransformer model '%s'. If the server "
            "cannot access Hugging Face or has SSL certificate issues, "
            "download all-MiniLM-L6-v2 on another machine, copy the model "
            "directory to the server, and pass that local path with "
            "--model-name /path/to/all-MiniLM-L6-v2 --local-files-only."
            % model_name
        ) from exc
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    ).astype("float32")
    if normalize:
        embeddings = l2_normalize(embeddings).astype("float32")
    return embeddings


def write_metadata(path, metadata):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode satellite endogenous/exogenous texts with "
            "SentenceTransformer all-MiniLM-L6-v2."
        )
    )
    parser.add_argument("--text-dir", default="text_data")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the SentenceTransformer model only from local files.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of text embeddings.",
    )
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

    normalize = not args.no_normalize
    endo_embeddings = encode_sentence_transformer(
        endo_texts,
        args.model_name,
        args.batch_size,
        normalize,
        args.local_files_only,
    )
    exo_embeddings = encode_sentence_transformer(
        exo_texts,
        args.model_name,
        args.batch_size,
        normalize,
        args.local_files_only,
    )

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
            "encoder_type": "sentence_transformer",
            "encoder": "sentence_transformer",
            "model_name": args.model_name,
            "embedding_dim": int(endo_embeddings.shape[1]),
            "dim": int(endo_embeddings.shape[1]),
            "normalize": normalize,
            "normalization": "l2" if normalize else "none",
            "batch_size": args.batch_size,
            "local_files_only": bool(args.local_files_only),
            "text_generation_mode": endo.get(
                "metadata",
                {},
            ).get("text_generation_mode", "deepseek"),
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
                "SentenceTransformer embeddings. all-MiniLM-L6-v2 normally "
                "outputs 384-dimensional sentence embeddings; model-side "
                "text_projection_dim controls the fusion dimension."
            ),
        },
    )

    print("Saved endo embeddings:", endo_embeddings.shape)
    print("Saved exo embeddings:", exo_embeddings.shape)
    print("Encoder:", args.model_name)
    print("Text dir:", args.text_dir)


if __name__ == "__main__":
    main()
