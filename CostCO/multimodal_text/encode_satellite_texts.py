import argparse
import json
import os

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "all-MiniLM-L6-v2")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-8)


def encode_with_transformers(texts, model_name, batch_size, normalize,
                             local_files_only):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for the fallback MiniLM "
            "encoder. Install them with sentence-transformers or install "
            "transformers torch."
        ) from exc

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    vectors = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(texts), batch_size), 1):
            batch_texts = texts[start:start + batch_size]
            print(
                "Encoding with transformers mean pooling: batch %d/%d"
                % (batch_index, total_batches),
                flush=True,
            )
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            pooled = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            vectors.append((pooled / counts).cpu().numpy())

    embeddings = np.concatenate(vectors, axis=0).astype("float32")
    if normalize:
        embeddings = l2_normalize(embeddings).astype("float32")
    return embeddings


def encode_sentence_transformer(texts, model_name, batch_size, normalize,
                                local_files_only):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(
            "sentence-transformers is not available; falling back to "
            "transformers AutoModel mean pooling. Original error: %s" % exc,
            flush=True,
        )
        embeddings = encode_with_transformers(
            texts,
            model_name,
            batch_size,
            normalize,
            local_files_only,
        )
        return embeddings, "transformers_mean_pooling"

    try:
        model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        print(
            "SentenceTransformer loading failed; falling back to transformers "
            "AutoModel mean pooling. Original error: %s" % exc,
            flush=True,
        )
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        embeddings = encode_with_transformers(
            texts,
            model_name,
            batch_size,
            normalize,
            local_files_only,
        )
        return embeddings, "transformers_mean_pooling"

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    ).astype("float32")
    if normalize:
        embeddings = l2_normalize(embeddings).astype("float32")
    return embeddings, "sentence_transformer"


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
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help=(
            "Load the SentenceTransformer model only from local files. This is "
            "the default behavior unless --allow-remote-model is passed."
        ),
    )
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help=(
            "Allow SentenceTransformer to resolve the model from remote "
            "Hugging Face/cache instead of forcing offline local loading."
        ),
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
    local_files_only = args.local_files_only or not args.allow_remote_model
    endo_embeddings, endo_encoder_backend = encode_sentence_transformer(
        endo_texts,
        args.model_name,
        args.batch_size,
        normalize,
        local_files_only,
    )
    exo_embeddings, exo_encoder_backend = encode_sentence_transformer(
        exo_texts,
        args.model_name,
        args.batch_size,
        normalize,
        local_files_only,
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
            "encoder": endo_encoder_backend,
            "exo_encoder": exo_encoder_backend,
            "model_name": args.model_name,
            "embedding_dim": int(endo_embeddings.shape[1]),
            "dim": int(endo_embeddings.shape[1]),
            "normalize": normalize,
            "normalization": "l2" if normalize else "none",
            "batch_size": args.batch_size,
            "local_files_only": bool(local_files_only),
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
                "MiniLM sentence embeddings. The script first tries "
                "SentenceTransformer and falls back to transformers AutoModel "
                "mean pooling if the local SentenceTransformer pooling config "
                "is incompatible. all-MiniLM-L6-v2 normally outputs "
                "384-dimensional embeddings; model-side text_projection_dim "
                "controls the fusion dimension."
            ),
        },
    )

    print("Saved endo embeddings:", endo_embeddings.shape)
    print("Saved exo embeddings:", exo_embeddings.shape)
    print("Encoder:", args.model_name)
    print("Endo encoder backend:", endo_encoder_backend)
    print("Exo encoder backend:", exo_encoder_backend)
    print("Text dir:", args.text_dir)


if __name__ == "__main__":
    main()
