import argparse
import json
import os

import numpy as np


SOURCE_NUMERIC_FIELDS = [
    "mean_out_load",
    "peak_out_load",
    "out_trend_ratio",
    "destination_entropy",
    "cross_plane_out_ratio",
    "source_closeness",
    "source_bottleneck_exposure",
    "source_neighbor_change",
]

DESTINATION_NUMERIC_FIELDS = [
    "mean_in_load",
    "peak_in_load",
    "in_trend_ratio",
    "source_entropy",
    "cross_plane_in_ratio",
    "destination_closeness",
    "destination_bottleneck_exposure",
    "destination_neighbor_change",
]

TIME_NUMERIC_FIELDS = [
    "edge_density",
    "average_degree",
    "inter_plane_edge_ratio",
    "reachable_od_ratio",
    "topology_change_rate",
    "topology_change_zscore",
    "mean_shortest_path_hops",
    "mean_shortest_path_zscore",
    "p90_shortest_path_hops",
    "bottleneck_concentration",
    "bottleneck_zscore",
    "cross_plane_path_pressure",
    "cross_plane_path_pressure_zscore",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sentence_encoder(model_path, local_files_only=True):
    try:
        from sentence_transformers import SentenceTransformer

        return "sentence_transformers", SentenceTransformer(
            model_path,
            local_files_only=local_files_only,
        )
    except Exception as first_error:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
            model.eval()
            return "transformers", (tokenizer, model, torch)
        except Exception as second_error:
            raise RuntimeError(
                "Could not load text encoder with sentence-transformers or "
                "transformers. First error: %s. Second error: %s" %
                (first_error, second_error)
            )


def l2_normalize(values):
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def encode_texts(encoder_type, encoder, texts, batch_size=32, normalize=True):
    if encoder_type == "sentence_transformers":
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        ).astype("float32")
        return embeddings

    tokenizer, model, torch = encoder
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (output.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            pooled = summed / counts
            batches.append(pooled.cpu().numpy())
    embeddings = np.concatenate(batches, axis=0).astype("float32")
    if normalize:
        embeddings = l2_normalize(embeddings).astype("float32")
    return embeddings


def ordered_records(payload, expected_kind):
    metadata = payload["metadata"]
    records = payload["records"]
    target_times = metadata["target_times"]
    node_count = int(metadata["node_count"])
    if expected_kind in ("source", "destination"):
        expected = len(target_times) * node_count
        if len(records) != expected:
            raise ValueError(
                "%s records count %d does not match expected %d" %
                (expected_kind, len(records), expected)
            )
        records = sorted(
            records,
            key=lambda item: (item["time_index"], item["satellite_id"]),
        )
        return records, metadata

    if len(records) != len(target_times):
        raise ValueError("time records count does not match target_times")
    records = sorted(records, key=lambda item: item["time_index"])
    return records, metadata


def normalize_numeric_features(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype("float32"), mean, std


def extract_numeric_features(records, field_names, expected_kind):
    rows = []
    for record in records:
        missing = [name for name in field_names if name not in record]
        if missing:
            raise KeyError(
                "%s record is missing numeric fields: %s" %
                (expected_kind, missing)
            )
        rows.append([float(record[name]) for name in field_names])
    values = np.asarray(rows, dtype="float32")
    if not np.all(np.isfinite(values)):
        raise ValueError("%s numeric features contain NaN or Inf" % expected_kind)
    return normalize_numeric_features(values)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode deterministic mode texts into source/destination/time embeddings."
    )
    parser.add_argument("--text-dir", default="mode_text_data")
    parser.add_argument("--model-path", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    local_files_only = not args.allow_remote_model
    encoder_type, encoder = load_sentence_encoder(
        args.model_path,
        local_files_only=local_files_only,
    )
    normalize = not args.no_normalize
    source_payload = read_json(os.path.join(args.text_dir, "source_text_records.json"))
    destination_payload = read_json(
        os.path.join(args.text_dir, "destination_text_records.json")
    )
    time_payload = read_json(os.path.join(args.text_dir, "time_text_records.json"))
    source_records, metadata = ordered_records(source_payload, "source")
    destination_records, dest_metadata = ordered_records(
        destination_payload,
        "destination",
    )
    time_records, time_metadata = ordered_records(time_payload, "time")
    if metadata["target_times"] != dest_metadata["target_times"]:
        raise ValueError("source/destination target times do not match")
    if metadata["target_times"] != time_metadata["target_times"]:
        raise ValueError("source/time target times do not match")
    target_count = len(metadata["target_times"])
    node_count = int(metadata["node_count"])
    source_texts = [record["text"] for record in source_records]
    destination_texts = [record["text"] for record in destination_records]
    time_texts = [record["text"] for record in time_records]

    source_numeric, source_numeric_mean, source_numeric_std = (
        extract_numeric_features(source_records, SOURCE_NUMERIC_FIELDS, "source")
    )
    destination_numeric, destination_numeric_mean, destination_numeric_std = (
        extract_numeric_features(
            destination_records,
            DESTINATION_NUMERIC_FIELDS,
            "destination",
        )
    )
    time_numeric, time_numeric_mean, time_numeric_std = (
        extract_numeric_features(time_records, TIME_NUMERIC_FIELDS, "time")
    )

    source_embeddings = encode_texts(
        encoder_type,
        encoder,
        source_texts,
        batch_size=args.batch_size,
        normalize=normalize,
    )
    destination_embeddings = encode_texts(
        encoder_type,
        encoder,
        destination_texts,
        batch_size=args.batch_size,
        normalize=normalize,
    )
    time_embeddings = encode_texts(
        encoder_type,
        encoder,
        time_texts,
        batch_size=args.batch_size,
        normalize=normalize,
    )
    dim = source_embeddings.shape[1]
    source_embeddings = source_embeddings.reshape(target_count, node_count, dim)
    destination_embeddings = destination_embeddings.reshape(
        target_count,
        node_count,
        dim,
    )
    source_numeric = source_numeric.reshape(
        target_count,
        node_count,
        len(SOURCE_NUMERIC_FIELDS),
    )
    destination_numeric = destination_numeric.reshape(
        target_count,
        node_count,
        len(DESTINATION_NUMERIC_FIELDS),
    )
    if destination_embeddings.shape[-1] != dim or time_embeddings.shape[-1] != dim:
        raise ValueError("source/destination/time embedding dimensions differ")

    np.save(os.path.join(args.text_dir, "source_text_embeddings.npy"), source_embeddings)
    np.save(
        os.path.join(args.text_dir, "source_text_numeric_features.npy"),
        source_numeric,
    )
    np.save(
        os.path.join(args.text_dir, "destination_text_embeddings.npy"),
        destination_embeddings,
    )
    np.save(
        os.path.join(args.text_dir, "destination_text_numeric_features.npy"),
        destination_numeric,
    )
    np.save(os.path.join(args.text_dir, "time_text_embeddings.npy"), time_embeddings)
    np.save(os.path.join(args.text_dir, "time_text_numeric_features.npy"), time_numeric)
    with open(os.path.join(args.text_dir, "text_embedding_metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            **metadata,
            "embedding_dim": int(dim),
            "encoder_type": encoder_type,
            "model_path": args.model_path,
            "normalize": normalize,
            "numeric_side_channel": True,
            "source_numeric_fields": SOURCE_NUMERIC_FIELDS,
            "destination_numeric_fields": DESTINATION_NUMERIC_FIELDS,
            "time_numeric_fields": TIME_NUMERIC_FIELDS,
            "source_numeric_mean": source_numeric_mean.astype(float).tolist(),
            "source_numeric_std": source_numeric_std.astype(float).tolist(),
            "destination_numeric_mean": (
                destination_numeric_mean.astype(float).tolist()
            ),
            "destination_numeric_std": (
                destination_numeric_std.astype(float).tolist()
            ),
            "time_numeric_mean": time_numeric_mean.astype(float).tolist(),
            "time_numeric_std": time_numeric_std.astype(float).tolist(),
        }, f, indent=2, ensure_ascii=False)
    print("Saved source embeddings:", source_embeddings.shape)
    print("Saved source numeric features:", source_numeric.shape)
    print("Saved destination embeddings:", destination_embeddings.shape)
    print("Saved destination numeric features:", destination_numeric.shape)
    print("Saved time embeddings:", time_embeddings.shape)
    print("Saved time numeric features:", time_numeric.shape)


if __name__ == "__main__":
    main()
