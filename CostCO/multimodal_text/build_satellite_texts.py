import argparse
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def compute_shortest_distances(adjacency):
    node_count = adjacency.shape[0]
    unreachable = node_count + 1
    distances = np.full((node_count, node_count), unreachable, dtype="int32")
    graph = adjacency > 0

    for src in range(node_count):
        distances[src, src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor in np.flatnonzero(graph[current]):
                if distances[src, neighbor] == unreachable:
                    distances[src, neighbor] = distances[src, current] + 1
                    queue.append(int(neighbor))

    reachable = distances < unreachable
    return distances, reachable


def load_tensor(path):
    tensor = np.load(path)
    if tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape %s" %
                         (tensor.shape,))
    return tensor.astype("float32")


def nonzero_finite_entries(tensor):
    finite_mask = np.isfinite(tensor)
    mask = finite_mask & (tensor != 0)
    indices = np.argwhere(mask).astype("int32")
    values = tensor[mask].astype("float32")
    stats = {
        "total_entries": int(tensor.size),
        "finite_entries": int(np.sum(finite_mask)),
        "nonzero_finite_entries": int(np.sum(mask)),
        "zero_finite_entries": int(np.sum(finite_mask & (tensor == 0))),
        "nonfinite_entries": int(tensor.size - np.sum(finite_mask)),
    }
    return indices, values, stats


def create_random_completion_split(tensor_path, split_path, observed_ratio,
                                   val_ratio, seed):
    tensor = load_tensor(tensor_path)
    indices, values, data_stats = nonzero_finite_entries(tensor)
    if indices.shape[0] < 3:
        raise ValueError("Need at least 3 non-zero finite entries to split")

    rng = np.random.RandomState(seed)
    order = rng.permutation(indices.shape[0])
    train_size = int(round(indices.shape[0] * observed_ratio))
    train_size = max(1, min(train_size, indices.shape[0] - 2))
    remaining_size = indices.shape[0] - train_size
    val_size = int(round(remaining_size * val_ratio))
    val_size = max(1, min(val_size, remaining_size - 1))

    train_order = order[:train_size]
    val_order = order[train_size:train_size + val_size]
    test_order = order[train_size + val_size:]

    split_dir = os.path.dirname(split_path)
    if split_dir and not os.path.exists(split_dir):
        os.makedirs(split_dir)
    np.savez(
        split_path,
        shape=np.array(tensor.shape).astype("int32"),
        train_indices=indices[train_order],
        train_values=values[train_order],
        val_indices=indices[val_order],
        val_values=values[val_order],
        test_indices=indices[test_order],
        test_values=values[test_order],
        observed_ratio=np.array(observed_ratio).astype("float32"),
        missing_rate=np.array(1.0 - observed_ratio).astype("float32"),
        val_ratio=np.array(val_ratio).astype("float32"),
        seed=np.array(seed).astype("int32"),
        total_entries=np.array(data_stats["total_entries"]).astype("int64"),
        finite_entries=np.array(data_stats["finite_entries"]).astype("int64"),
        nonzero_finite_entries=np.array(
            data_stats["nonzero_finite_entries"]
        ).astype("int64"),
        zero_finite_entries=np.array(
            data_stats["zero_finite_entries"]
        ).astype("int64"),
        nonfinite_entries=np.array(
            data_stats["nonfinite_entries"]
        ).astype("int64"),
    )

    return (
        np.array(tensor.shape).astype("int32"),
        indices[train_order],
        values[train_order],
        indices[val_order],
        values[val_order],
        indices[test_order],
        values[test_order],
        data_stats,
    )


def load_completion_split(split_path, tensor_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            "Split file does not exist: %s. Create the split first, or pass "
            "--create-split to generate it explicitly." % split_path
        )

    split = np.load(split_path)
    required = [
        "train_indices",
        "train_values",
        "val_indices",
        "val_values",
        "test_indices",
        "test_values",
    ]
    missing = [key for key in required if key not in split.files]
    if missing:
        raise ValueError("Split file is missing keys: %s" % missing)

    if "shape" in split.files:
        shape = split["shape"].astype("int32")
    else:
        shape = np.array(load_tensor(tensor_path).shape).astype("int32")

    data_stats = {}
    stat_keys = [
        "total_entries",
        "finite_entries",
        "nonzero_finite_entries",
        "zero_finite_entries",
        "nonfinite_entries",
    ]
    if all(key in split.files for key in stat_keys):
        for key in stat_keys:
            data_stats[key] = int(split[key])
    else:
        _indices, _values, data_stats = nonzero_finite_entries(
            load_tensor(tensor_path)
        )

    return (
        shape,
        split["train_indices"].astype("int32"),
        split["train_values"].astype("float32"),
        split["val_indices"].astype("int32"),
        split["val_values"].astype("float32"),
        split["test_indices"].astype("int32"),
        split["test_values"].astype("float32"),
        data_stats,
    )


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
        raise ValueError("Cannot find topology tensor in %s, keys=%s" %
                         (path, data.files))

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
        raise ValueError("Unexpected topology shape %s for traffic shape %s" %
                         (topo.shape, traffic_shape))

    topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
    return (topo > 0).astype("float32")


def trend_label(current, previous):
    if previous is None:
        return "not available for the first time slice"
    if current > previous * 1.05:
        return "increased compared with the previous time slice"
    if current < previous * 0.95:
        return "decreased compared with the previous time slice"
    return "remained approximately stable compared with the previous time slice"


def build_time_statistics(shape, topology, train_indices, train_values):
    time_count, node_count, _ = topology.shape
    if time_count != int(shape[2]):
        raise ValueError("topology time length does not match traffic shape")

    train_time = train_indices[:, 2]
    previous_train_mean = None
    stats = []
    previous_adj = None

    for time_idx in range(time_count):
        adjacency = topology[time_idx]
        degrees = adjacency.sum(axis=1)
        edge_count_directed = int(adjacency.sum())
        edge_count_undirected = int(round(edge_count_directed / 2.0))
        changed_entries = 0
        changed_edges = 0
        if previous_adj is not None:
            changed_entries = int(np.sum(adjacency != previous_adj))
            changed_edges = int(round(changed_entries / 2.0))

        distances, reachable = compute_shortest_distances(adjacency)
        off_diag = ~np.eye(node_count, dtype=bool)
        reachable_pairs = reachable & off_diag
        if np.any(reachable_pairs):
            reachable_distances = distances[reachable_pairs]
            avg_shortest_path = float(np.mean(reachable_distances))
            diameter = int(np.max(reachable_distances))
        else:
            avg_shortest_path = float(node_count + 1)
            diameter = int(node_count + 1)

        is_connected = bool(np.all(reachable | np.eye(node_count, dtype=bool)))

        time_mask = train_time == time_idx
        observed_values = train_values[time_mask]
        observed_count = int(observed_values.shape[0])
        if observed_count > 0:
            train_mean = float(np.mean(observed_values))
            train_max = float(np.max(observed_values))
            train_min = float(np.min(observed_values))
            train_std = float(np.std(observed_values))
            train_p25 = float(np.percentile(observed_values, 25))
            train_p50 = float(np.percentile(observed_values, 50))
            train_p75 = float(np.percentile(observed_values, 75))
        else:
            train_mean = 0.0
            train_max = 0.0
            train_min = 0.0
            train_std = 0.0
            train_p25 = 0.0
            train_p50 = 0.0
            train_p75 = 0.0

        top_source_ids = []
        top_destination_ids = []
        if observed_count > 0:
            time_indices = train_indices[time_mask]
            src_counts = np.bincount(time_indices[:, 0], minlength=shape[0])
            dst_counts = np.bincount(time_indices[:, 1], minlength=shape[1])
            top_source_ids = np.argsort(src_counts)[-3:][::-1].astype(
                int
            ).tolist()
            top_destination_ids = np.argsort(dst_counts)[-3:][::-1].astype(
                int
            ).tolist()

        possible_entries = int(shape[0] * shape[1])
        observed_ratio = observed_count / float(possible_entries)
        if observed_ratio < 0.1:
            sparsity = "high"
        elif observed_ratio < 0.3:
            sparsity = "moderate"
        else:
            sparsity = "low"

        item = {
            "time_index": int(time_idx),
            "num_satellites": int(node_count),
            "edge_count_undirected": edge_count_undirected,
            "edge_count_directed_entries": edge_count_directed,
            "avg_degree": float(np.mean(degrees)),
            "min_degree": float(np.min(degrees)),
            "max_degree": float(np.max(degrees)),
            "is_connected": is_connected,
            "avg_shortest_path_hops": avg_shortest_path,
            "diameter_hops": diameter,
            "changed_adjacency_entries_from_prev": changed_entries,
            "changed_edges_from_prev": changed_edges,
            "observed_train_nonzero_count": observed_count,
            "observed_train_ratio_within_slice": observed_ratio,
            "observed_train_mean_mb": train_mean,
            "observed_train_min_mb": train_min,
            "observed_train_max_mb": train_max,
            "observed_train_std_mb": train_std,
            "observed_train_p25_mb": train_p25,
            "observed_train_median_mb": train_p50,
            "observed_train_p75_mb": train_p75,
            "observed_train_sparsity": sparsity,
            "observed_train_trend": trend_label(
                train_mean,
                previous_train_mean,
            ),
            "top_observed_source_ids": top_source_ids,
            "top_observed_destination_ids": top_destination_ids,
        }
        stats.append(item)
        previous_train_mean = train_mean
        previous_adj = adjacency

    return stats


def render_endogenous_text(stats, mode):
    texts = []
    for item in stats:
        connected_text = "connected" if item["is_connected"] else "not fully connected"
        topo_text = (
            "At time {time_index}, the LEO satellite network contains "
            "{num_satellites} satellites and {edge_count_undirected} "
            "undirected inter-satellite links. The topology is {connected}, "
            "with an average node degree of {avg_degree:.2f}, a degree range "
            "from {min_degree:.0f} to {max_degree:.0f}, an average shortest "
            "path distance of {avg_shortest_path_hops:.2f} hops, and a "
            "diameter of {diameter_hops} hops. Compared with the previous "
            "second, {changed_edges_from_prev} links changed."
        ).format(
            time_index=item["time_index"],
            num_satellites=item["num_satellites"],
            edge_count_undirected=item["edge_count_undirected"],
            connected=connected_text,
            avg_degree=item["avg_degree"],
            min_degree=item["min_degree"],
            max_degree=item["max_degree"],
            avg_shortest_path_hops=item["avg_shortest_path_hops"],
            diameter_hops=item["diameter_hops"],
            changed_edges_from_prev=item["changed_edges_from_prev"],
        )
        flow_text = (
            " Based only on observed training entries, this time slice has "
            "{count} visible non-zero path-traffic samples, {sparsity} "
            "observed sparsity, mean traffic {mean:.4f} MB, median traffic "
            "{median:.4f} MB, interquartile range {p25:.4f}-{p75:.4f} MB, "
            "standard deviation {std:.4f} MB, top observed source satellite "
            "IDs {top_src}, top observed destination satellite IDs {top_dst}, "
            "and a trend that {trend}."
        ).format(
            count=item["observed_train_nonzero_count"],
            sparsity=item["observed_train_sparsity"],
            mean=item["observed_train_mean_mb"],
            median=item["observed_train_median_mb"],
            p25=item["observed_train_p25_mb"],
            p75=item["observed_train_p75_mb"],
            std=item["observed_train_std_mb"],
            top_src=item["top_observed_source_ids"],
            top_dst=item["top_observed_destination_ids"],
            trend=item["observed_train_trend"],
        )
        if mode == "topo":
            text = topo_text
        elif mode == "topoflow":
            text = topo_text + flow_text
        else:
            raise ValueError("Unsupported endogenous text mode: %s" % mode)
        texts.append({
            "time_index": item["time_index"],
            "endo_mode": mode,
            "text": text,
        })
    return texts


def default_exogenous_segments():
    return [
        {
            "segment_id": "C1_simulation_configuration",
            "text": (
                "The experiment simulates a LEO satellite network with 120 "
                "satellites over 60 time slices at a 1000 ms interval. The "
                "target tensor represents source-destination satellite path "
                "traffic over time."
            ),
        },
        {
            "segment_id": "C2_link_capacity",
            "text": (
                "Inter-satellite links provide connectivity between satellites. "
                "Path traffic can be affected by link availability, link "
                "capacity constraints, and the number of hops used by the "
                "routing path."
            ),
        },
        {
            "segment_id": "C3_routing_mechanism",
            "text": (
                "Routing over the satellite topology can map one "
                "source-destination demand to a multi-hop path. Long-distance "
                "pairs may traverse several inter-satellite links before "
                "reaching the destination satellite."
            ),
        },
        {
            "segment_id": "C4_topology_structure",
            "text": (
                "Each satellite can maintain multiple inter-satellite links, "
                "including local and cross-neighbor connections. The dynamic "
                "topology affects reachability, path length, and traffic "
                "distribution."
            ),
        },
        {
            "segment_id": "C5_bottleneck_explanation",
            "text": (
                "Network bottlenecks may occur when many paths share a small "
                "set of links or when access constraints limit delivery. Such "
                "bottlenecks can influence observed path-traffic values."
            ),
        },
        {
            "segment_id": "C6_dynamic_topology_change",
            "text": (
                "Satellite movement can change inter-satellite connectivity "
                "over time. Even sparse topology changes may alter shortest "
                "paths and redistribute path traffic across the network."
            ),
        },
        {
            "segment_id": "C7_path_traffic_semantics",
            "text": (
                "The path traffic tensor records traffic volume between source "
                "and destination satellites at each time slice. It reflects "
                "traffic demand together with the routing paths induced by the "
                "current satellite topology."
            ),
        },
    ]


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def load_json_compat(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build leakage-safe satellite text descriptions."
    )
    parser.add_argument(
        "--tensor-path",
        default=os.path.join(PROJECT_DIR, "sat_path_bytes_mb_tensor.npy"),
    )
    parser.add_argument(
        "--topology-path",
        default=os.path.join(
            PROJECT_DIR,
            "sat_connectivity_tensor_dynamic_60s_1000ms.npz",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "text_data"),
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Explicitly create the split if it does not exist.",
    )
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Recreate the split even when split-path already exists.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--endo-mode",
        choices=["topo", "topoflow", "both"],
        default="both",
    )
    parser.add_argument(
        "--text-generation-mode",
        choices=["template"],
        default="template",
    )
    parser.add_argument(
        "--write-template-endo",
        action="store_true",
        help="Write template endogenous texts as a fallback/debug artifact.",
    )
    parser.add_argument(
        "--write-template-exo",
        action="store_true",
        help="Write manually constructed template exogenous text segments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    observed_ratio = args.observed_ratio
    if observed_ratio is None:
        observed_ratio = 1.0 - args.missing_rate

    if args.split_path is None:
        args.split_path = os.path.join(
            PROJECT_DIR,
            "splits",
            "random_observed%d_val%d_seed_%d.npz" % (
                int(round(observed_ratio * 100)),
                int(round(args.val_ratio * 100)),
                args.seed,
            ),
        )

    if args.create_split or args.overwrite_split:
        if os.path.exists(args.split_path) and not args.overwrite_split:
            print("Using existing split:", args.split_path)
            split_data = load_completion_split(args.split_path, args.tensor_path)
        else:
            print("Creating split:", args.split_path)
            split_data = create_random_completion_split(
                args.tensor_path,
                args.split_path,
                observed_ratio,
                args.val_ratio,
                args.seed,
            )
    else:
        print("Loading existing split:", args.split_path)
        split_data = load_completion_split(args.split_path, args.tensor_path)

    (
        shape,
        train_indices,
        train_values,
        _val_indices,
        _val_values,
        _test_indices,
        _test_values,
        data_stats,
    ) = split_data
    topology = load_connectivity_tensor(args.topology_path, shape)
    stats = build_time_statistics(
        shape,
        topology,
        train_indices,
        train_values,
    )
    endo_modes = ["topo", "topoflow"] if args.endo_mode == "both" else [
        args.endo_mode
    ]

    output_dir = args.output_dir
    write_json(
        os.path.join(output_dir, "time_stats_train_only.json"),
        {
            "metadata": {
                "tensor_path": args.tensor_path,
                "topology_path": args.topology_path,
                "split_path": args.split_path,
                "mask_type": "fixed_global_random_transductive_mask",
                "mask_scope": "global_nonzero_finite_tensor_entries",
                "mask_lifecycle": "fixed_after_split_generation",
                "observed_ratio": observed_ratio,
                "missing_rate": 1.0 - observed_ratio,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "sample_filter": "finite_and_nonzero",
                "traffic_stats_source": "train_observed_entries_only",
                "total_entries": data_stats["total_entries"],
                "nonzero_finite_entries": data_stats[
                    "nonzero_finite_entries"
                ],
            },
            "time_statistics": stats,
        },
    )
    if args.write_template_endo:
        for endo_mode in endo_modes:
            endo_texts = render_endogenous_text(stats, endo_mode)
            if endo_mode == "topo":
                source = "topology_statistics_only"
                filename = "endo_texts_topo_template.json"
            else:
                source = "topology_and_train_observed_statistics_only"
                filename = "endo_texts_topoflow_template.json"
            write_json(
                os.path.join(output_dir, filename),
                {
                    "metadata": {
                        "source": source,
                        "text_generation_mode": args.text_generation_mode,
                        "num_time_slices": int(shape[2]),
                    },
                    "texts": endo_texts,
                },
            )
    if args.write_template_exo:
        exo_segments = default_exogenous_segments()
        write_json(
            os.path.join(output_dir, "exo_text_segments_template.json"),
            {
                "metadata": {
                    "source": "manually_constructed_domain_prior_template",
                    "num_segments": len(exo_segments),
                },
                "segments": exo_segments,
            },
        )

    print("Wrote text data to:", output_dir)
    print("Time slices:", len(stats))
    print("Split path:", args.split_path)
    print("Template endogenous written:", bool(args.write_template_endo))
    print("Template exogenous written:", bool(args.write_template_exo))


if __name__ == "__main__":
    main()
