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


def load_connectivity_tensor(path):
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

    topo = np.asarray(topo, dtype="float32")
    if topo.ndim != 3:
        raise ValueError("Expected topology shape [time, nodes, nodes], got %s"
                         % (topo.shape,))
    topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
    return (topo > 0).astype("float32")


def build_topology_time_statistics(topology):
    time_count, node_count, _ = topology.shape
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
        stats.append({
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
        })
        previous_adj = adjacency

    return stats


def render_endogenous_text(stats):
    texts = []
    for item in stats:
        connected_text = (
            "connected" if item["is_connected"] else "not fully connected"
        )
        text = (
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
        texts.append({
            "time_index": item["time_index"],
            "endo_mode": "topo",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build topology-only satellite text statistics. This script does "
            "not read traffic values and does not create or load random masks."
        )
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
    parser.add_argument(
        "--write-template-endo",
        action="store_true",
        help="Write template endogenous topology texts as a fallback artifact.",
    )
    parser.add_argument(
        "--write-template-exo",
        action="store_true",
        help="Write manually constructed template exogenous text segments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    topology = load_connectivity_tensor(args.topology_path)
    stats = build_topology_time_statistics(topology)
    output_dir = args.output_dir

    write_json(
        os.path.join(output_dir, "time_stats_topo_only.json"),
        {
            "metadata": {
                "topology_path": args.topology_path,
                "stats_source": "topology_only",
                "num_time_slices": int(topology.shape[0]),
                "num_satellites": int(topology.shape[1]),
                "note": (
                    "Topology-only statistics for DeepSeek endogenous text "
                    "generation. No traffic tensor values, training mask, "
                    "validation entries, or test entries are included."
                ),
            },
            "time_statistics": stats,
        },
    )
    if args.write_template_endo:
        write_json(
            os.path.join(output_dir, "endo_texts_topo_template.json"),
            {
                "metadata": {
                    "source": "topology_statistics_only",
                    "text_generation_mode": "template",
                    "num_time_slices": int(topology.shape[0]),
                },
                "texts": render_endogenous_text(stats),
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

    print("Wrote topology-only text data to:", output_dir)
    print("Time slices:", len(stats))
    print("Topology path:", args.topology_path)
    print("Template endogenous written:", bool(args.write_template_endo))
    print("Template exogenous written:", bool(args.write_template_exo))


if __name__ == "__main__":
    main()
