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


def algebraic_connectivity(adjacency):
    graph = np.maximum(adjacency, adjacency.T).astype("float32")
    degree = np.sum(graph, axis=1)
    laplacian = np.diag(degree) - graph
    eigenvalues = np.linalg.eigvalsh(laplacian)
    if eigenvalues.shape[0] < 2:
        return 0.0
    return float(max(eigenvalues[1], 0.0))


def edge_betweenness_top3(adjacency):
    """Brandes edge betweenness for an undirected unweighted graph."""
    graph = np.maximum(adjacency, adjacency.T) > 0
    node_count = graph.shape[0]
    edge_scores = {}

    for source in range(node_count):
        stack = []
        predecessors = [[] for _ in range(node_count)]
        sigma = np.zeros(node_count, dtype="float64")
        distance = np.full(node_count, -1, dtype="int32")
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1
            stack.append(current)
            for neighbor in np.flatnonzero(graph[current]):
                if distance[neighbor] < 0:
                    queue.append(int(neighbor))
                    distance[neighbor] = distance[current] + 1
                if distance[neighbor] == distance[current] + 1:
                    sigma[neighbor] += sigma[current]
                    predecessors[neighbor].append(current)

        dependency = np.zeros(node_count, dtype="float64")
        while stack:
            node = stack.pop()
            if sigma[node] <= 0:
                continue
            for predecessor in predecessors[node]:
                share = (
                    sigma[predecessor] / sigma[node] *
                    (1.0 + dependency[node])
                )
                edge = tuple(sorted((int(predecessor), int(node))))
                edge_scores[edge] = edge_scores.get(edge, 0.0) + share
                dependency[predecessor] += share

    for edge in list(edge_scores):
        edge_scores[edge] *= 0.5

    total_score = sum(edge_scores.values())
    top_edges = sorted(
        edge_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    top_links = [
        {
            "source": int(edge[0]),
            "destination": int(edge[1]),
            "score": float(score),
        }
        for edge, score in top_edges
    ]
    top_share = 0.0
    if total_score > 0.0:
        top_share = float(sum(score for _edge, score in top_edges) / total_score)
    return top_links, top_share


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
    topo = normalize_topology_layout(topo)
    topo = np.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
    return (topo > 0).astype("float32")


def normalize_topology_layout(topo):
    """Normalize topology to [time, nodes, nodes] without reading traffic data."""
    shape = topo.shape
    if shape[1] == shape[2]:
        return topo
    if shape[0] == shape[1]:
        return np.transpose(topo, (2, 0, 1))
    if shape[0] == shape[2]:
        return np.transpose(topo, (1, 0, 2))
    raise ValueError(
        "Cannot infer topology time axis from shape %s. Expected one square "
        "node-node plane and one time axis." % (shape,)
    )


def build_topology_time_statistics(topology):
    time_count, node_count, _ = topology.shape
    stats = []
    previous_adj = None
    previous_lambda2 = None
    previous_major_change = 0
    edge_change_history = []

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
        edge_change_history.append(changed_edges)
        recent_changes = edge_change_history[-5:]
        rolling_edge_change_5 = float(np.mean(recent_changes))
        if changed_edges >= 4:
            previous_major_change = time_idx
        steps_since_major_change = int(time_idx - previous_major_change)

        distances, reachable = compute_shortest_distances(adjacency)
        off_diag = ~np.eye(node_count, dtype=bool)
        reachable_pairs = reachable & off_diag
        if np.any(reachable_pairs):
            reachable_distances = distances[reachable_pairs]
            avg_shortest_path = float(np.mean(reachable_distances))
            shortest_path_variance = float(np.var(reachable_distances))
            diameter = int(np.max(reachable_distances))
            long_path_ratio = float(np.mean(reachable_distances > 8))
        else:
            avg_shortest_path = float(node_count + 1)
            shortest_path_variance = 0.0
            diameter = int(node_count + 1)
            long_path_ratio = 0.0

        is_connected = bool(np.all(reachable | np.eye(node_count, dtype=bool)))
        lambda2 = algebraic_connectivity(adjacency)
        lambda2_delta = 0.0 if previous_lambda2 is None else (
            lambda2 - previous_lambda2
        )
        top_bottleneck_links, bottleneck_top3_share = edge_betweenness_top3(
            adjacency
        )
        stats.append({
            "time_index": int(time_idx),
            "normalized_time_phase": (
                float(time_idx / max(time_count - 1, 1))
            ),
            "num_satellites": int(node_count),
            "edge_count_undirected": edge_count_undirected,
            "edge_count_directed_entries": edge_count_directed,
            "avg_degree": float(np.mean(degrees)),
            "min_degree": float(np.min(degrees)),
            "max_degree": float(np.max(degrees)),
            "is_connected": is_connected,
            "avg_shortest_path_hops": avg_shortest_path,
            "shortest_path_variance": shortest_path_variance,
            "long_path_ratio_gt8": long_path_ratio,
            "diameter_hops": diameter,
            "algebraic_connectivity_lambda2": float(lambda2),
            "lambda2_delta_from_prev": float(lambda2_delta),
            "changed_adjacency_entries_from_prev": changed_entries,
            "changed_edges_from_prev": changed_edges,
            "rolling_edge_change_5": rolling_edge_change_5,
            "steps_since_major_reconfiguration": steps_since_major_change,
            "top_bottleneck_links": top_bottleneck_links,
            "bottleneck_top3_shortest_path_share": bottleneck_top3_share,
        })
        previous_lambda2 = lambda2
        previous_adj = adjacency

    return stats


def render_endogenous_text(stats):
    texts = []
    for item in stats:
        connected_text = (
            "connected" if item["is_connected"] else "not fully connected"
        )
        if item["long_path_ratio_gt8"] >= 0.25:
            path_pressure = "high"
        elif item["long_path_ratio_gt8"] >= 0.10:
            path_pressure = "moderate"
        else:
            path_pressure = "low"

        if item["algebraic_connectivity_lambda2"] >= 0.40:
            redundancy = "strong"
        elif item["algebraic_connectivity_lambda2"] >= 0.20:
            redundancy = "moderate"
        else:
            redundancy = "weak"

        if item["rolling_edge_change_5"] >= 4.0:
            change_state = "active"
        elif item["rolling_edge_change_5"] >= 1.0:
            change_state = "mild"
        else:
            change_state = "stable"

        link_text = "none"
        if item["top_bottleneck_links"]:
            link_text = ", ".join([
                "sat-%d to sat-%d" % (
                    link["source"],
                    link["destination"],
                )
                for link in item["top_bottleneck_links"][:3]
            ])

        text = (
            "At time {time_index}, normalized phase {phase:.3f}, the LEO "
            "topology has {num_satellites} satellites and "
            "{edge_count_undirected} undirected ISLs. It is {connected}; "
            "mean shortest path is {avg_shortest_path_hops:.2f} hops, "
            "diameter is {diameter_hops}, and long-path pressure is "
            "{path_pressure}. Algebraic connectivity lambda2 is "
            "{lambda2:.4f}, indicating {redundancy} redundancy "
            "({lambda2_delta:+.4f} from the previous step). The main "
            "bottleneck links are {link_text}, covering {bottleneck_share:.2%} "
            "of shortest-path edge usage. Recent topology change is "
            "{change_state}, with a 5-step average of {rolling_change:.2f} "
            "changed links."
        ).format(
            time_index=item["time_index"],
            phase=item["normalized_time_phase"],
            num_satellites=item["num_satellites"],
            edge_count_undirected=item["edge_count_undirected"],
            connected=connected_text,
            avg_shortest_path_hops=item["avg_shortest_path_hops"],
            diameter_hops=item["diameter_hops"],
            path_pressure=path_pressure,
            lambda2=item["algebraic_connectivity_lambda2"],
            lambda2_delta=item["lambda2_delta_from_prev"],
            redundancy=redundancy,
            link_text=link_text,
            bottleneck_share=item["bottleneck_top3_shortest_path_share"],
            change_state=change_state,
            rolling_change=item["rolling_edge_change_5"],
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
        "--endo-output-path",
        default=None,
        help="Path for template endogenous text JSON. Defaults to output-dir/endo_texts.json.",
    )
    parser.add_argument(
        "--write-template-endo",
        action="store_true",
        help="Also write endo_texts_topo_template.json as a debug artifact.",
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
    endo_output_path = args.endo_output_path
    if endo_output_path is None:
        endo_output_path = os.path.join(output_dir, "endo_texts.json")
    write_json(
        endo_output_path,
        {
            "metadata": {
                "source": "topology_statistics_template",
                "text_generation_mode": "template",
                "num_time_slices": int(topology.shape[0]),
                "template_policy": (
                    "Uniform endogenous text template from structured "
                    "topology features. No traffic values, split masks, "
                    "validation entries, or test entries are included."
                ),
                "template_features": [
                    "normalized_time_phase",
                    "edge_count_undirected",
                    "avg_shortest_path_hops",
                    "diameter_hops",
                    "long_path_ratio_gt8",
                    "algebraic_connectivity_lambda2",
                    "lambda2_delta_from_prev",
                    "top_bottleneck_links",
                    "bottleneck_top3_shortest_path_share",
                    "rolling_edge_change_5",
                ],
            },
            "texts": render_endogenous_text(stats),
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
    print("Template endogenous path:", endo_output_path)
    print("Debug template endogenous written:", bool(args.write_template_endo))
    print("Template exogenous written:", bool(args.write_template_exo))


if __name__ == "__main__":
    main()
