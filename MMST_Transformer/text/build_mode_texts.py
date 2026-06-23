import argparse
import json
import os

import numpy as np

from topology_feature_utils import (
    all_pairs_shortest_paths,
    brandes_node_edge_betweenness,
    edge_set,
    load_connectivity_tensor,
    make_plane_ids,
)


EPS = 1e-8


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def level_by_quantiles(value, low_q, high_q, labels=("low", "medium", "high")):
    if value <= low_q:
        return labels[0]
    if value >= high_q:
        return labels[2]
    return labels[1]


def trend_label(series, window=5, threshold=0.05):
    ratio = trend_ratio(series, window=window)
    if ratio < -threshold:
        return "decreasing"
    if ratio > threshold:
        return "increasing"
    return "stable"


def trend_ratio(series, window=5):
    if series.size < 2:
        return 0.0
    head = float(np.mean(series[:min(window, series.size)]))
    tail = float(np.mean(series[-min(window, series.size):]))
    return (tail - head) / max(abs(head), EPS)


def normalized_entropy(weights):
    weights = np.asarray(weights, dtype="float64")
    total = np.sum(weights)
    if total <= EPS:
        return 0.0
    prob = weights / total
    prob = prob[prob > 0]
    if prob.size <= 1:
        return 0.0
    return float(-np.sum(prob * np.log(prob + EPS)) / np.log(weights.size))


def diversity_label(value):
    if value < 0.33:
        return "concentrated"
    if value > 0.66:
        return "diverse"
    return "moderate"


def activity_label(value):
    if value < 0.25:
        return "sparse"
    if value > 0.60:
        return "widespread"
    return "moderate"


def change_label(value):
    if value < 0.05:
        return "stable"
    if value < 0.20:
        return "mildly changing"
    return "actively changing"


def zscore_label(value, low=-0.75, high=0.75,
                 labels=("below average", "near average", "above average")):
    if value <= low:
        return labels[0]
    if value >= high:
        return labels[2]
    return labels[1]


def topology_change_phase(z_value):
    if z_value <= -0.75:
        return "stable coverage phase"
    if z_value >= 1.25:
        return "highly volatile handoff phase"
    if z_value >= 0.75:
        return "active handoff phase"
    return "transitioning coverage phase"


def orbit_phase_label(phase):
    if phase < 0.25:
        return "early orbit phase"
    if phase < 0.50:
        return "mid-ascending orbit phase"
    if phase < 0.75:
        return "late orbit phase"
    return "wraparound orbit phase"


def zscore(value, values):
    values = np.asarray(values, dtype="float32")
    return float((value - np.mean(values)) / max(np.std(values), EPS))


def load_tensor(path):
    tensor = np.load(path).astype("float32")
    if tensor.ndim != 3:
        raise ValueError("traffic tensor must have shape [source,destination,time]")
    return tensor


def plane_distance(src, dst, plane_ids, plane_count):
    diff = abs(int(plane_ids[src]) - int(plane_ids[dst]))
    return min(diff, plane_count - diff)


def edge_change_rate(cur_edges, prev_edges):
    if prev_edges is None:
        return 0.0
    return len(cur_edges ^ prev_edges) / max(len(cur_edges | prev_edges), EPS)


def node_topology_roles(topology, planes):
    topo = (np.asarray(topology) > 0).astype("int8")
    time_len, node_count, _ = topo.shape
    plane_ids = make_plane_ids(node_count, planes)
    plane_count = int(np.max(plane_ids)) + 1
    closeness = np.zeros((time_len, node_count), dtype="float32")
    node_bc = np.zeros((time_len, node_count), dtype="float32")
    bottleneck_exposure = np.zeros((time_len, node_count), dtype="float32")
    neighbor_change = np.zeros((time_len, node_count), dtype="float32")
    time_path_mean = np.zeros(time_len, dtype="float32")
    time_path_p90 = np.zeros(time_len, dtype="float32")
    time_edge_density = np.zeros(time_len, dtype="float32")
    time_average_degree = np.zeros(time_len, dtype="float32")
    time_inter_plane_edge_ratio = np.zeros(time_len, dtype="float32")
    time_reachable_od_ratio = np.zeros(time_len, dtype="float32")
    time_bottleneck_conc = np.zeros(time_len, dtype="float32")
    time_cross_plane_pressure = np.zeros(time_len, dtype="float32")
    time_change = np.zeros(time_len, dtype="float32")
    prev_adj = None
    prev_edges = None

    for time_idx in range(time_len):
        adj = topo[time_idx].copy()
        np.fill_diagonal(adj, 0)
        dist = all_pairs_shortest_paths(adj)
        finite = np.isfinite(dist)
        non_diag = ~np.eye(node_count, dtype=bool)
        finite_non_diag = finite & non_diag
        finite_dists = dist[finite_non_diag]
        if finite_dists.size > 0:
            time_path_mean[time_idx] = float(np.mean(finite_dists))
            time_path_p90[time_idx] = float(np.percentile(finite_dists, 90))
            time_reachable_od_ratio[time_idx] = (
                finite_dists.size / max(node_count * (node_count - 1), 1)
            )

        bc, edge_bc = brandes_node_edge_betweenness(adj)
        node_bc[time_idx] = bc
        sorted_edges = sorted(edge_bc.items(), key=lambda item: item[1], reverse=True)
        if sorted_edges:
            top_count = max(1, int(np.ceil(len(sorted_edges) * 0.1)))
            top_edges = {edge for edge, _ in sorted_edges[:top_count]}
            total_bc = sum(edge_bc.values())
            time_bottleneck_conc[time_idx] = (
                sum(value for _, value in sorted_edges[:top_count]) /
                max(total_bc, EPS)
            )
        else:
            top_edges = set()

        cur_edges = edge_set(adj)
        edge_count = len(cur_edges)
        degrees = adj.sum(axis=1).astype("float32")
        inter_edges = sum(
            int(plane_ids[src] != plane_ids[dst])
            for src, dst in cur_edges
        )
        time_edge_density[time_idx] = (
            2.0 * edge_count / max(node_count * (node_count - 1), 1)
        )
        time_average_degree[time_idx] = (
            float(np.mean(degrees)) / max(node_count - 1, 1)
        )
        time_inter_plane_edge_ratio[time_idx] = inter_edges / max(edge_count, 1)
        time_change[time_idx] = edge_change_rate(cur_edges, prev_edges)
        inter_paths = 0
        finite_paths = 0

        for node in range(node_count):
            reachable = finite[node] & (np.arange(node_count) != node)
            dists = dist[node, reachable]
            if dists.size > 0:
                closeness[time_idx, node] = dists.size / max(np.sum(dists), EPS)
            exposed = 0
            for dst in range(node_count):
                if dst == node or not np.isfinite(dist[node, dst]):
                    continue
                on_bottleneck = False
                for u, v in top_edges:
                    if (
                        dist[node, u] + 1.0 + dist[v, dst] == dist[node, dst] or
                        dist[node, v] + 1.0 + dist[u, dst] == dist[node, dst]
                    ):
                        on_bottleneck = True
                        break
                exposed += int(on_bottleneck)
                finite_paths += 1
                inter_paths += int(plane_ids[node] != plane_ids[dst])
            bottleneck_exposure[time_idx, node] = exposed / max(node_count - 1, 1)

            if prev_adj is not None:
                prev_nbrs = set(np.flatnonzero(prev_adj[node]).tolist())
                cur_nbrs = set(np.flatnonzero(adj[node]).tolist())
                neighbor_change[time_idx, node] = (
                    1.0 - len(prev_nbrs & cur_nbrs) /
                    max(len(prev_nbrs | cur_nbrs), EPS)
                )

        time_cross_plane_pressure[time_idx] = inter_paths / max(finite_paths, 1)
        prev_adj = adj
        prev_edges = cur_edges

    return {
        "plane_ids": plane_ids,
        "plane_count": plane_count,
        "closeness": closeness,
        "node_betweenness": node_bc,
        "bottleneck_exposure": bottleneck_exposure,
        "neighbor_change": neighbor_change,
        "time_path_mean": time_path_mean,
        "time_path_p90": time_path_p90,
        "time_edge_density": time_edge_density,
        "time_average_degree": time_average_degree,
        "time_inter_plane_edge_ratio": time_inter_plane_edge_ratio,
        "time_reachable_od_ratio": time_reachable_od_ratio,
        "time_bottleneck_concentration": time_bottleneck_conc,
        "time_cross_plane_pressure": time_cross_plane_pressure,
        "time_change": time_change,
    }


def build_records(tensor, topology, history_len=30, target_start=0,
                  target_end=None, planes=10):
    source_count, destination_count, time_len = tensor.shape
    if source_count != destination_count:
        raise ValueError("source and destination dimensions must match")
    if target_end is None:
        target_end = time_len
    history_len = min(history_len, time_len)
    history = tensor[:, :, :history_len]
    target_times = list(range(target_start, target_end))
    topo_stats = node_topology_roles(topology, planes)
    plane_ids = topo_stats["plane_ids"]
    plane_count = topo_stats["plane_count"]

    outbound = history.sum(axis=1).T
    inbound = history.sum(axis=0).T
    out_mean = outbound.mean(axis=0)
    in_mean = inbound.mean(axis=0)
    out_peak = np.percentile(outbound, 95, axis=0)
    in_peak = np.percentile(inbound, 95, axis=0)
    out_mean_q = np.quantile(out_mean, [0.33, 0.66])
    in_mean_q = np.quantile(in_mean, [0.33, 0.66])
    out_peak_q = np.quantile(out_peak, [0.33, 0.66])
    in_peak_q = np.quantile(in_peak, [0.33, 0.66])

    source_total_by_dst = history.sum(axis=2)
    dest_total_by_src = history.sum(axis=2)
    source_diversity = np.array([
        normalized_entropy(source_total_by_dst[src])
        for src in range(source_count)
    ])
    dest_diversity = np.array([
        normalized_entropy(dest_total_by_src[:, dst])
        for dst in range(destination_count)
    ])

    cross_out = np.zeros(source_count, dtype="float32")
    cross_in = np.zeros(destination_count, dtype="float32")
    for src in range(source_count):
        mask = plane_ids != plane_ids[src]
        cross_out[src] = (
            source_total_by_dst[src, mask].sum() /
            max(source_total_by_dst[src].sum(), EPS)
        )
    for dst in range(destination_count):
        mask = plane_ids != plane_ids[dst]
        cross_in[dst] = (
            dest_total_by_src[mask, dst].sum() /
            max(dest_total_by_src[:, dst].sum(), EPS)
        )
    cross_out_q = np.quantile(cross_out, [0.33, 0.66])
    cross_in_q = np.quantile(cross_in, [0.33, 0.66])

    path_q = np.quantile(topo_stats["time_path_mean"], [0.33, 0.66])
    path_p90_q = np.quantile(topo_stats["time_path_p90"], [0.33, 0.66])
    inter_edge_q = np.quantile(
        topo_stats["time_inter_plane_edge_ratio"],
        [0.33, 0.66],
    )
    reachable_q = np.quantile(topo_stats["time_reachable_od_ratio"], [0.33, 0.66])
    bottle_q = np.quantile(
        topo_stats["time_bottleneck_concentration"],
        [0.33, 0.66],
    )
    cross_path_q = np.quantile(
        topo_stats["time_cross_plane_pressure"],
        [0.33, 0.66],
    )
    source_records = []
    destination_records = []
    time_records = []

    for time_idx in target_times:
        change_z = zscore(topo_stats["time_change"][time_idx],
                          topo_stats["time_change"])
        path_mean_z = zscore(topo_stats["time_path_mean"][time_idx],
                             topo_stats["time_path_mean"])
        bottleneck_z = zscore(
            topo_stats["time_bottleneck_concentration"][time_idx],
            topo_stats["time_bottleneck_concentration"],
        )
        cross_plane_z = zscore(
            topo_stats["time_cross_plane_pressure"][time_idx],
            topo_stats["time_cross_plane_pressure"],
        )
        orbit_phase_ratio = (
            (time_idx - target_start) /
            max(float(max(target_end - target_start, 1)), 1.0)
        )
        time_state = {
            "time_index": time_idx,
            "average_degree": float(topo_stats["time_average_degree"][time_idx]),
            "average_degree_level": level_by_quantiles(
                topo_stats["time_average_degree"][time_idx],
                np.quantile(topo_stats["time_average_degree"], 0.33),
                np.quantile(topo_stats["time_average_degree"], 0.66),
                labels=("low-degree", "moderate-degree", "high-degree"),
            ),
            "inter_plane_edge_level": level_by_quantiles(
                topo_stats["time_inter_plane_edge_ratio"][time_idx],
                inter_edge_q[0],
                inter_edge_q[1],
                labels=("low", "moderate", "high"),
            ),
            "inter_plane_edge_ratio": float(
                topo_stats["time_inter_plane_edge_ratio"][time_idx]
            ),
            "reachable_od_level": level_by_quantiles(
                topo_stats["time_reachable_od_ratio"][time_idx],
                reachable_q[0],
                reachable_q[1],
                labels=("limited", "moderate", "broad"),
            ),
            "reachable_od_ratio": float(topo_stats["time_reachable_od_ratio"][time_idx]),
            "topology_change_state": change_label(
                topo_stats["time_change"][time_idx]
            ),
            "topology_change_rate": float(topo_stats["time_change"][time_idx]),
            "topology_change_zscore": change_z,
            "topology_change_relative": zscore_label(change_z),
            "handoff_phase": topology_change_phase(change_z),
            "path_length_level": level_by_quantiles(
                topo_stats["time_path_mean"][time_idx],
                path_q[0],
                path_q[1],
                labels=("short", "moderate", "long"),
            ),
            "mean_shortest_path_hops": float(topo_stats["time_path_mean"][time_idx]),
            "mean_shortest_path_zscore": path_mean_z,
            "path_length_relative": zscore_label(path_mean_z),
            "p90_path_level": level_by_quantiles(
                topo_stats["time_path_p90"][time_idx],
                path_p90_q[0],
                path_p90_q[1],
                labels=("short-tail", "moderate-tail", "long-tail"),
            ),
            "p90_shortest_path_hops": float(topo_stats["time_path_p90"][time_idx]),
            "bottleneck_level": level_by_quantiles(
                topo_stats["time_bottleneck_concentration"][time_idx],
                bottle_q[0],
                bottle_q[1],
                labels=("diffuse", "moderate", "concentrated"),
            ),
            "bottleneck_concentration": float(
                topo_stats["time_bottleneck_concentration"][time_idx]
            ),
            "bottleneck_zscore": bottleneck_z,
            "bottleneck_relative": zscore_label(bottleneck_z),
            "cross_plane_pressure_level": level_by_quantiles(
                topo_stats["time_cross_plane_pressure"][time_idx],
                cross_path_q[0],
                cross_path_q[1],
                labels=("low", "moderate", "high"),
            ),
            "cross_plane_path_pressure": float(
                topo_stats["time_cross_plane_pressure"][time_idx]
            ),
            "cross_plane_path_pressure_zscore": cross_plane_z,
            "cross_plane_pressure_relative": zscore_label(cross_plane_z),
            "orbit_phase_ratio": float(orbit_phase_ratio),
            "orbit_phase_label": orbit_phase_label(orbit_phase_ratio),
        }
        time_state["text"] = (
            "The satellite network is in a {average_degree_level}, "
            "{handoff_phase}. Inter-plane connectivity is "
            "{inter_plane_edge_level}, and OD reachability is "
            "{reachable_od_level}. Topology change is "
            "{topology_change_relative} for this slice. Routing paths are "
            "{path_length_level} with a {p90_path_level} tail profile, and "
            "the path length is {path_length_relative}. Potential structural "
            "bottlenecks are {bottleneck_level} and {bottleneck_relative}. "
            "Cross-plane shortest-path pressure is {cross_plane_pressure_level} "
            "and {cross_plane_pressure_relative}. The slice is in the "
            "{orbit_phase_label}."
        ).format(**time_state)
        time_records.append(time_state)

        for sat in range(source_count):
            record = {
                "time_index": time_idx,
                "satellite_id": sat,
                "mean_out_level": level_by_quantiles(
                    out_mean[sat],
                    out_mean_q[0],
                    out_mean_q[1],
                ),
                "mean_out_load": float(out_mean[sat]),
                "peak_out_level": level_by_quantiles(
                    out_peak[sat],
                    out_peak_q[0],
                    out_peak_q[1],
                    labels=("low", "moderate", "high"),
                ),
                "peak_out_load": float(out_peak[sat]),
                "out_trend": trend_label(outbound[:, sat]),
                "out_trend_ratio": float(trend_ratio(outbound[:, sat])),
                "destination_diversity": diversity_label(source_diversity[sat]),
                "destination_entropy": float(source_diversity[sat]),
                "cross_plane_out_level": level_by_quantiles(
                    cross_out[sat],
                    cross_out_q[0],
                    cross_out_q[1],
                    labels=("low", "moderate", "high"),
                ),
                "cross_plane_out_ratio": float(cross_out[sat]),
                "source_closeness": float(topo_stats["closeness"][time_idx, sat]),
                "source_bottleneck_exposure": float(
                    topo_stats["bottleneck_exposure"][time_idx, sat]
                ),
                "source_neighbor_change": float(
                    topo_stats["neighbor_change"][time_idx, sat]
                ),
            }
            record["text"] = (
                "This satellite has a {mean_out_level} historical outbound "
                "load and a {peak_out_level} peak load. Its outbound traffic "
                "trend is {out_trend}. Its destination distribution is "
                "{destination_diversity}, with a {cross_plane_out_level} "
                "cross-plane sending tendency. As a source, it mainly "
                "represents a historical sender role rather than a current "
                "topology state."
            ).format(**record)
            source_records.append(record)

            dest_record = {
                "time_index": time_idx,
                "satellite_id": sat,
                "mean_in_level": level_by_quantiles(
                    in_mean[sat],
                    in_mean_q[0],
                    in_mean_q[1],
                ),
                "mean_in_load": float(in_mean[sat]),
                "peak_in_level": level_by_quantiles(
                    in_peak[sat],
                    in_peak_q[0],
                    in_peak_q[1],
                    labels=("low", "moderate", "high"),
                ),
                "peak_in_load": float(in_peak[sat]),
                "in_trend": trend_label(inbound[:, sat]),
                "in_trend_ratio": float(trend_ratio(inbound[:, sat])),
                "source_diversity": diversity_label(dest_diversity[sat]),
                "source_entropy": float(dest_diversity[sat]),
                "cross_plane_in_level": level_by_quantiles(
                    cross_in[sat],
                    cross_in_q[0],
                    cross_in_q[1],
                    labels=("low", "moderate", "high"),
                ),
                "cross_plane_in_ratio": float(cross_in[sat]),
                "destination_closeness": float(
                    topo_stats["closeness"][time_idx, sat]
                ),
                "destination_bottleneck_exposure": float(
                    topo_stats["bottleneck_exposure"][time_idx, sat]
                ),
                "destination_neighbor_change": float(
                    topo_stats["neighbor_change"][time_idx, sat]
                ),
            }
            dest_record["text"] = (
                "This satellite has a {mean_in_level} historical inbound "
                "load and a {peak_in_level} peak load. Its inbound traffic "
                "trend is {in_trend}. Its source distribution is "
                "{source_diversity}, with a {cross_plane_in_level} "
                "cross-plane receiving tendency. As a destination, it mainly "
                "represents a historical receiver role rather than a current "
                "topology state."
            ).format(**dest_record)
            destination_records.append(dest_record)

    metadata = {
        "history_len": history_len,
        "target_start": target_start,
        "target_end": target_end,
        "target_times": target_times,
        "node_count": source_count,
        "planes": planes,
        "text_generation": (
            "static_source_destination_text_dynamic_time_text_with_numeric_side_channel"
        ),
        "source_destination_text_granularity": "static_node_repeated_by_time",
        "time_text_granularity": "dynamic_time_slice",
    }
    return source_records, destination_records, time_records, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic source/destination/time text records."
    )
    parser.add_argument("--tensor-path", default="sat_path_bytes_mb_tensor.npy")
    parser.add_argument(
        "--topology-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument("--output-dir", default="mode_text_data")
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=None)
    parser.add_argument("--planes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    tensor = load_tensor(args.tensor_path)
    topology = load_connectivity_tensor(
        args.topology_path,
        tensor.shape[2],
        tensor.shape[0],
    )
    source_records, destination_records, time_records, metadata = build_records(
        tensor,
        topology,
        history_len=args.history_len,
        target_start=args.target_start,
        target_end=args.target_end,
        planes=args.planes,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    write_json(os.path.join(args.output_dir, "source_text_records.json"), {
        "metadata": metadata,
        "records": source_records,
    })
    write_json(os.path.join(args.output_dir, "destination_text_records.json"), {
        "metadata": metadata,
        "records": destination_records,
    })
    write_json(os.path.join(args.output_dir, "time_text_records.json"), {
        "metadata": metadata,
        "records": time_records,
    })
    print("Saved source records:", len(source_records))
    print("Saved destination records:", len(destination_records))
    print("Saved time records:", len(time_records))
    print("Output dir:", args.output_dir)


if __name__ == "__main__":
    main()
