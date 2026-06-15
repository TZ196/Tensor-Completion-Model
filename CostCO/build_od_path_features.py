import argparse
import os
from collections import deque

import numpy as np

from build_mode_struct_features import (
    brandes_node_edge_betweenness,
    load_connectivity_tensor,
    make_plane_ids,
    normalize_features,
)


OD_PATH_FEATURE_NAMES = np.array([
    "reachable",
    "shortest_path_hops",
    "direct_link",
    "two_hop_path",
    "endpoint_inter_plane",
    "min_cross_plane_edge_ratio",
    "bottleneck_edge_ratio",
])


def shortest_paths_with_cross_counts(adj, plane_ids, source):
    node_count = adj.shape[0]
    dist = np.full(node_count, np.inf, dtype="float32")
    cross_count = np.full(node_count, np.inf, dtype="float32")
    dist[source] = 0.0
    cross_count[source] = 0.0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for nbr in np.flatnonzero(adj[node]):
            edge_cross = float(plane_ids[node] != plane_ids[nbr])
            next_dist = dist[node] + 1.0
            next_cross = cross_count[node] + edge_cross
            improves = (
                next_dist < dist[nbr] or
                (next_dist == dist[nbr] and next_cross < cross_count[nbr])
            )
            if improves:
                dist[nbr] = next_dist
                cross_count[nbr] = next_cross
                queue.append(int(nbr))

    return dist, cross_count


def all_pairs_shortest_paths_with_cross_counts(adj, plane_ids):
    distances = []
    cross_counts = []
    for source in range(adj.shape[0]):
        dist, cross_count = shortest_paths_with_cross_counts(
            adj,
            plane_ids,
            source,
        )
        distances.append(dist)
        cross_counts.append(cross_count)
    return np.stack(distances), np.stack(cross_counts)


def edge_set(adj):
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    return set(zip(rows.tolist(), cols.tolist()))


def edge_on_shortest_path(dist, src, dst, edge):
    if not np.isfinite(dist[src, dst]) or dist[src, dst] <= 0.0:
        return False
    u, v = edge
    path_len = dist[src, dst]
    return (
        dist[src, u] + 1.0 + dist[v, dst] == path_len or
        dist[src, v] + 1.0 + dist[u, dst] == path_len
    )


def build_od_path_features(topology, planes=10, top_bottleneck_ratio=0.1):
    topo = (np.asarray(topology) > 0).astype("int8")
    time_len, node_count, _ = topo.shape
    plane_ids = make_plane_ids(node_count, planes)
    features = np.zeros(
        (time_len, node_count, node_count, len(OD_PATH_FEATURE_NAMES)),
        dtype="float32",
    )
    eps = 1e-8

    for time_idx in range(time_len):
        adj = topo[time_idx].copy()
        np.fill_diagonal(adj, 0)
        dist, cross_count = all_pairs_shortest_paths_with_cross_counts(
            adj,
            plane_ids,
        )

        _, edge_bc = brandes_node_edge_betweenness(adj)
        if edge_bc:
            sorted_edges = sorted(
                edge_bc.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top_count = max(1, int(np.ceil(len(sorted_edges) * top_bottleneck_ratio)))
            bottleneck_edges = [edge for edge, _ in sorted_edges[:top_count]]
        else:
            bottleneck_edges = []

        reachable = np.isfinite(dist)
        hops = np.where(reachable, dist, float(node_count - 1))
        direct_link = adj.astype("float32")
        two_hop = (hops == 2.0).astype("float32")

        endpoint_inter_plane = (
            plane_ids[:, None] != plane_ids[None, :]
        ).astype("float32")
        cross_ratio = np.zeros((node_count, node_count), dtype="float32")
        valid_path = reachable & (hops > 0.0)
        cross_ratio[valid_path] = (
            cross_count[valid_path] / np.maximum(hops[valid_path], eps)
        )

        bottleneck_ratio = np.zeros((node_count, node_count), dtype="float32")
        for src in range(node_count):
            for dst in range(node_count):
                if not valid_path[src, dst] or not bottleneck_edges:
                    continue
                exposed = sum(
                    1 for edge in bottleneck_edges
                    if edge_on_shortest_path(dist, src, dst, edge)
                )
                bottleneck_ratio[src, dst] = exposed / max(hops[src, dst], eps)

        features[time_idx] = np.stack(
            [
                reachable.astype("float32"),
                hops / max(node_count - 1, 1),
                direct_link,
                two_hop,
                endpoint_inter_plane,
                cross_ratio,
                bottleneck_ratio,
            ],
            axis=-1,
        )

    normalized, mean, std = normalize_features(features)
    return normalized, features, mean, std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build OD-level path features from dynamic topology."
    )
    parser.add_argument(
        "--topology-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument("--output-path", default="mode_od_path_features.npz")
    parser.add_argument("--planes", type=int, default=10)
    parser.add_argument("--node-count", type=int, default=120)
    parser.add_argument("--time-len", type=int, default=60)
    parser.add_argument("--top-bottleneck-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    topology = load_connectivity_tensor(
        args.topology_path,
        args.time_len,
        args.node_count,
    )
    features, raw_features, mean, std = build_od_path_features(
        topology,
        planes=args.planes,
        top_bottleneck_ratio=args.top_bottleneck_ratio,
    )
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez_compressed(
        args.output_path,
        od_path_features=features,
        od_path_raw_features=raw_features,
        od_path_feature_names=OD_PATH_FEATURE_NAMES,
        normalization_mean=mean,
        normalization_std=std,
    )
    print("Saved OD path features to:", args.output_path)
    print("OD path features:", features.shape)
    print("Feature names:", OD_PATH_FEATURE_NAMES.tolist())


if __name__ == "__main__":
    main()
