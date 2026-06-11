import argparse
import os
from collections import deque

import numpy as np


NODE_FEATURE_NAMES = np.array([
    "normalized_degree",
    "inter_plane_degree_ratio",
    "two_hop_neighborhood_ratio",
    "local_clustering_coefficient",
    "closeness_centrality",
    "node_betweenness_centrality",
    "reachable_node_ratio",
    "mean_shortest_path_hops",
    "bottleneck_path_exposure_ratio",
    "neighbor_change_rate",
])

TIME_FEATURE_NAMES = np.array([
    "edge_density",
    "average_degree",
    "degree_std",
    "inter_plane_edge_ratio",
    "reachable_od_ratio",
    "mean_shortest_path_hops",
    "p90_shortest_path_hops",
    "algebraic_connectivity",
    "edge_change_rate",
    "bottleneck_concentration",
])


def load_connectivity_tensor(path, time_len, node_count):
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ("sat_connectivity", "arr_0", "connectivity", "adjacency"):
            if key in data:
                topology = data[key]
                break
        else:
            raise KeyError(
                "Could not find a connectivity array in %s. Available keys: %s" %
                (path, list(data.keys()))
            )
    else:
        topology = data

    topology = np.asarray(topology)
    if topology.shape == (time_len, node_count, node_count):
        return topology.astype("float32")
    if topology.shape == (node_count, node_count, time_len):
        return np.transpose(topology, (2, 0, 1)).astype("float32")
    raise ValueError(
        "Topology shape %s is incompatible with expected [T,N,N]=%s or [N,N,T]=%s" %
        (
            topology.shape,
            (time_len, node_count, node_count),
            (node_count, node_count, time_len),
        )
    )


def make_plane_ids(node_count, planes):
    if planes <= 0 or node_count % planes != 0:
        raise ValueError("node_count must be divisible by --planes")
    sats_per_plane = node_count // planes
    return np.repeat(np.arange(planes), sats_per_plane)


def bfs_distances(adj, start):
    node_count = adj.shape[0]
    dist = np.full(node_count, np.inf, dtype="float32")
    dist[start] = 0.0
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in np.flatnonzero(adj[node]):
            if np.isinf(dist[nbr]):
                dist[nbr] = dist[node] + 1.0
                queue.append(int(nbr))
    return dist


def all_pairs_shortest_paths(adj):
    return np.stack([bfs_distances(adj, node) for node in range(adj.shape[0])])


def brandes_node_edge_betweenness(adj):
    node_count = adj.shape[0]
    node_bc = np.zeros(node_count, dtype="float64")
    edge_bc = {}
    neighbors = [np.flatnonzero(adj[node]).astype(int) for node in range(node_count)]

    for source in range(node_count):
        stack = []
        pred = [[] for _ in range(node_count)]
        sigma = np.zeros(node_count, dtype="float64")
        dist = np.full(node_count, -1, dtype="int32")
        sigma[source] = 1.0
        dist[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            stack.append(node)
            for nbr in neighbors[node]:
                if dist[nbr] < 0:
                    queue.append(int(nbr))
                    dist[nbr] = dist[node] + 1
                if dist[nbr] == dist[node] + 1:
                    sigma[nbr] += sigma[node]
                    pred[nbr].append(node)

        delta = np.zeros(node_count, dtype="float64")
        while stack:
            node = stack.pop()
            coeff = (1.0 + delta[node]) / max(sigma[node], 1e-12)
            for parent in pred[node]:
                contrib = sigma[parent] * coeff
                edge = (parent, node) if parent < node else (node, parent)
                edge_bc[edge] = edge_bc.get(edge, 0.0) + contrib
                delta[parent] += contrib
            if node != source:
                node_bc[node] += delta[node]

    # Undirected graph: each shortest path was counted from both endpoints.
    node_bc *= 0.5
    for edge in list(edge_bc):
        edge_bc[edge] *= 0.5
    return node_bc.astype("float32"), edge_bc


def local_clustering(adj):
    node_count = adj.shape[0]
    clustering = np.zeros(node_count, dtype="float32")
    for node in range(node_count):
        nbrs = np.flatnonzero(adj[node])
        degree = nbrs.size
        if degree < 2:
            continue
        subgraph = adj[np.ix_(nbrs, nbrs)]
        links = np.sum(subgraph) / 2.0
        clustering[node] = (2.0 * links) / (degree * (degree - 1))
    return clustering


def edge_set(adj):
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    return set(zip(rows.tolist(), cols.tolist()))


def normalize_features(values):
    mean = values.reshape(-1, values.shape[-1]).mean(axis=0)
    std = values.reshape(-1, values.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype("float32"), mean.astype("float32"), std.astype("float32")


def build_features(topology, planes=10, top_bottleneck_ratio=0.1):
    topo = (np.asarray(topology) > 0).astype("int8")
    time_len, node_count, _ = topo.shape
    plane_ids = make_plane_ids(node_count, planes)
    node_features = np.zeros((time_len, node_count, len(NODE_FEATURE_NAMES)), dtype="float32")
    time_features = np.zeros((time_len, len(TIME_FEATURE_NAMES)), dtype="float32")
    prev_adj = None
    prev_edges = set()
    eps = 1e-8

    for time_idx in range(time_len):
        adj = topo[time_idx].copy()
        np.fill_diagonal(adj, 0)
        degrees = adj.sum(axis=1).astype("float32")
        edges = edge_set(adj)
        edge_count = len(edges)
        dist = all_pairs_shortest_paths(adj)
        finite = np.isfinite(dist)
        non_diag = ~np.eye(node_count, dtype=bool)
        finite_non_diag = finite & non_diag
        finite_dists = dist[finite_non_diag]

        node_bc, edge_bc = brandes_node_edge_betweenness(adj)
        clustering = local_clustering(adj)
        if edge_bc:
            sorted_edges = sorted(edge_bc.items(), key=lambda item: item[1], reverse=True)
            top_count = max(1, int(np.ceil(len(sorted_edges) * top_bottleneck_ratio)))
            bottleneck_edges = {edge for edge, _ in sorted_edges[:top_count]}
            total_edge_bc = sum(edge_bc.values())
            top_edge_bc = sum(value for _, value in sorted_edges[:top_count])
        else:
            bottleneck_edges = set()
            total_edge_bc = 0.0
            top_edge_bc = 0.0

        inter_edge_count = 0
        for src, dst in edges:
            if plane_ids[src] != plane_ids[dst]:
                inter_edge_count += 1

        for node in range(node_count):
            nbrs = np.flatnonzero(adj[node])
            degree = degrees[node]
            if degree > 0:
                inter_degree = np.sum(plane_ids[nbrs] != plane_ids[node])
                inter_degree_ratio = inter_degree / degree
            else:
                inter_degree_ratio = 0.0

            two_hop = np.sum(dist[node] == 2.0)
            reachable = finite[node] & (np.arange(node_count) != node)
            reachable_count = np.sum(reachable)
            reachable_dists = dist[node, reachable]
            if reachable_dists.size > 0:
                mean_hops = np.mean(reachable_dists)
                closeness = reachable_count / max(np.sum(reachable_dists), eps)
            else:
                mean_hops = 0.0
                closeness = 0.0

            bottleneck_exposed = 0
            if bottleneck_edges:
                for dst in range(node_count):
                    if dst == node or not np.isfinite(dist[node, dst]):
                        continue
                    on_bottleneck = False
                    for u, v in bottleneck_edges:
                        if (
                            dist[node, u] + 1.0 + dist[v, dst] == dist[node, dst] or
                            dist[node, v] + 1.0 + dist[u, dst] == dist[node, dst]
                        ):
                            on_bottleneck = True
                            break
                    bottleneck_exposed += int(on_bottleneck)

            if prev_adj is None:
                neighbor_change = 0.0
            else:
                prev_nbrs = set(np.flatnonzero(prev_adj[node]).tolist())
                cur_nbrs = set(nbrs.tolist())
                union = len(prev_nbrs | cur_nbrs)
                inter = len(prev_nbrs & cur_nbrs)
                neighbor_change = 1.0 - (inter / max(union, eps))

            node_features[time_idx, node] = np.array([
                degree / max(node_count - 1, 1),
                inter_degree_ratio,
                two_hop / max(node_count - 1, 1),
                clustering[node],
                closeness,
                node_bc[node],
                reachable_count / max(node_count - 1, 1),
                mean_hops / max(node_count - 1, 1),
                bottleneck_exposed / max(node_count - 1, 1),
                neighbor_change,
            ], dtype="float32")

        if finite_dists.size > 0:
            mean_shortest = float(np.mean(finite_dists))
            p90_shortest = float(np.percentile(finite_dists, 90))
            reachable_od_ratio = finite_dists.size / (node_count * (node_count - 1))
        else:
            mean_shortest = 0.0
            p90_shortest = 0.0
            reachable_od_ratio = 0.0

        if prev_adj is None:
            edge_change = 0.0
        else:
            union = len(edges | prev_edges)
            diff = len(edges ^ prev_edges)
            edge_change = diff / max(union, eps)

        laplacian = np.diag(degrees) - adj.astype("float32")
        eigvals = np.linalg.eigvalsh(laplacian)
        lambda2 = float(eigvals[1]) if eigvals.size > 1 else 0.0

        time_features[time_idx] = np.array([
            (2.0 * edge_count) / max(node_count * (node_count - 1), 1),
            np.mean(degrees) / max(node_count - 1, 1),
            np.std(degrees) / max(node_count - 1, 1),
            inter_edge_count / max(edge_count, 1),
            reachable_od_ratio,
            mean_shortest / max(node_count - 1, 1),
            p90_shortest / max(node_count - 1, 1),
            lambda2 / max(node_count, 1),
            edge_change,
            top_edge_bc / max(total_edge_bc, eps),
        ], dtype="float32")

        prev_adj = adj
        prev_edges = edges

    node_features, node_mean, node_std = normalize_features(node_features)
    time_features, time_mean, time_std = normalize_features(time_features)
    return node_features, time_features, node_mean, node_std, time_mean, time_std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build mode-wise structural features from dynamic topology."
    )
    parser.add_argument(
        "--topology-path",
        default="sat_connectivity_tensor_dynamic_60s_1000ms.npz",
    )
    parser.add_argument(
        "--output-path",
        default="mode_struct_features.npz",
    )
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
    features = build_features(
        topology,
        planes=args.planes,
        top_bottleneck_ratio=args.top_bottleneck_ratio,
    )
    node_features, time_features, node_mean, node_std, time_mean, time_std = features
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez_compressed(
        args.output_path,
        node_features=node_features,
        time_features=time_features,
        node_feature_names=NODE_FEATURE_NAMES,
        time_feature_names=TIME_FEATURE_NAMES,
        node_normalization_mean=node_mean,
        node_normalization_std=node_std,
        time_normalization_mean=time_mean,
        time_normalization_std=time_std,
    )
    print("Saved mode structural features to:", args.output_path)
    print("Node features:", node_features.shape)
    print("Time features:", time_features.shape)


if __name__ == "__main__":
    main()
