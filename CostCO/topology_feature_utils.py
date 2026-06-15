from collections import deque

import numpy as np


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

    node_bc *= 0.5
    for edge in list(edge_bc):
        edge_bc[edge] *= 0.5
    return node_bc.astype("float32"), edge_bc


def edge_set(adj):
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    return set(zip(rows.tolist(), cols.tolist()))


def normalize_features(values):
    flat = values.reshape(-1, values.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype("float32"), mean.astype("float32"), std.astype("float32")
