import numpy as np
import tensorflow as tf
from tensorflow import keras as k

from costco_model import mae, nmae, nrmse, rmse, transform_indices


class TemporalGCNPairLayer(k.layers.Layer):
    """Encode source/destination nodes with A_t selected by time index."""

    def __init__(self, topology, node_dim=32, gcn_dim=64, **kwargs):
        super().__init__(**kwargs)
        topo = np.asarray(topology, dtype="float32")
        if topo.ndim != 3:
            raise ValueError("topology must have shape [time, nodes, nodes]")

        topo = (topo > 0).astype("float32")
        time_len, node_count, _ = topo.shape
        eye = np.eye(node_count, dtype="float32")[None, :, :]
        topo = topo + eye
        degree = np.sum(topo, axis=-1)
        degree_inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
        topo_norm = (
            degree_inv_sqrt[:, :, None] *
            topo *
            degree_inv_sqrt[:, None, :]
        )

        self.time_len = int(time_len)
        self.node_count = int(node_count)
        self.node_dim = int(node_dim)
        self.gcn_dim = int(gcn_dim)
        self.topology = tf.constant(topo_norm, dtype=tf.float32)

    def build(self, input_shape):
        self.node_embeddings = self.add_weight(
            name="node_embeddings",
            shape=(self.node_count, self.node_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gcn_kernel_1 = self.add_weight(
            name="gcn_kernel_1",
            shape=(self.node_dim, self.gcn_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gcn_kernel_2 = self.add_weight(
            name="gcn_kernel_2",
            shape=(self.gcn_dim, self.gcn_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        src_input, dst_input, time_input = inputs
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)

        adjacency = tf.gather(self.topology, time)
        node_features = tf.broadcast_to(
            self.node_embeddings[None, :, :],
            [tf.shape(time)[0], self.node_count, self.node_dim],
        )

        hidden = tf.matmul(adjacency, node_features)
        hidden = tf.matmul(hidden, self.gcn_kernel_1)
        hidden = tf.nn.relu(hidden)
        hidden = tf.matmul(adjacency, hidden)
        hidden = tf.matmul(hidden, self.gcn_kernel_2)
        hidden = tf.nn.relu(hidden)

        batch_ids = tf.range(tf.shape(time)[0], dtype=tf.int32)
        src_hidden = tf.gather_nd(hidden, tf.stack([batch_ids, src], axis=1))
        dst_hidden = tf.gather_nd(hidden, tf.stack([batch_ids, dst], axis=1))
        pair_hidden = tf.concat(
            [
                src_hidden,
                dst_hidden,
                tf.abs(src_hidden - dst_hidden),
                src_hidden * dst_hidden,
            ],
            axis=1,
        )
        return pair_hidden

    def get_config(self):
        config = super().get_config()
        config.update({
            "node_dim": self.node_dim,
            "gcn_dim": self.gcn_dim,
            "time_len": self.time_len,
            "node_count": self.node_count,
        })
        return config


def create_gcn_costco(shape, topology, rank=50, nc=64, node_dim=32,
                      gcn_dim=64):
    """Create CoSTCo with a trainable temporal GCN topology branch.

    Inputs are still source, destination, and time indices. The GCN branch uses
    the time index to select A_t, propagates trainable satellite node embeddings
    through that adjacency matrix, and fuses the resulting OD-pair topology
    representation with the CoSTCo/KPI branch.
    """
    shape = list(shape)
    if len(shape) != 3:
        raise ValueError("GCN-CoSTCo expects a 3-D traffic tensor")

    inputs = [
        k.Input(shape=(1,), dtype="int32", name="source_input"),
        k.Input(shape=(1,), dtype="int32", name="destination_input"),
        k.Input(shape=(1,), dtype="int32", name="time_input"),
    ]
    embeds = [
        k.layers.Embedding(
            output_dim=rank,
            input_dim=shape[i],
            name="mode_%d_embedding" % i,
        )(inputs[i])
        for i in range(len(shape))
    ]

    kpi_x = k.layers.Concatenate(axis=1, name="kpi_mode_concat")(embeds)
    kpi_x = k.layers.Reshape(
        target_shape=(rank, len(shape), 1),
        name="kpi_mode_reshape",
    )(kpi_x)
    kpi_x = k.layers.Conv2D(
        nc,
        kernel_size=(1, len(shape)),
        activation="relu",
        padding="valid",
        name="kpi_conv_modes",
    )(kpi_x)
    kpi_x = k.layers.Conv2D(
        nc,
        kernel_size=(rank, 1),
        activation="relu",
        padding="valid",
        name="kpi_conv_rank",
    )(kpi_x)
    kpi_x = k.layers.Flatten(name="kpi_flatten")(kpi_x)
    kpi_x = k.layers.Dense(nc, activation="relu", name="kpi_dense")(kpi_x)

    gcn_pair = TemporalGCNPairLayer(
        topology=topology,
        node_dim=node_dim,
        gcn_dim=gcn_dim,
        name="temporal_gcn_pair",
    )(inputs)
    gcn_x = k.layers.Dense(nc, activation="relu", name="gcn_pair_dense_1")(
        gcn_pair
    )
    gcn_x = k.layers.Dense(nc, activation="relu", name="gcn_pair_dense_2")(
        gcn_x
    )

    fused = k.layers.Concatenate(name="kpi_gcn_fusion")([kpi_x, gcn_x])
    fused = k.layers.Dense(nc, activation="relu", name="fusion_dense_1")(
        fused
    )
    fused = k.layers.Dense(
        max(1, nc // 2),
        activation="relu",
        name="fusion_dense_2",
    )(fused)
    output = k.layers.Dense(1, activation="relu", name="traffic_output")(
        fused
    )
    return k.Model(inputs=inputs, outputs=output, name="GCN_CoSTCo")


def compile_gcn_costco(model, lr=1e-4):
    optimizer = k.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def evaluate_gcn_costco(model, indices, values, batch_size=1024, verbose=1,
                        target_scale=1.0):
    pred = model.predict(
        transform_indices(indices),
        batch_size=batch_size,
        verbose=verbose,
    ).flatten()
    pred = pred * target_scale
    pred = np.maximum(pred, 0.0)
    return {
        "rmse": float(rmse(values, pred)),
        "mae": float(mae(values, pred)),
        "nmae": float(nmae(values, pred)),
        "nrmse": float(nrmse(values, pred)),
        "y_true_min": float(np.min(values)),
        "y_true_max": float(np.max(values)),
        "y_true_mean": float(np.mean(values)),
        "y_pred_min": float(np.min(pred)),
        "y_pred_max": float(np.max(pred)),
        "y_pred_mean": float(np.mean(pred)),
    }
