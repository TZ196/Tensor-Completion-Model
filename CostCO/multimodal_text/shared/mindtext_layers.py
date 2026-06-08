import numpy as np
import tensorflow as tf
from tensorflow import keras as k


def _as_2d_float32(name, values):
    values = np.asarray(values, dtype="float32")
    if values.ndim != 2:
        raise ValueError("%s must have shape [items, dim]" % name)
    return values


class EndoExoTextLayer(k.layers.Layer):
    """Return time-level endogenous text plus pooled exogenous text.

    This is the Stage-1 text input used by CoSTCo+Text and GCN-CoSTCo+Text.
    The layer keeps both text sources, but the downstream model can treat them
    as one combined text modality.
    """

    def __init__(self, endo_embeddings, exo_embeddings, **kwargs):
        super().__init__(**kwargs)
        endo = _as_2d_float32("endo_embeddings", endo_embeddings)
        exo = _as_2d_float32("exo_embeddings", exo_embeddings)
        if endo.shape[1] != exo.shape[1]:
            raise ValueError(
                "endo/exo embedding dimensions must match, got %d and %d" %
                (endo.shape[1], exo.shape[1])
            )
        self.time_len = int(endo.shape[0])
        self.text_dim = int(endo.shape[1])
        self.exo_segments = int(exo.shape[0])
        self.endo_embeddings = tf.constant(endo, dtype=tf.float32)
        self.exo_pooled = tf.constant(np.mean(exo, axis=0), dtype=tf.float32)

    def call(self, time_input):
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        endo = tf.gather(self.endo_embeddings, time)
        exo = tf.broadcast_to(
            self.exo_pooled[None, :],
            [tf.shape(time)[0], self.text_dim],
        )
        return tf.concat([endo, exo], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "time_len": self.time_len,
            "text_dim": self.text_dim,
            "exo_segments": self.exo_segments,
        })
        return config


class MindTextFusionLayer(k.layers.Layer):
    """Fuse endogenous and exogenous text with staged MindTS-style options."""

    def __init__(self, endo_embeddings, exo_embeddings, stage="concat",
                 hidden_dim=128, condenser_alpha=0.5,
                 condenser_epsilon=0.05, condenser_loss_weight=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        endo = _as_2d_float32("endo_embeddings", endo_embeddings)
        exo = _as_2d_float32("exo_embeddings", exo_embeddings)
        if endo.shape[1] != exo.shape[1]:
            raise ValueError(
                "endo/exo embedding dimensions must match, got %d and %d" %
                (endo.shape[1], exo.shape[1])
            )
        if stage not in [
            "concat",
            "cross_attention",
            "semantic_gating",
            "segment_condenser",
        ]:
            raise ValueError("Unsupported text fusion stage: %s" % stage)

        self.stage = stage
        self.time_len = int(endo.shape[0])
        self.text_dim = int(endo.shape[1])
        self.exo_segments = int(exo.shape[0])
        self.hidden_dim = int(hidden_dim)
        self.condenser_alpha = float(condenser_alpha)
        self.condenser_epsilon = float(condenser_epsilon)
        self.condenser_loss_weight = float(condenser_loss_weight)
        self.endo_embeddings = tf.constant(endo, dtype=tf.float32)
        self.exo_embeddings = tf.constant(exo, dtype=tf.float32)
        self.exo_pooled = tf.constant(np.mean(exo, axis=0), dtype=tf.float32)

        self.query_dense = k.layers.Dense(hidden_dim, name="text_query")
        self.key_dense = k.layers.Dense(hidden_dim, name="text_key")
        self.value_dense = k.layers.Dense(hidden_dim, name="text_value")
        self.residual_dense = k.layers.Dense(hidden_dim, name="text_residual")
        self.ffn_1 = k.layers.Dense(hidden_dim, activation="relu",
                                    name="text_ffn_1")
        self.ffn_2 = k.layers.Dense(hidden_dim, name="text_ffn_2")
        self.gate_dense = k.layers.Dense(hidden_dim, activation="sigmoid",
                                         name="semantic_gate")
        self.segment_score = k.layers.Dense(1, name="segment_score")

    def _cross_attention(self, endo, exo_segments):
        query = self.query_dense(endo)
        keys = self.key_dense(exo_segments)
        values = self.value_dense(exo_segments)
        logits = tf.matmul(query[:, None, :], keys, transpose_b=True)
        logits = tf.squeeze(logits, axis=1)
        logits = logits / tf.sqrt(tf.cast(self.hidden_dim, tf.float32))
        weights = tf.nn.softmax(logits, axis=-1)
        attended = tf.matmul(weights[:, None, :], values)
        attended = tf.squeeze(attended, axis=1)
        residual = self.residual_dense(endo)
        x = residual + attended
        x = x + self.ffn_2(self.ffn_1(x))
        return x

    def _condense_exo(self, endo, exo_segments):
        batch = tf.shape(endo)[0]
        endo_expanded = tf.broadcast_to(
            endo[:, None, :],
            [batch, self.exo_segments, self.text_dim],
        )
        score_input = tf.concat([endo_expanded, exo_segments], axis=-1)
        logits = tf.squeeze(self.segment_score(score_input), axis=-1)
        weights = tf.nn.softmax(logits, axis=-1)

        eps = tf.cast(self.condenser_epsilon, tf.float32)
        uniform = tf.ones_like(weights) / tf.cast(self.exo_segments, tf.float32)
        smooth_weights = (1.0 - eps) * weights + eps * uniform
        condensed = exo_segments * smooth_weights[:, :, None]
        pooled = tf.reduce_sum(condensed, axis=1)
        original = tf.broadcast_to(
            self.exo_pooled[None, :],
            [batch, self.text_dim],
        )
        mixed = (
            (1.0 - self.condenser_alpha) * original +
            self.condenser_alpha * pooled
        )

        if self.condenser_loss_weight > 0.0:
            kl = tf.reduce_mean(
                tf.reduce_sum(
                    smooth_weights * tf.math.log(
                        (smooth_weights + 1e-8) / (uniform + 1e-8)
                    ),
                    axis=-1,
                )
            )
            self.add_loss(self.condenser_loss_weight * kl)
        return mixed

    def call(self, time_input):
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        endo = tf.gather(self.endo_embeddings, time)
        exo_segments = tf.broadcast_to(
            self.exo_embeddings[None, :, :],
            [tf.shape(time)[0], self.exo_segments, self.text_dim],
        )

        if self.stage == "concat":
            exo = tf.broadcast_to(
                self.exo_pooled[None, :],
                [tf.shape(time)[0], self.text_dim],
            )
            return tf.concat([endo, exo], axis=1)

        if self.stage == "segment_condenser":
            exo = self._condense_exo(endo, exo_segments)
            endo = tf.concat([endo, exo], axis=1)
            exo_segments = tf.broadcast_to(
                exo[:, None, :],
                [tf.shape(time)[0], 1, self.text_dim],
            )

        x = self._cross_attention(endo, exo_segments)
        if self.stage in ["semantic_gating", "segment_condenser"]:
            x = self.gate_dense(x) * x
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "stage": self.stage,
            "time_len": self.time_len,
            "text_dim": self.text_dim,
            "exo_segments": self.exo_segments,
            "hidden_dim": self.hidden_dim,
            "condenser_alpha": self.condenser_alpha,
            "condenser_epsilon": self.condenser_epsilon,
            "condenser_loss_weight": self.condenser_loss_weight,
        })
        return config


class TemporalSemanticAlignmentLayer(k.layers.Layer):
    """Add temporal InfoNCE losses between flow/graph/text representations."""

    def __init__(self, projection_dim=128, temperature=0.2,
                 temporal_delta=2, flow_text_weight=0.0,
                 graph_text_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = int(projection_dim)
        self.temperature = float(temperature)
        self.temporal_delta = int(temporal_delta)
        self.flow_text_weight = float(flow_text_weight)
        self.graph_text_weight = float(graph_text_weight)
        self.flow_proj = k.layers.Dense(projection_dim, name="align_flow")
        self.graph_proj = k.layers.Dense(projection_dim, name="align_graph")
        self.text_proj = k.layers.Dense(projection_dim, name="align_text")

    def _masked_infonce(self, left, right, times):
        left = tf.math.l2_normalize(left, axis=-1)
        right = tf.math.l2_normalize(right, axis=-1)
        logits = tf.matmul(left, right, transpose_b=True) / self.temperature
        time_dist = tf.abs(times[:, None] - times[None, :])
        near = time_dist <= self.temporal_delta
        labels = tf.range(tf.shape(times)[0], dtype=tf.int32)
        diagonal = tf.eye(tf.shape(times)[0], dtype=tf.bool)
        mask = tf.logical_and(near, tf.logical_not(diagonal))
        logits = tf.where(mask, tf.ones_like(logits) * -1e9, logits)
        loss_a = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        loss_b = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )
        return 0.5 * (tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b))

    def call(self, inputs):
        if len(inputs) == 3:
            flow_x, text_x, time_input = inputs
            graph_x = None
        else:
            flow_x, graph_x, text_x, time_input = inputs
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        unique_time, segment_ids = tf.unique(time)
        num_time = tf.shape(unique_time)[0]
        flow_time = tf.math.unsorted_segment_mean(
            flow_x, segment_ids, num_time
        )
        text_time = tf.math.unsorted_segment_mean(
            text_x, segment_ids, num_time
        )

        text_z = self.text_proj(text_time)
        if self.flow_text_weight > 0.0:
            flow_z = self.flow_proj(flow_time)
            self.add_loss(
                self.flow_text_weight *
                self._masked_infonce(flow_z, text_z, unique_time)
            )
        if graph_x is not None and self.graph_text_weight > 0.0:
            graph_time = tf.math.unsorted_segment_mean(
                graph_x, segment_ids, num_time
            )
            graph_z = self.graph_proj(graph_time)
            self.add_loss(
                self.graph_text_weight *
                self._masked_infonce(graph_z, text_z, unique_time)
            )
        return text_x

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
            "temporal_delta": self.temporal_delta,
            "flow_text_weight": self.flow_text_weight,
            "graph_text_weight": self.graph_text_weight,
        })
        return config


def create_text_projection(text_features, text_projection_dim, name_prefix):
    x = k.layers.Dense(
        text_projection_dim,
        activation="relu",
        name="%s_projection_dense_1" % name_prefix,
    )(text_features)
    x = k.layers.Dense(
        text_projection_dim,
        activation="relu",
        name="%s_projection_dense_2" % name_prefix,
    )(x)
    return x
