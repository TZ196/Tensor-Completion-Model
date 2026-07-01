import numpy as np
import tensorflow as tf
from tensorflow import keras as k


class TextInjectionLayer(k.layers.Layer):
    """Project text/numeric features to d_model and inject them into mode tokens."""

    def __init__(
        self,
        source_text_embeddings,
        destination_text_embeddings,
        time_text_embeddings,
        source_numeric_features=None,
        destination_numeric_features=None,
        time_numeric_features=None,
        d_model=64,
        hidden_dim=128,
        alpha_init=0.02,
        text_mode="text_numeric",
        align_dim=64,
        text_align_target_ratio=0.0,
        alignment_temperature=0.2,
        temporal_delta=2,
        text_align_sample_size=0,
        emit_text_metrics=True,
        target_start=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if text_mode not in ("text_only", "numeric_only", "text_numeric"):
            raise ValueError("text_mode must be text_only, numeric_only, or text_numeric")

        source = np.asarray(source_text_embeddings, dtype="float32")
        destination = np.asarray(destination_text_embeddings, dtype="float32")
        time = np.asarray(time_text_embeddings, dtype="float32")
        if source.ndim not in (2, 3):
            raise ValueError("source text must have shape [node,dim] or [time,node,dim]")
        if destination.shape != source.shape:
            raise ValueError("source/destination text embeddings must share shape")
        if time.ndim != 2:
            raise ValueError("time text embeddings must have shape [time,dim]")
        if source.ndim == 3 and source.shape[0] != time.shape[0]:
            raise ValueError("source/destination/time text lengths disagree")

        self.static_source_destination_text = source.ndim == 2
        self.text_time_len = int(time.shape[0])
        self.node_count = int(source.shape[0] if source.ndim == 2 else source.shape[1])
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.alpha_init = float(alpha_init)
        self.text_mode = text_mode
        self.align_dim = int(align_dim)
        self.text_align_target_ratio = float(text_align_target_ratio)
        self.text_align_enabled = self.text_align_target_ratio > 0.0 and text_mode != "numeric_only"
        self.initial_text_align_weight = 1e-5 if self.text_align_enabled else 0.0
        self.alignment_temperature = float(alignment_temperature)
        self.temporal_delta = int(temporal_delta)
        self.text_align_sample_size = int(text_align_sample_size)
        self.emit_text_metrics = bool(emit_text_metrics)
        self.target_start = int(target_start)

        self.source_text_embeddings = tf.constant(source, dtype=tf.float32)
        self.destination_text_embeddings = tf.constant(destination, dtype=tf.float32)
        self.time_text_embeddings = tf.constant(time, dtype=tf.float32)

        numeric_inputs = [
            source_numeric_features,
            destination_numeric_features,
            time_numeric_features,
        ]
        self.has_numeric = any(value is not None for value in numeric_inputs)
        if text_mode in ("numeric_only", "text_numeric") and not self.has_numeric:
            raise ValueError("%s requires numeric feature files" % text_mode)
        if self.has_numeric:
            if any(value is None for value in numeric_inputs):
                raise ValueError("source, destination, and time numeric files must be provided together")
            source_numeric = np.asarray(source_numeric_features, dtype="float32")
            destination_numeric = np.asarray(destination_numeric_features, dtype="float32")
            time_numeric = np.asarray(time_numeric_features, dtype="float32")
            if source_numeric.ndim != source.ndim or source_numeric.shape[:-1] != source.shape[:-1]:
                raise ValueError("source numeric prefix shape must match source text")
            if destination_numeric.ndim != destination.ndim or destination_numeric.shape[:-1] != destination.shape[:-1]:
                raise ValueError("destination numeric prefix shape must match destination text")
            if time_numeric.ndim != 2 or time_numeric.shape[0] != time.shape[0]:
                raise ValueError("time numeric must have shape [time,dim]")
            self.source_numeric_features = tf.constant(source_numeric, dtype=tf.float32)
            self.destination_numeric_features = tf.constant(destination_numeric, dtype=tf.float32)
            self.time_numeric_features = tf.constant(time_numeric, dtype=tf.float32)
        else:
            self.source_numeric_features = None
            self.destination_numeric_features = None
            self.time_numeric_features = None

        alpha_ratio = np.clip(self.alpha_init / 0.2, 1e-4, 1.0 - 1e-4)
        alpha_bias = float(np.log(alpha_ratio / (1.0 - alpha_ratio)))
        self.source_projector = self._make_projector("source_text_projector")
        self.destination_projector = self._make_projector("destination_text_projector")
        self.time_projector = self._make_projector("time_text_projector")
        self.source_gate = k.layers.Dense(d_model, activation="sigmoid", name="source_text_gate")
        self.destination_gate = k.layers.Dense(d_model, activation="sigmoid", name="destination_text_gate")
        self.time_gate = k.layers.Dense(d_model, activation="sigmoid", name="time_text_gate")
        self.source_numeric_gate = k.layers.Dense(d_model, activation="sigmoid", name="source_numeric_control_gate")
        self.destination_numeric_gate = k.layers.Dense(d_model, activation="sigmoid", name="destination_numeric_control_gate")
        self.time_numeric_gate = k.layers.Dense(d_model, activation="sigmoid", name="time_numeric_control_gate")
        self.source_alpha_gate = k.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer=k.initializers.Constant(alpha_bias),
            name="source_adaptive_text_alpha",
        )
        self.destination_alpha_gate = k.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer=k.initializers.Constant(alpha_bias),
            name="destination_adaptive_text_alpha",
        )
        self.time_alpha_gate = k.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer=k.initializers.Constant(alpha_bias),
            name="time_adaptive_text_alpha",
        )
        self.source_norm = k.layers.LayerNormalization(name="source_text_injection_norm")
        self.destination_norm = k.layers.LayerNormalization(name="destination_text_injection_norm")
        self.time_norm = k.layers.LayerNormalization(name="time_text_injection_norm")

        self.source_mode_align = k.layers.Dense(align_dim, name="source_mode_align")
        self.source_text_align = k.layers.Dense(align_dim, name="source_text_align")
        self.destination_mode_align = k.layers.Dense(align_dim, name="destination_mode_align")
        self.destination_text_align = k.layers.Dense(align_dim, name="destination_text_align")
        self.time_mode_align = k.layers.Dense(align_dim, name="time_mode_align")
        self.time_text_align = k.layers.Dense(align_dim, name="time_text_align")

    def _make_projector(self, name):
        return k.Sequential(
            [
                k.layers.Dense(self.hidden_dim),
                k.layers.LayerNormalization(),
                k.layers.Activation("gelu"),
                k.layers.Dense(self.d_model),
                k.layers.LayerNormalization(),
            ],
            name=name,
        )

    def build(self, input_shape):
        self.source_text_align_weight_var = self.add_weight(
            name="source_text_align_weight",
            shape=(),
            initializer=k.initializers.Constant(self.initial_text_align_weight),
            trainable=False,
        )
        self.destination_text_align_weight_var = self.add_weight(
            name="destination_text_align_weight",
            shape=(),
            initializer=k.initializers.Constant(self.initial_text_align_weight),
            trainable=False,
        )
        self.time_text_align_weight_var = self.add_weight(
            name="time_text_align_weight",
            shape=(),
            initializer=k.initializers.Constant(self.initial_text_align_weight),
            trainable=False,
        )
        super().build(input_shape)

    def _lookup_source_destination(self, src, dst, text_time, source_store, destination_store):
        if self.static_source_destination_text:
            return tf.gather(source_store, src), tf.gather(destination_store, dst)

        source_t = tf.gather(source_store, text_time)
        destination_t = tf.gather(destination_store, text_time)
        batch = tf.range(tf.shape(src)[0], dtype=tf.int32)
        source = tf.gather_nd(source_t, tf.stack([batch, src], axis=1))
        destination = tf.gather_nd(destination_t, tf.stack([batch, dst], axis=1))
        return source, destination

    def _raw_features(self, src, dst, time):
        text_time_raw = time - self.target_start
        text_time = tf.clip_by_value(text_time_raw, 0, self.text_time_len - 1)
        clipped = tf.not_equal(text_time_raw, text_time)
        if self.emit_text_metrics:
            self.add_metric(
                tf.reduce_mean(tf.cast(clipped, tf.float32)),
                name="text_time_clipped_ratio",
                aggregation="mean",
            )

        source_text, destination_text = self._lookup_source_destination(
            src,
            dst,
            text_time,
            self.source_text_embeddings,
            self.destination_text_embeddings,
        )
        time_text = tf.gather(self.time_text_embeddings, text_time)

        source_num = None
        destination_num = None
        time_num = None
        if self.text_mode in ("numeric_only", "text_numeric"):
            source_num, destination_num = self._lookup_source_destination(
                src,
                dst,
                text_time,
                self.source_numeric_features,
                self.destination_numeric_features,
            )
            time_num = tf.gather(self.time_numeric_features, text_time)
        return (
            source_text,
            destination_text,
            time_text,
            source_num,
            destination_num,
            time_num,
            text_time,
        )

    def _masked_infonce(self, left, right, mask=None):
        left = tf.math.l2_normalize(left, axis=-1)
        right = tf.math.l2_normalize(right, axis=-1)
        logits = tf.matmul(left, right, transpose_b=True) / self.alignment_temperature
        if mask is not None:
            logits = tf.where(mask, tf.ones_like(logits) * -1e9, logits)
        labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
        loss_a = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_b = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return 0.5 * (tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b))

    def _aggregate_by_key(self, left, right, keys):
        if self.text_align_sample_size > 0:
            count = tf.shape(keys)[0]
            limit = tf.minimum(count, self.text_align_sample_size)
            sample_positions = tf.cast(
                tf.linspace(0.0, tf.cast(count - 1, tf.float32), limit),
                tf.int32,
            )
            left = tf.gather(left, sample_positions)
            right = tf.gather(right, sample_positions)
            keys = tf.gather(keys, sample_positions)
        unique_keys, segment_ids = tf.unique(keys)
        count = tf.shape(unique_keys)[0]
        left = tf.math.unsorted_segment_mean(left, segment_ids, count)
        right = tf.math.unsorted_segment_mean(right, segment_ids, count)
        return left, right, unique_keys

    def _add_entity_alignment(self, mode_x, text_x, keys, weight, mode_proj, text_proj, prefix):
        if not self.text_align_enabled:
            return
        mode_x, text_x, unique_keys = self._aggregate_by_key(mode_x, text_x, keys)
        unique_count = tf.shape(unique_keys)[0]
        mask = None
        if not self.static_source_destination_text:
            entity_ids = unique_keys // self.text_time_len
            entity_times = unique_keys % self.text_time_len
            same_entity = entity_ids[:, None] == entity_ids[None, :]
            time_dist = tf.abs(entity_times[:, None] - entity_times[None, :])
            diagonal = tf.eye(tf.shape(unique_keys)[0], dtype=tf.bool)
            mask = tf.logical_and(tf.logical_and(same_entity, time_dist <= self.temporal_delta), tf.logical_not(diagonal))
        loss = tf.cond(
            unique_count < 2,
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: self._masked_infonce(mode_proj(mode_x), text_proj(text_x), mask=mask),
        )
        self.add_loss(weight * loss)
        self.add_metric(loss, name="%s_text_align_loss" % prefix, aggregation="mean")
        self.add_metric(weight * loss, name="%s_text_align_weighted_loss" % prefix, aggregation="mean")

    def _add_time_alignment(self, mode_x, text_x, times):
        if not self.text_align_enabled:
            return
        mode_x, text_x, unique_times = self._aggregate_by_key(mode_x, text_x, times)
        unique_count = tf.shape(unique_times)[0]
        dist = tf.abs(unique_times[:, None] - unique_times[None, :])
        diagonal = tf.eye(tf.shape(unique_times)[0], dtype=tf.bool)
        near = tf.logical_and(dist <= self.temporal_delta, tf.logical_not(diagonal))
        loss = tf.cond(
            unique_count < 2,
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: self._masked_infonce(self.time_mode_align(mode_x), self.time_text_align(text_x), mask=near),
        )
        self.add_loss(self.time_text_align_weight_var * loss)
        self.add_metric(loss, name="time_text_align_loss", aggregation="mean")
        self.add_metric(
            self.time_text_align_weight_var * loss,
            name="time_text_align_weighted_loss",
            aggregation="mean",
        )

    def call(self, inputs):
        source_token, destination_token, time_token, graph_context, src_input, dst_input, time_input = inputs
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)

        (
            source_text_raw,
            destination_text_raw,
            time_text_raw,
            source_numeric_raw,
            destination_numeric_raw,
            time_numeric_raw,
            text_time,
        ) = self._raw_features(src, dst, time)

        if self.text_mode == "numeric_only":
            source_proj = self.source_projector(source_numeric_raw)
            destination_proj = self.destination_projector(destination_numeric_raw)
            time_proj = self.time_projector(time_numeric_raw)
            source_numeric_gate = 1.0
            destination_numeric_gate = 1.0
            time_numeric_gate = 1.0
        else:
            source_proj = self.source_projector(source_text_raw)
            destination_proj = self.destination_projector(destination_text_raw)
            time_proj = self.time_projector(time_text_raw)
            if self.text_mode == "text_numeric":
                source_numeric_gate = self.source_numeric_gate(
                    tf.concat(
                        [source_numeric_raw, source_token, source_proj, graph_context],
                        axis=-1,
                    )
                )
                destination_numeric_gate = self.destination_numeric_gate(
                    tf.concat(
                        [
                            destination_numeric_raw,
                            destination_token,
                            destination_proj,
                            graph_context,
                        ],
                        axis=-1,
                    )
                )
                time_numeric_gate = self.time_numeric_gate(
                    tf.concat([time_numeric_raw, time_token, time_proj, graph_context], axis=-1)
                )
            else:
                source_numeric_gate = 1.0
                destination_numeric_gate = 1.0
                time_numeric_gate = 1.0

        source_alpha = 0.2 * self.source_alpha_gate(
            tf.concat([source_token, source_proj, graph_context], axis=-1)
        )
        destination_alpha = 0.2 * self.destination_alpha_gate(
            tf.concat([destination_token, destination_proj, graph_context], axis=-1)
        )
        time_alpha = 0.2 * self.time_alpha_gate(
            tf.concat([time_token, time_proj, graph_context], axis=-1)
        )

        source_gate = self.source_gate(
            tf.concat([source_token, source_proj, graph_context], axis=-1)
        )
        destination_gate = self.destination_gate(
            tf.concat([destination_token, destination_proj, graph_context], axis=-1)
        )
        time_gate = self.time_gate(tf.concat([time_token, time_proj, graph_context], axis=-1))

        source_out = self.source_norm(
            source_token + source_alpha * source_gate * source_numeric_gate * source_proj
        )
        destination_out = self.destination_norm(
            destination_token +
            destination_alpha * destination_gate * destination_numeric_gate * destination_proj
        )
        time_out = self.time_norm(
            time_token + time_alpha * time_gate * time_numeric_gate * time_proj
        )

        if self.emit_text_metrics:
            self.add_metric(tf.reduce_mean(source_alpha), name="source_text_alpha", aggregation="mean")
            self.add_metric(
                tf.reduce_mean(destination_alpha),
                name="destination_text_alpha",
                aggregation="mean",
            )
            self.add_metric(tf.reduce_mean(time_alpha), name="time_text_alpha", aggregation="mean")

        if self.static_source_destination_text:
            source_keys = src
            destination_keys = dst
        else:
            source_keys = src * self.text_time_len + text_time
            destination_keys = dst * self.text_time_len + text_time

        self._add_entity_alignment(
            source_token,
            source_text_raw,
            source_keys,
            self.source_text_align_weight_var,
            self.source_mode_align,
            self.source_text_align,
            "source",
        )
        self._add_entity_alignment(
            destination_token,
            destination_text_raw,
            destination_keys,
            self.destination_text_align_weight_var,
            self.destination_mode_align,
            self.destination_text_align,
            "destination",
        )
        self._add_time_alignment(time_token, time_text_raw, text_time)

        return source_out, destination_out, time_out


class DynamicGCNGraphToken(k.layers.Layer):
    """Build a graph token from topology[t] for each queried time id."""

    def __init__(self, topology, node_dim=64, gcn_dim=128, d_model=64, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        topo = np.asarray(topology, dtype="float32")
        if topo.ndim != 3 or topo.shape[1] != topo.shape[2]:
            raise ValueError("topology must have shape [time,node,node]")
        topo_binary = (topo > 0).astype("float32")
        adjacency_with_self = topo_binary + np.eye(topo.shape[1], dtype="float32")[None, :, :]
        degree = np.sum(adjacency_with_self, axis=-1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
        normalized = adjacency_with_self * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]
        self.node_count = int(topo.shape[1])
        self.time_len = int(topo.shape[0])
        self.normalized_topology = tf.constant(normalized, dtype=tf.float32)
        self.attention_mask = tf.constant(adjacency_with_self > 0, dtype=tf.bool)
        self.node_dim = int(node_dim)
        self.gcn_dim = int(gcn_dim)
        self.d_model = int(d_model)
        self.dropout_rate = float(dropout)

        self.node_embedding = k.layers.Embedding(self.node_count, node_dim, name="gcn_node_embedding")
        self.gcn_dense_1 = k.layers.Dense(gcn_dim, name="gcn_dense_1")
        self.gcn_norm_1 = k.layers.LayerNormalization(name="gcn_norm_1")
        self.gcn_dense_2 = k.layers.Dense(gcn_dim, name="gcn_dense_2")
        self.gcn_norm_2 = k.layers.LayerNormalization(name="gcn_norm_2")
        self.dropout = k.layers.Dropout(dropout)
        self.attention_query = k.layers.Dense(gcn_dim, name="graph_context_query")
        self.attention_key = k.layers.Dense(gcn_dim, name="graph_context_key")
        self.attention_value = k.layers.Dense(gcn_dim, name="graph_context_value")
        self.attention_norm = k.layers.LayerNormalization(name="graph_attention_context_norm")
        self.graph_projector = k.Sequential(
            [
                k.layers.Dense(gcn_dim, activation="gelu"),
                k.layers.LayerNormalization(),
                k.layers.Dense(d_model),
                k.layers.LayerNormalization(),
            ],
            name="graph_token_projector",
        )

    def _attention_context(self, h, attention_mask):
        query = self.attention_query(h)
        key = self.attention_key(h)
        value = self.attention_value(h)
        scale = tf.math.rsqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        scores = tf.matmul(query, key, transpose_b=True) * scale
        scores = tf.where(
            attention_mask,
            scores,
            tf.ones_like(scores) * -1e9,
        )
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, value)
        return self.attention_norm(h + context)

    def call(self, inputs, training=None):
        src_input, dst_input, time_input = inputs
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        time = tf.clip_by_value(time, 0, self.time_len - 1)
        batch_size = tf.shape(src)[0]

        adjacency = tf.gather(self.normalized_topology, time)
        attention_mask = tf.gather(self.attention_mask, time)
        node_ids = tf.range(self.node_count, dtype=tf.int32)
        x = self.node_embedding(node_ids)
        x = tf.broadcast_to(x[None, :, :], [batch_size, self.node_count, self.node_dim])

        x1 = tf.matmul(adjacency, x)
        x1 = self.gcn_dense_1(x1)
        x1 = tf.nn.gelu(x1)
        x1 = self.dropout(x1, training=training)
        x1 = self.gcn_norm_1(x1)

        x2 = tf.matmul(adjacency, x1)
        x2 = self.gcn_dense_2(x2)
        x2 = tf.nn.gelu(x2)
        x2 = self.dropout(x2, training=training)
        h = self.gcn_norm_2(x1 + x2)

        attention_context = self._attention_context(h, attention_mask)
        batch = tf.range(batch_size, dtype=tf.int32)
        h_i = tf.gather_nd(h, tf.stack([batch, src], axis=1))
        h_j = tf.gather_nd(h, tf.stack([batch, dst], axis=1))
        context_i = tf.gather_nd(attention_context, tf.stack([batch, src], axis=1))
        context_j = tf.gather_nd(attention_context, tf.stack([batch, dst], axis=1))
        pair = tf.concat(
            [
                h_i,
                h_j,
                tf.abs(h_i - h_j),
                h_i * h_j,
                context_i,
                context_j,
            ],
            axis=-1,
        )
        graph_token = self.graph_projector(pair)

        # Explicit runtime shape assertion: graph token must match mode tokens.
        tf.debugging.assert_equal(tf.shape(graph_token)[-1], self.d_model)
        return graph_token


class TokenAssembly(k.layers.Layer):
    """Stack mode tokens and add role embeddings. Output: [B, token_count, d_model]."""

    def __init__(self, token_count=4, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.token_count = int(token_count)
        self.d_model = int(d_model)
        self.role_embedding = k.layers.Embedding(token_count, d_model, name="role_embedding")

    def call(self, tokens):
        x = tf.stack(tokens, axis=1)
        roles = self.role_embedding(tf.range(self.token_count, dtype=tf.int32))
        return x + roles[None, :, :]


class GraphAttentionBias(k.layers.Layer):
    """Map graph context to bounded additive Transformer attention bias.

    Output shape is [B, heads, token_count, token_count]. The final projection
    is zero-initialized, so enabling this layer starts from the old vanilla
    attention behavior and learns graph priors during training.
    """

    def __init__(self, token_count=4, num_heads=4, hidden_dim=128, max_bias=2.0, **kwargs):
        super().__init__(**kwargs)
        self.token_count = int(token_count)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.max_bias = float(max_bias)
        self.hidden = k.Sequential(
            [
                k.layers.Dense(self.hidden_dim, activation="gelu"),
                k.layers.LayerNormalization(),
            ],
            name="graph_attention_bias_hidden",
        )
        self.bias_projector = k.layers.Dense(
            self.num_heads * self.token_count * self.token_count,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="graph_attention_bias_projector",
        )

    def call(self, graph_context):
        hidden = self.hidden(graph_context)
        bias = tf.tanh(self.bias_projector(hidden)) * self.max_bias
        return tf.reshape(
            bias,
            [
                tf.shape(graph_context)[0],
                self.num_heads,
                self.token_count,
                self.token_count,
            ],
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_count": self.token_count,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "max_bias": self.max_bias,
        })
        return config


class TransformerEncoderBlock(k.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.key_dim = max(1, self.d_model // self.num_heads)
        self.query_dense = k.layers.Dense(self.num_heads * self.key_dim, name="query")
        self.key_dense = k.layers.Dense(self.num_heads * self.key_dim, name="key")
        self.value_dense = k.layers.Dense(self.num_heads * self.key_dim, name="value")
        self.attention_output = k.layers.Dense(self.d_model, name="attention_output")
        self.attention_dropout = k.layers.Dropout(dropout)
        self.dropout_1 = k.layers.Dropout(dropout)
        self.norm_1 = k.layers.LayerNormalization()
        self.ffn = k.Sequential(
            [
                k.layers.Dense(ff_dim, activation="gelu"),
                k.layers.Dropout(dropout),
                k.layers.Dense(d_model),
            ]
        )
        self.dropout_2 = k.layers.Dropout(dropout)
        self.norm_2 = k.layers.LayerNormalization()

    def _split_heads(self, x):
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.num_heads, self.key_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.num_heads * self.key_dim])

    def call(self, x, attention_bias=None, training=None):
        query = self._split_heads(self.query_dense(x))
        key = self._split_heads(self.key_dense(x))
        value = self._split_heads(self.value_dense(x))
        scale = tf.math.rsqrt(tf.cast(self.key_dim, tf.float32))
        scores = tf.matmul(query, key, transpose_b=True) * scale
        if attention_bias is not None:
            scores = scores + attention_bias
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attention_dropout(weights, training=training)
        attn = tf.matmul(weights, value)
        attn = self.attention_output(self._merge_heads(attn))
        x = self.norm_1(x + self.dropout_1(attn, training=training))
        ffn = self.ffn(x, training=training)
        return self.norm_2(x + self.dropout_2(ffn, training=training))


class TransformerEncoderStack(k.layers.Layer):
    def __init__(self, num_layers=2, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name="transformer_block_%d" % idx,
            )
            for idx in range(int(num_layers))
        ]

    def call(self, x, attention_bias=None, training=None):
        for block in self.blocks:
            x = block(x, attention_bias=attention_bias, training=training)
        return x


class TokenPooling(k.layers.Layer):
    """Concatenate all token outputs and mean pooling.

    Keras Dense layers require a known last dimension. The Transformer output
    has static shape [B, token_count, d_model], so the pooled representation is
    explicitly reshaped to [B, token_count * d_model + d_model].
    """

    def __init__(self, token_count, d_model, **kwargs):
        super().__init__(**kwargs)
        self.token_count = int(token_count)
        self.d_model = int(d_model)

    def call(self, x):
        flat = tf.reshape(x, [tf.shape(x)[0], self.token_count * self.d_model])
        mean = tf.reduce_mean(x, axis=1)
        return tf.concat([flat, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.token_count * self.d_model + self.d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_count": self.token_count,
            "d_model": self.d_model,
        })
        return config


def create_gt_mst_model(
    shape,
    topology,
    d_model=64,
    node_dim=64,
    gcn_dim=128,
    transformer_layers=2,
    num_heads=4,
    ff_dim=128,
    dropout=0.1,
    use_graph_token=True,
    use_transformer=True,
    use_mode_text=False,
    text_mode="text_numeric",
    source_text_embeddings=None,
    destination_text_embeddings=None,
    time_text_embeddings=None,
    source_numeric_features=None,
    destination_numeric_features=None,
    time_numeric_features=None,
    text_hidden_dim=128,
    text_alpha=0.02,
    text_align_dim=64,
    text_align_target_ratio=0.0,
    alignment_temperature=0.2,
    temporal_delta=2,
    text_align_sample_size=0,
    emit_text_metrics=True,
    text_target_start=0,
    use_graph_attention_bias=True,
    max_graph_attention_bias=2.0,
    output_bias_init=0.0,
):
    """Create GT-MST.

    Inputs:
      source_id, destination_id, time_id: each [B, 1] int32.

    Core token shapes:
      source/destination/time token: [B, d_model]
      graph token: [B, d_model]
      transformer tokens: [B, 3 or 4, d_model]
    """

    shape = [int(value) for value in shape]
    src_input = k.Input(shape=(1,), dtype="int32", name="source_id")
    dst_input = k.Input(shape=(1,), dtype="int32", name="destination_id")
    time_input = k.Input(shape=(1,), dtype="int32", name="time_id")

    src_flat = k.layers.Flatten()(src_input)
    dst_flat = k.layers.Flatten()(dst_input)
    time_flat = k.layers.Flatten()(time_input)

    source_token = k.layers.Embedding(shape[0], d_model, name="source_embedding")(src_flat)
    destination_token = k.layers.Embedding(shape[1], d_model, name="destination_embedding")(dst_flat)
    time_token = k.layers.Embedding(shape[2], d_model, name="time_embedding")(time_flat)
    source_token = k.layers.Reshape((d_model,))(source_token)
    destination_token = k.layers.Reshape((d_model,))(destination_token)
    time_token = k.layers.Reshape((d_model,))(time_token)

    graph_token = None
    if use_graph_token:
        graph_token = DynamicGCNGraphToken(
            topology=topology,
            node_dim=node_dim,
            gcn_dim=gcn_dim,
            d_model=d_model,
            dropout=dropout,
            name="dynamic_gcn_graph_token",
        )([src_input, dst_input, time_input])
        graph_context = graph_token
    else:
        graph_context = k.layers.Lambda(
            lambda x: tf.zeros_like(x),
            name="zero_graph_context",
        )(source_token)

    if use_mode_text:
        source_token, destination_token, time_token = TextInjectionLayer(
            source_text_embeddings=source_text_embeddings,
            destination_text_embeddings=destination_text_embeddings,
            time_text_embeddings=time_text_embeddings,
            source_numeric_features=source_numeric_features,
            destination_numeric_features=destination_numeric_features,
            time_numeric_features=time_numeric_features,
            d_model=d_model,
            hidden_dim=text_hidden_dim,
            alpha_init=text_alpha,
            text_mode=text_mode,
            align_dim=text_align_dim,
            text_align_target_ratio=text_align_target_ratio,
            alignment_temperature=alignment_temperature,
            temporal_delta=temporal_delta,
            text_align_sample_size=text_align_sample_size,
            emit_text_metrics=emit_text_metrics,
            target_start=text_target_start,
            name="text_injection",
        )([
            source_token,
            destination_token,
            time_token,
            graph_context,
            src_input,
            dst_input,
            time_input,
        ])

    tokens = [source_token, destination_token, time_token]
    if graph_token is not None:
        tokens.append(graph_token)

    if use_transformer:
        token_tensor = TokenAssembly(
            token_count=len(tokens),
            d_model=d_model,
            name="token_assembly",
        )(tokens)
        attention_bias = None
        if use_graph_token and use_graph_attention_bias and graph_token is not None:
            attention_bias = GraphAttentionBias(
                token_count=len(tokens),
                num_heads=num_heads,
                hidden_dim=ff_dim,
                max_bias=max_graph_attention_bias,
                name="graph_attention_bias",
            )(graph_token)
        token_tensor = TransformerEncoderStack(
            num_layers=transformer_layers,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name="transformer_encoder",
        )(token_tensor, attention_bias=attention_bias)
        x = TokenPooling(
            token_count=len(tokens),
            d_model=d_model,
            name="token_pooling",
        )(token_tensor)
    else:
        x = k.layers.Concatenate(name="token_concat_without_transformer")(tokens)

    x = k.layers.Dense(ff_dim, activation="gelu", name="prediction_dense_1")(x)
    x = k.layers.Dropout(dropout, name="prediction_dropout")(x)
    x = k.layers.Dense(max(1, ff_dim // 2), activation="gelu", name="prediction_dense_2")(x)
    output = k.layers.Dense(
        1,
        bias_initializer=k.initializers.Constant(output_bias_init),
        name="traffic_prediction",
    )(x)

    return k.Model(inputs=[src_input, dst_input, time_input], outputs=output, name="GT_MST")
