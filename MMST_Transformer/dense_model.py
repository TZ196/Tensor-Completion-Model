import numpy as np
import tensorflow as tf
from tensorflow import keras as k


def _as_float_array(value, name):
    if value is None:
        return None
    array = np.asarray(value, dtype="float32")
    if array.ndim < 2:
        raise ValueError("%s must have at least 2 dimensions" % name)
    return array


class DenseStaticGraphContext(k.layers.Layer):
    """Static satellite graph encoder used by dense time-block GT-MST.

    The layer computes static node states from topology[0] once per forward
    pass, then expands them into pair-level graph tokens for all source-
    destination pairs. The output shape is [N, N, d_model].
    """

    def __init__(self, topology, node_dim=64, gcn_dim=128, d_model=64, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        topo = np.asarray(topology, dtype="float32")
        if topo.ndim != 3 or topo.shape[1] != topo.shape[2]:
            raise ValueError("topology must have shape [time,node,node]")
        topo_static = (topo[0] > 0).astype("float32")
        self.node_count = int(topo.shape[1])
        adjacency_with_self = topo_static + np.eye(self.node_count, dtype="float32")
        degree = np.sum(adjacency_with_self, axis=-1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
        normalized = adjacency_with_self * inv_sqrt[:, None] * inv_sqrt[None, :]
        self.normalized_topology = tf.constant(normalized, dtype=tf.float32)
        self.attention_mask = tf.constant(adjacency_with_self > 0, dtype=tf.bool)
        self.node_dim = int(node_dim)
        self.gcn_dim = int(gcn_dim)
        self.d_model = int(d_model)

        self.node_embedding = k.layers.Embedding(self.node_count, node_dim, name="dense_gcn_node_embedding")
        self.gcn_dense_1 = k.layers.Dense(gcn_dim, name="dense_gcn_dense_1")
        self.gcn_norm_1 = k.layers.LayerNormalization(name="dense_gcn_norm_1")
        self.gcn_dense_2 = k.layers.Dense(gcn_dim, name="dense_gcn_dense_2")
        self.gcn_norm_2 = k.layers.LayerNormalization(name="dense_gcn_norm_2")
        self.dropout = k.layers.Dropout(dropout)
        self.attention_query = k.layers.Dense(gcn_dim, name="dense_graph_context_query")
        self.attention_key = k.layers.Dense(gcn_dim, name="dense_graph_context_key")
        self.attention_value = k.layers.Dense(gcn_dim, name="dense_graph_context_value")
        self.attention_norm = k.layers.LayerNormalization(name="dense_graph_attention_context_norm")
        self.graph_projector = k.Sequential(
            [
                k.layers.Dense(gcn_dim, activation="gelu"),
                k.layers.LayerNormalization(),
                k.layers.Dense(d_model),
                k.layers.LayerNormalization(),
            ],
            name="dense_graph_token_projector",
        )

    def _node_context(self, h):
        query = self.attention_query(h)
        key = self.attention_key(h)
        value = self.attention_value(h)
        scale = tf.math.rsqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        scores = tf.matmul(query, key, transpose_b=True) * scale
        scores = tf.where(self.attention_mask, scores, tf.ones_like(scores) * -1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, value)
        return self.attention_norm(h + context)

    def call(self, inputs=None, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            src_ids = tf.cast(tf.reshape(inputs[0], [-1]), tf.int32)
            dst_ids = tf.cast(tf.reshape(inputs[1], [-1]), tf.int32)
        else:
            src_ids = tf.range(self.node_count, dtype=tf.int32)
            dst_ids = tf.range(self.node_count, dtype=tf.int32)

        node_ids = tf.range(self.node_count, dtype=tf.int32)
        x = self.node_embedding(node_ids)
        adjacency = self.normalized_topology

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

        context = self._node_context(h)
        h_src = tf.gather(h, src_ids)
        h_dst = tf.gather(h, dst_ids)
        context_src = tf.gather(context, src_ids)
        context_dst = tf.gather(context, dst_ids)
        src_count = tf.shape(src_ids)[0]
        dst_count = tf.shape(dst_ids)[0]
        h_i = tf.tile(h_src[:, None, :], [1, dst_count, 1])
        h_j = tf.tile(h_dst[None, :, :], [src_count, 1, 1])
        context_i = tf.tile(context_src[:, None, :], [1, dst_count, 1])
        context_j = tf.tile(context_dst[None, :, :], [src_count, 1, 1])
        pair = tf.concat([h_i, h_j, tf.abs(h_i - h_j), h_i * h_j, context_i, context_j], axis=-1)
        return self.graph_projector(pair)


class DenseGraphAttentionBias(k.layers.Layer):
    def __init__(self, token_count, num_heads=4, hidden_dim=128, max_bias=2.0, **kwargs):
        super().__init__(**kwargs)
        self.token_count = int(token_count)
        self.num_heads = int(num_heads)
        self.max_bias = float(max_bias)
        self.hidden = k.Sequential(
            [
                k.layers.Dense(hidden_dim, activation="gelu"),
                k.layers.LayerNormalization(),
            ],
            name="dense_graph_attention_bias_hidden",
        )
        self.projector = k.layers.Dense(
            self.num_heads * self.token_count * self.token_count,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="dense_graph_attention_bias_projector",
        )

    def call(self, graph_token_flat):
        hidden = self.hidden(graph_token_flat)
        bias = tf.tanh(self.projector(hidden)) * self.max_bias
        return tf.reshape(
            bias,
            [tf.shape(graph_token_flat)[0], self.num_heads, self.token_count, self.token_count],
        )


class DenseTransformerBlock(k.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.key_dim = max(1, self.d_model // self.num_heads)
        self.query_dense = k.layers.Dense(self.num_heads * self.key_dim)
        self.key_dense = k.layers.Dense(self.num_heads * self.key_dim)
        self.value_dense = k.layers.Dense(self.num_heads * self.key_dim)
        self.attention_output = k.layers.Dense(self.d_model)
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
        y = self.ffn(x, training=training)
        return self.norm_2(x + self.dropout_2(y, training=training))


class DenseIndependentTextGTMST(k.Model):
    """Dense time-block GT-MST with independent text/numeric tokens.

    Input:
      time_ids: [B, Tc]

    Output:
      prediction: [B, N, N, Tc]

    Variants:
      M7_dense: source/destination/time/graph + text tokens
      M8_dense: M7_dense + numeric control token
      M9_dense: M8_dense + mode-level TextAlign loss
    """

    def __init__(
        self,
        shape,
        topology,
        source_text_embeddings,
        destination_text_embeddings,
        time_text_embeddings,
        source_numeric_features=None,
        destination_numeric_features=None,
        time_numeric_features=None,
        variant="M8_dense",
        d_model=64,
        node_dim=64,
        gcn_dim=128,
        transformer_layers=2,
        num_heads=4,
        ff_dim=128,
        dropout=0.1,
        text_hidden_dim=128,
        text_align_weight=0.0,
        alignment_temperature=0.2,
        max_graph_attention_bias=2.0,
        output_bias_init=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if variant not in ("M7_dense", "M8_dense", "M9_dense"):
            raise ValueError("Dense variant must be M7_dense, M8_dense, or M9_dense")
        self.shape_ = [int(value) for value in shape]
        self.node_count = self.shape_[0]
        self.time_count = self.shape_[2]
        self.variant = variant
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.text_align_weight = float(text_align_weight if variant == "M9_dense" else 0.0)
        self.alignment_temperature = float(alignment_temperature)

        source_text = _as_float_array(source_text_embeddings, "source_text_embeddings")
        destination_text = _as_float_array(destination_text_embeddings, "destination_text_embeddings")
        time_text = _as_float_array(time_text_embeddings, "time_text_embeddings")
        if source_text.shape != destination_text.shape:
            raise ValueError("source/destination text shapes must match")
        if source_text.ndim not in (2, 3):
            raise ValueError("source text must have shape [node,dim] or [time,node,dim]")
        if time_text.ndim != 2:
            raise ValueError("time text must have shape [time,dim]")
        self.static_source_destination_text = source_text.ndim == 2
        self.source_text_embeddings = tf.constant(source_text, dtype=tf.float32)
        self.destination_text_embeddings = tf.constant(destination_text, dtype=tf.float32)
        self.time_text_embeddings = tf.constant(time_text, dtype=tf.float32)

        self.use_numeric = variant in ("M8_dense", "M9_dense")
        if self.use_numeric:
            for name, value in (
                ("source_numeric_features", source_numeric_features),
                ("destination_numeric_features", destination_numeric_features),
                ("time_numeric_features", time_numeric_features),
            ):
                if value is None:
                    raise ValueError("%s is required for %s" % (name, variant))
            source_numeric = _as_float_array(source_numeric_features, "source_numeric_features")
            destination_numeric = _as_float_array(destination_numeric_features, "destination_numeric_features")
            time_numeric = _as_float_array(time_numeric_features, "time_numeric_features")
            if source_numeric.shape[:-1] != source_text.shape[:-1]:
                raise ValueError("source numeric prefix shape must match source text")
            if destination_numeric.shape[:-1] != destination_text.shape[:-1]:
                raise ValueError("destination numeric prefix shape must match destination text")
            if time_numeric.shape[0] != time_text.shape[0]:
                raise ValueError("time numeric length must match time text length")
            self.source_numeric_features = tf.constant(source_numeric, dtype=tf.float32)
            self.destination_numeric_features = tf.constant(destination_numeric, dtype=tf.float32)
            self.time_numeric_features = tf.constant(time_numeric, dtype=tf.float32)
        else:
            self.source_numeric_features = None
            self.destination_numeric_features = None
            self.time_numeric_features = None

        self.source_embedding = k.layers.Embedding(self.node_count, d_model, name="dense_source_embedding")
        self.destination_embedding = k.layers.Embedding(self.node_count, d_model, name="dense_destination_embedding")
        self.time_embedding = k.layers.Embedding(self.time_count, d_model, name="dense_time_embedding")
        self.graph_encoder = DenseStaticGraphContext(
            topology=topology,
            node_dim=node_dim,
            gcn_dim=gcn_dim,
            d_model=d_model,
            dropout=dropout,
            name="dense_static_graph_context",
        )
        self.source_text_projector = self._projector(text_hidden_dim, "dense_source_text_projector")
        self.destination_text_projector = self._projector(text_hidden_dim, "dense_destination_text_projector")
        self.time_text_projector = self._projector(text_hidden_dim, "dense_time_text_projector")
        self.numeric_projector = (
            self._projector(text_hidden_dim, "dense_numeric_control_projector")
            if self.use_numeric
            else None
        )

        self.token_count = 8 if self.use_numeric else 7
        self.role_embedding = k.layers.Embedding(self.token_count, d_model, name="dense_role_embedding")
        self.graph_bias = DenseGraphAttentionBias(
            token_count=self.token_count,
            num_heads=num_heads,
            hidden_dim=ff_dim,
            max_bias=max_graph_attention_bias,
            name="dense_graph_attention_bias",
        )
        self.blocks = [
            DenseTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name="dense_transformer_block_%d" % idx,
            )
            for idx in range(int(transformer_layers))
        ]
        self.prediction_dense_1 = k.layers.Dense(ff_dim, activation="gelu", name="dense_prediction_dense_1")
        self.prediction_dropout = k.layers.Dropout(dropout)
        self.prediction_dense_2 = k.layers.Dense(max(1, ff_dim // 2), activation="gelu", name="dense_prediction_dense_2")
        self.prediction = k.layers.Dense(
            1,
            bias_initializer=k.initializers.Constant(output_bias_init),
            name="dense_traffic_prediction",
        )

        self.source_mode_align = k.layers.Dense(d_model, name="dense_source_mode_align")
        self.source_text_align = k.layers.Dense(d_model, name="dense_source_text_align")
        self.destination_mode_align = k.layers.Dense(d_model, name="dense_destination_mode_align")
        self.destination_text_align = k.layers.Dense(d_model, name="dense_destination_text_align")
        self.time_mode_align = k.layers.Dense(d_model, name="dense_time_mode_align")
        self.time_text_align = k.layers.Dense(d_model, name="dense_time_text_align")

    def _projector(self, hidden_dim, name):
        return k.Sequential(
            [
                k.layers.Dense(hidden_dim),
                k.layers.LayerNormalization(),
                k.layers.Activation("gelu"),
                k.layers.Dense(self.d_model),
                k.layers.LayerNormalization(),
            ],
            name=name,
        )

    def _lookup_node_time(self, store, time_ids):
        if store.shape.rank == 2:
            return store
        gathered = tf.gather(store, time_ids)
        return gathered

    def _node_text_tokens(self, time_ids, source_ids, destination_ids):
        if self.static_source_destination_text:
            source_text = tf.gather(self.source_text_projector(self.source_text_embeddings), source_ids)
            destination_text = tf.gather(
                self.destination_text_projector(self.destination_text_embeddings),
                destination_ids,
            )
            return source_text, destination_text

        # For dynamic node text, average over the requested time block to keep
        # source/destination text mode-level rather than OD-time-level.
        source_raw = tf.reduce_mean(tf.gather(self.source_text_embeddings, time_ids), axis=0)
        destination_raw = tf.reduce_mean(tf.gather(self.destination_text_embeddings, time_ids), axis=0)
        return (
            tf.gather(self.source_text_projector(source_raw), source_ids),
            tf.gather(self.destination_text_projector(destination_raw), destination_ids),
        )

    def _numeric_control_token(self, time_ids, graph_pair, source_ids, destination_ids):
        if not self.use_numeric:
            return None
        if self.numeric_projector is None:
            raise RuntimeError("numeric_projector is required when use_numeric=True")
        if self.source_numeric_features.shape.rank == 2:
            source_numeric = tf.gather(self.source_numeric_features, source_ids)
            destination_numeric = tf.gather(self.destination_numeric_features, destination_ids)
        else:
            source_numeric = tf.gather(
                tf.reduce_mean(tf.gather(self.source_numeric_features, time_ids), axis=0),
                source_ids,
            )
            destination_numeric = tf.gather(
                tf.reduce_mean(tf.gather(self.destination_numeric_features, time_ids), axis=0),
                destination_ids,
            )
        time_numeric = tf.gather(self.time_numeric_features, time_ids)

        src_count = tf.shape(source_ids)[0]
        dst_count = tf.shape(destination_ids)[0]
        tc = tf.shape(time_ids)[0]
        source_pair = tf.tile(source_numeric[:, None, :], [1, dst_count, 1])
        destination_pair = tf.tile(destination_numeric[None, :, :], [src_count, 1, 1])
        pair_numeric = tf.concat([source_pair, destination_pair, graph_pair], axis=-1)
        pair_numeric = tf.tile(pair_numeric[:, :, None, :], [1, 1, tc, 1])
        time_numeric = tf.tile(time_numeric[None, None, :, :], [src_count, dst_count, 1, 1])
        return self.numeric_projector(tf.concat([pair_numeric, time_numeric], axis=-1))

    def _infonce(self, left, right):
        left = tf.math.l2_normalize(left, axis=-1)
        right = tf.math.l2_normalize(right, axis=-1)
        logits = tf.matmul(left, right, transpose_b=True) / self.alignment_temperature
        labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
        loss_a = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_b = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return 0.5 * (tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b))

    def _add_text_align_loss(self, source_mode, destination_mode, time_mode, source_text, destination_text, time_text):
        if self.text_align_weight <= 0.0:
            return
        source_loss = self._infonce(self.source_mode_align(source_mode), self.source_text_align(source_text))
        destination_loss = self._infonce(
            self.destination_mode_align(destination_mode),
            self.destination_text_align(destination_text),
        )
        time_loss = self._infonce(self.time_mode_align(time_mode), self.time_text_align(time_text))
        raw = source_loss + destination_loss + time_loss
        weighted = self.text_align_weight * raw
        self.add_loss(weighted)
        self.add_metric(source_loss, name="dense_source_text_align_loss", aggregation="mean")
        self.add_metric(destination_loss, name="dense_destination_text_align_loss", aggregation="mean")
        self.add_metric(time_loss, name="dense_time_text_align_loss", aggregation="mean")
        self.add_metric(weighted, name="dense_text_align_weighted_loss", aggregation="mean")

    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)):
            source_ids = tf.cast(tf.reshape(inputs[0], [-1]), tf.int32)
            destination_ids = tf.cast(tf.reshape(inputs[1], [-1]), tf.int32)
            time_ids = tf.cast(inputs[2], tf.int32)
        else:
            source_ids = tf.range(self.node_count, dtype=tf.int32)
            destination_ids = tf.range(self.node_count, dtype=tf.int32)
            time_ids = tf.cast(inputs, tf.int32)
        if time_ids.shape.rank == 2:
            time_ids = time_ids[0]
        time_ids = tf.reshape(time_ids, [-1])
        tc = tf.shape(time_ids)[0]
        src_count = tf.shape(source_ids)[0]
        dst_count = tf.shape(destination_ids)[0]

        node_ids = tf.range(self.node_count, dtype=tf.int32)
        source_mode_all = self.source_embedding(node_ids)
        destination_mode_all = self.destination_embedding(node_ids)
        source_mode = tf.gather(source_mode_all, source_ids)
        destination_mode = tf.gather(destination_mode_all, destination_ids)
        time_mode = self.time_embedding(time_ids)
        graph_pair = self.graph_encoder([source_ids, destination_ids], training=training)

        source_text, destination_text = self._node_text_tokens(time_ids, source_ids, destination_ids)
        time_text = self.time_text_projector(tf.gather(self.time_text_embeddings, time_ids))

        source_pair = tf.tile(source_mode[:, None, None, :], [1, dst_count, tc, 1])
        destination_pair = tf.tile(destination_mode[None, :, None, :], [src_count, 1, tc, 1])
        time_pair = tf.tile(time_mode[None, None, :, :], [src_count, dst_count, 1, 1])
        graph_pair_time = tf.tile(graph_pair[:, :, None, :], [1, 1, tc, 1])
        source_text_pair = tf.tile(source_text[:, None, None, :], [1, dst_count, tc, 1])
        destination_text_pair = tf.tile(destination_text[None, :, None, :], [src_count, 1, tc, 1])
        time_text_pair = tf.tile(time_text[None, None, :, :], [src_count, dst_count, 1, 1])

        tokens = [
            source_pair,
            destination_pair,
            time_pair,
            graph_pair_time,
            source_text_pair,
            destination_text_pair,
            time_text_pair,
        ]
        numeric_token = self._numeric_control_token(time_ids, graph_pair, source_ids, destination_ids)
        if numeric_token is not None:
            tokens.append(numeric_token)

        token_tensor = tf.stack(tokens, axis=3)
        token_tensor = tf.reshape(token_tensor, [src_count * dst_count * tc, self.token_count, self.d_model])
        role = self.role_embedding(tf.range(self.token_count, dtype=tf.int32))
        token_tensor = token_tensor + role[None, :, :]

        graph_flat = tf.reshape(graph_pair_time, [src_count * dst_count * tc, self.d_model])
        attention_bias = self.graph_bias(graph_flat)
        for block in self.blocks:
            token_tensor = block(token_tensor, attention_bias=attention_bias, training=training)

        flat = tf.reshape(token_tensor, [tf.shape(token_tensor)[0], self.token_count * self.d_model])
        mean = tf.reduce_mean(token_tensor, axis=1)
        x = tf.concat([flat, mean], axis=-1)
        x = self.prediction_dense_1(x)
        x = self.prediction_dropout(x, training=training)
        x = self.prediction_dense_2(x)
        pred = self.prediction(x)
        pred = tf.reshape(pred, [src_count, dst_count, tc])

        self._add_text_align_loss(source_mode, destination_mode, time_mode, source_text, destination_text, time_text)
        return pred[None, :, :, :]
