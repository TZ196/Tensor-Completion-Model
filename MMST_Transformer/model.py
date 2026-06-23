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

        self.source_projector = self._make_projector("source_text_projector")
        self.destination_projector = self._make_projector("destination_text_projector")
        self.time_projector = self._make_projector("time_text_projector")
        self.source_gate = k.layers.Dense(d_model, activation="sigmoid", name="source_text_gate")
        self.destination_gate = k.layers.Dense(d_model, activation="sigmoid", name="destination_text_gate")
        self.time_gate = k.layers.Dense(d_model, activation="sigmoid", name="time_text_gate")
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
        init = np.clip(self.alpha_init / 0.2, 1e-4, 1.0 - 1e-4)
        alpha_logit = float(np.log(init / (1.0 - init)))
        initializer = k.initializers.Constant(alpha_logit)
        self.source_alpha_logit = self.add_weight(
            name="source_text_alpha_logit",
            shape=(),
            initializer=initializer,
            trainable=True,
        )
        self.destination_alpha_logit = self.add_weight(
            name="destination_text_alpha_logit",
            shape=(),
            initializer=initializer,
            trainable=True,
        )
        self.time_alpha_logit = self.add_weight(
            name="time_text_alpha_logit",
            shape=(),
            initializer=initializer,
            trainable=True,
        )
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

        source_parts = []
        destination_parts = []
        time_parts = []
        if self.text_mode in ("text_only", "text_numeric"):
            source_parts.append(source_text)
            destination_parts.append(destination_text)
            time_parts.append(time_text)
        if self.text_mode in ("numeric_only", "text_numeric"):
            source_num, destination_num = self._lookup_source_destination(
                src,
                dst,
                text_time,
                self.source_numeric_features,
                self.destination_numeric_features,
            )
            time_num = tf.gather(self.time_numeric_features, text_time)
            source_parts.append(source_num)
            destination_parts.append(destination_num)
            time_parts.append(time_num)

        source_raw = tf.concat(source_parts, axis=-1)
        destination_raw = tf.concat(destination_parts, axis=-1)
        time_raw = tf.concat(time_parts, axis=-1)
        return source_raw, destination_raw, time_raw, source_text, destination_text, time_text, text_time

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
        source_token, destination_token, time_token, src_input, dst_input, time_input = inputs
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)

        (
            source_raw,
            destination_raw,
            time_raw,
            source_text_raw,
            destination_text_raw,
            time_text_raw,
            text_time,
        ) = self._raw_features(src, dst, time)

        source_proj = self.source_projector(source_raw)
        destination_proj = self.destination_projector(destination_raw)
        time_proj = self.time_projector(time_raw)

        source_alpha = 0.2 * tf.sigmoid(self.source_alpha_logit)
        destination_alpha = 0.2 * tf.sigmoid(self.destination_alpha_logit)
        time_alpha = 0.2 * tf.sigmoid(self.time_alpha_logit)

        source_gate = self.source_gate(tf.concat([source_token, source_proj], axis=-1))
        destination_gate = self.destination_gate(tf.concat([destination_token, destination_proj], axis=-1))
        time_gate = self.time_gate(tf.concat([time_token, time_proj], axis=-1))

        source_out = self.source_norm(source_token + source_alpha * source_gate * source_proj)
        destination_out = self.destination_norm(destination_token + destination_alpha * destination_gate * destination_proj)
        time_out = self.time_norm(time_token + time_alpha * time_gate * time_proj)

        self.add_metric(source_alpha, name="source_text_alpha", aggregation="mean")
        self.add_metric(destination_alpha, name="destination_text_alpha", aggregation="mean")
        self.add_metric(time_alpha, name="time_text_alpha", aggregation="mean")

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
    """Build a graph token in exactly d_model dimensions from dynamic A_t."""

    def __init__(self, topology, node_dim=64, gcn_dim=128, d_model=64, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        topo = np.asarray(topology, dtype="float32")
        if topo.ndim != 3 or topo.shape[1] != topo.shape[2]:
            raise ValueError("topology must have shape [time,node,node]")
        self.topology = tf.constant(topo, dtype=tf.float32)
        self.time_len = int(topo.shape[0])
        self.node_count = int(topo.shape[1])
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
        self.graph_projector = k.Sequential(
            [
                k.layers.Dense(gcn_dim, activation="gelu"),
                k.layers.LayerNormalization(),
                k.layers.Dense(d_model),
                k.layers.LayerNormalization(),
            ],
            name="graph_token_projector",
        )

    def _normalize_adjacency(self, adjacency):
        batch = tf.shape(adjacency)[0]
        eye = tf.eye(self.node_count, batch_shape=[batch], dtype=tf.float32)
        adjacency = adjacency + eye
        degree = tf.reduce_sum(adjacency, axis=-1)
        inv_sqrt = tf.math.rsqrt(tf.maximum(degree, 1e-6))
        return adjacency * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]

    def call(self, inputs, training=None):
        src_input, dst_input, time_input = inputs
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        time = tf.clip_by_value(time, 0, self.time_len - 1)

        adjacency = tf.gather(self.topology, time)
        adjacency = self._normalize_adjacency(adjacency)
        node_ids = tf.range(self.node_count, dtype=tf.int32)
        x = self.node_embedding(node_ids)
        x = tf.broadcast_to(x[None, :, :], [tf.shape(time)[0], self.node_count, self.node_dim])

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

        batch = tf.range(tf.shape(src)[0], dtype=tf.int32)
        h_i = tf.gather_nd(h, tf.stack([batch, src], axis=1))
        h_j = tf.gather_nd(h, tf.stack([batch, dst], axis=1))
        pair = tf.concat([h_i, h_j, tf.abs(h_i - h_j), h_i * h_j], axis=-1)
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


class TransformerEncoderBlock(k.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = k.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, d_model // num_heads),
            dropout=dropout,
        )
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

    def call(self, x, training=None):
        attn = self.attention(x, x, training=training)
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

    def call(self, x, training=None):
        for block in self.blocks:
            x = block(x, training=training)
        return x


class TokenPooling(k.layers.Layer):
    """Concatenate all token outputs and mean pooling."""

    def call(self, x):
        flat = tf.reshape(x, [tf.shape(x)[0], -1])
        mean = tf.reduce_mean(x, axis=1)
        return tf.concat([flat, mean], axis=-1)


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
    text_target_start=0,
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
            target_start=text_target_start,
            name="text_injection",
        )([source_token, destination_token, time_token, src_input, dst_input, time_input])

    tokens = [source_token, destination_token, time_token]
    if use_graph_token:
        graph_token = DynamicGCNGraphToken(
            topology=topology,
            node_dim=node_dim,
            gcn_dim=gcn_dim,
            d_model=d_model,
            dropout=dropout,
            name="dynamic_gcn_graph_token",
        )([src_input, dst_input, time_input])
        tokens.append(graph_token)

    if use_transformer:
        token_tensor = TokenAssembly(
            token_count=len(tokens),
            d_model=d_model,
            name="token_assembly",
        )(tokens)
        token_tensor = TransformerEncoderStack(
            num_layers=transformer_layers,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name="transformer_encoder",
        )(token_tensor)
        x = TokenPooling(name="token_pooling")(token_tensor)
    else:
        x = k.layers.Concatenate(name="token_concat_without_transformer")(tokens)

    x = k.layers.Dense(ff_dim, activation="gelu", name="prediction_dense_1")(x)
    x = k.layers.Dropout(dropout, name="prediction_dropout")(x)
    x = k.layers.Dense(max(1, ff_dim // 2), activation="gelu", name="prediction_dense_2")(x)
    output = k.layers.Dense(
        1,
        activation="relu",
        bias_initializer=k.initializers.Constant(output_bias_init),
        name="traffic_prediction",
    )(x)

    return k.Model(inputs=[src_input, dst_input, time_input], outputs=output, name="GT_MST")

