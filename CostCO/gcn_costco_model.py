import numpy as np
import tensorflow as tf
from tensorflow import keras as k

from costco_model import mae, nmae, nrmse, rmse, transform_indices


class ModeWiseStructuralLayer(k.layers.Layer):
    """Inject topology-derived structure into CoSTCo source/dest/time modes."""

    def __init__(self, node_features=None, time_features=None, rank=50,
                 hidden_dim=64, align_dim=64, beta=0.1, alpha_init=0.1,
                 source_align_weight=0.0, destination_align_weight=0.0,
                 time_align_weight=0.0, alignment_temperature=0.2,
                 temporal_delta=2, **kwargs):
        super().__init__(**kwargs)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.align_dim = int(align_dim)
        self.beta_init = float(beta)
        self.alpha_init = float(alpha_init)
        self.source_align_weight = float(source_align_weight)
        self.destination_align_weight = float(destination_align_weight)
        self.time_align_weight = float(time_align_weight)
        self.alignment_temperature = float(alignment_temperature)
        self.temporal_delta = int(temporal_delta)

        self.structural_enabled = (
            node_features is not None and time_features is not None
        )
        if (node_features is None) != (time_features is None):
            raise ValueError(
                "node_features and time_features must be provided together"
            )
        if self.structural_enabled:
            node = np.asarray(node_features, dtype="float32")
            time = np.asarray(time_features, dtype="float32")
            if node.ndim != 3:
                raise ValueError("node_features must have shape [time, node, dim]")
            if time.ndim != 2:
                raise ValueError("time_features must have shape [time, dim]")
            if node.shape[0] != time.shape[0]:
                raise ValueError("node/time structural features must share time length")
            self.node_features = tf.constant(node, dtype=tf.float32)
            self.time_features = tf.constant(time, dtype=tf.float32)
            self.time_len = int(node.shape[0])
            self.node_count = int(node.shape[1])
            self.node_feature_dim = int(node.shape[2])
            self.time_feature_dim = int(time.shape[1])
        else:
            self.node_features = None
            self.time_features = None
            self.time_len = 0
            self.node_count = 0
            self.node_feature_dim = 0
            self.time_feature_dim = 0

        self.source_time_dense_1 = k.layers.Dense(hidden_dim, activation="gelu")
        self.source_time_dense_2 = k.layers.Dense(rank)
        self.dest_time_dense_1 = k.layers.Dense(hidden_dim, activation="gelu")
        self.dest_time_dense_2 = k.layers.Dense(rank)
        self.source_norm = k.layers.LayerNormalization()
        self.dest_norm = k.layers.LayerNormalization()
        self.time_norm = k.layers.LayerNormalization()

        self.node_proj_1 = k.layers.Dense(hidden_dim)
        self.node_proj_norm_1 = k.layers.LayerNormalization()
        self.node_proj_act = k.layers.Activation("gelu")
        self.node_proj_2 = k.layers.Dense(rank)
        self.node_proj_norm_2 = k.layers.LayerNormalization()
        self.source_adapter = k.layers.Dense(rank)
        self.dest_adapter = k.layers.Dense(rank)
        self.time_proj_1 = k.layers.Dense(hidden_dim)
        self.time_proj_norm_1 = k.layers.LayerNormalization()
        self.time_proj_act = k.layers.Activation("gelu")
        self.time_proj_2 = k.layers.Dense(rank)
        self.time_proj_norm_2 = k.layers.LayerNormalization()

        self.source_gate = k.layers.Dense(rank, activation="sigmoid")
        self.dest_gate = k.layers.Dense(rank, activation="sigmoid")
        self.time_gate = k.layers.Dense(rank, activation="sigmoid")
        self.source_fuse_norm = k.layers.LayerNormalization()
        self.dest_fuse_norm = k.layers.LayerNormalization()
        self.time_fuse_norm = k.layers.LayerNormalization()

        self.source_mode_proj = k.layers.Dense(align_dim)
        self.source_struct_proj = k.layers.Dense(align_dim)
        self.dest_mode_proj = k.layers.Dense(align_dim)
        self.dest_struct_proj = k.layers.Dense(align_dim)
        self.time_mode_proj = k.layers.Dense(align_dim)
        self.time_struct_proj = k.layers.Dense(align_dim)

    def build(self, input_shape):
        def init_alpha(value):
            value = np.clip(value / 0.2, 1e-4, 1.0 - 1e-4)
            return float(np.log(value / (1.0 - value)))

        beta_initializer = k.initializers.Constant(init_alpha(self.beta_init))
        self.beta_logit = self.add_weight(
            name="structural_beta_logit",
            shape=(),
            initializer=beta_initializer,
            trainable=True,
        )
        if self.structural_enabled:
            initializer = k.initializers.Constant(init_alpha(self.alpha_init))
            self.source_alpha_logit = self.add_weight(
                name="source_alpha_logit",
                shape=(),
                initializer=initializer,
                trainable=True,
            )
            self.dest_alpha_logit = self.add_weight(
                name="destination_alpha_logit",
                shape=(),
                initializer=initializer,
                trainable=True,
            )
            self.time_alpha_logit = self.add_weight(
                name="time_alpha_logit",
                shape=(),
                initializer=initializer,
                trainable=True,
            )
        super().build(input_shape)

    def _project_node(self, features):
        x = self.node_proj_1(features)
        x = self.node_proj_norm_1(x)
        x = self.node_proj_act(x)
        x = self.node_proj_2(x)
        return self.node_proj_norm_2(x)

    def _project_time(self, features):
        x = self.time_proj_1(features)
        x = self.time_proj_norm_1(x)
        x = self.time_proj_act(x)
        x = self.time_proj_2(x)
        return self.time_proj_norm_2(x)

    def _masked_infonce(self, left, right, mask=None):
        left = tf.math.l2_normalize(left, axis=-1)
        right = tf.math.l2_normalize(right, axis=-1)
        logits = tf.matmul(left, right, transpose_b=True)
        logits = logits / self.alignment_temperature
        if mask is not None:
            logits = tf.where(mask, tf.ones_like(logits) * -1e9, logits)
        labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
        loss_a = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        loss_b = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )
        return 0.5 * (tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b))

    def _aggregate_by_key(self, left, right, keys):
        unique_keys, segment_ids = tf.unique(keys)
        count = tf.shape(unique_keys)[0]
        left = tf.math.unsorted_segment_mean(left, segment_ids, count)
        right = tf.math.unsorted_segment_mean(right, segment_ids, count)
        return left, right, unique_keys

    def _add_entity_alignment(self, mode_x, struct_x, keys, weight,
                              mode_proj, struct_proj, metric_prefix):
        if weight <= 0.0:
            return
        mode_x, struct_x, unique_keys = self._aggregate_by_key(
            mode_x, struct_x, keys
        )
        unique_count = tf.shape(unique_keys)[0]
        entity_ids = unique_keys // self.time_len
        entity_times = unique_keys % self.time_len
        same_entity = entity_ids[:, None] == entity_ids[None, :]
        time_dist = tf.abs(entity_times[:, None] - entity_times[None, :])
        near_time = time_dist <= self.temporal_delta
        diagonal = tf.eye(tf.shape(unique_keys)[0], dtype=tf.bool)
        mask = tf.logical_and(
            tf.logical_and(same_entity, near_time),
            tf.logical_not(diagonal),
        )
        loss = tf.cond(
            unique_count < 2,
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: self._masked_infonce(
                mode_proj(mode_x),
                struct_proj(struct_x),
                mask=mask,
            ),
        )
        self.add_loss(weight * loss)
        self.add_metric(
            loss,
            name="%s_struct_align_loss" % metric_prefix,
            aggregation="mean",
        )
        self.add_metric(
            weight * loss,
            name="%s_struct_align_weighted_loss" % metric_prefix,
            aggregation="mean",
        )

    def _add_time_alignment(self, mode_x, struct_x, times):
        if self.time_align_weight <= 0.0:
            return
        mode_x, struct_x, unique_times = self._aggregate_by_key(
            mode_x, struct_x, times
        )
        unique_count = tf.shape(unique_times)[0]
        dist = tf.abs(unique_times[:, None] - unique_times[None, :])
        diagonal = tf.eye(tf.shape(unique_times)[0], dtype=tf.bool)
        near = tf.logical_and(dist <= self.temporal_delta, tf.logical_not(diagonal))
        loss = tf.cond(
            unique_count < 2,
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: self._masked_infonce(
                self.time_mode_proj(mode_x),
                self.time_struct_proj(struct_x),
                mask=near,
            ),
        )
        self.add_loss(self.time_align_weight * loss)
        self.add_metric(
            loss,
            name="time_struct_align_loss",
            aggregation="mean",
        )
        self.add_metric(
            self.time_align_weight * loss,
            name="time_struct_align_weighted_loss",
            aggregation="mean",
        )

    def call(self, inputs):
        src_embed, dst_embed, time_embed, src_input, dst_input, time_input = inputs
        src_embed = tf.squeeze(src_embed, axis=1)
        dst_embed = tf.squeeze(dst_embed, axis=1)
        time_embed = tf.squeeze(time_embed, axis=1)
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)

        source_delta = self.source_time_dense_2(
            self.source_time_dense_1(tf.concat([src_embed, time_embed], axis=-1))
        )
        dest_delta = self.dest_time_dense_2(
            self.dest_time_dense_1(tf.concat([dst_embed, time_embed], axis=-1))
        )
        beta = 0.2 * tf.sigmoid(self.beta_logit)
        source_mode = self.source_norm(src_embed + beta * source_delta)
        dest_mode = self.dest_norm(dst_embed + beta * dest_delta)
        time_mode = self.time_norm(time_embed)

        if not self.structural_enabled:
            return [
                source_mode[:, None, :],
                dest_mode[:, None, :],
                time_mode[:, None, :],
            ]

        node_t = tf.gather(self.node_features, time)
        src_struct_raw = tf.gather_nd(
            node_t,
            tf.stack([tf.range(tf.shape(time)[0], dtype=tf.int32), src], axis=1),
        )
        dst_struct_raw = tf.gather_nd(
            node_t,
            tf.stack([tf.range(tf.shape(time)[0], dtype=tf.int32), dst], axis=1),
        )
        time_struct_raw = tf.gather(self.time_features, time)

        src_base = self._project_node(src_struct_raw)
        dst_base = self._project_node(dst_struct_raw)
        src_struct = src_base + 0.1 * self.source_adapter(src_base)
        dst_struct = dst_base + 0.1 * self.dest_adapter(dst_base)
        time_struct = self._project_time(time_struct_raw)

        source_alpha = 0.2 * tf.sigmoid(self.source_alpha_logit)
        dest_alpha = 0.2 * tf.sigmoid(self.dest_alpha_logit)
        time_alpha = 0.2 * tf.sigmoid(self.time_alpha_logit)
        source_gate = self.source_gate(tf.concat([source_mode, src_struct], axis=-1))
        dest_gate = self.dest_gate(tf.concat([dest_mode, dst_struct], axis=-1))
        time_gate = self.time_gate(tf.concat([time_mode, time_struct], axis=-1))

        source_out = self.source_fuse_norm(
            source_mode + source_alpha * source_gate * src_struct
        )
        dest_out = self.dest_fuse_norm(
            dest_mode + dest_alpha * dest_gate * dst_struct
        )
        time_out = self.time_fuse_norm(
            time_mode + time_alpha * time_gate * time_struct
        )

        source_keys = src * self.time_len + time
        dest_keys = dst * self.time_len + time
        self._add_entity_alignment(
            source_mode,
            src_struct,
            source_keys,
            self.source_align_weight,
            self.source_mode_proj,
            self.source_struct_proj,
            "source",
        )
        self._add_entity_alignment(
            dest_mode,
            dst_struct,
            dest_keys,
            self.destination_align_weight,
            self.dest_mode_proj,
            self.dest_struct_proj,
            "destination",
        )
        self._add_time_alignment(time_mode, time_struct, time)

        return [
            source_out[:, None, :],
            dest_out[:, None, :],
            time_out[:, None, :],
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "hidden_dim": self.hidden_dim,
            "align_dim": self.align_dim,
            "beta_init": self.beta_init,
            "alpha_init": self.alpha_init,
            "source_align_weight": self.source_align_weight,
            "destination_align_weight": self.destination_align_weight,
            "time_align_weight": self.time_align_weight,
            "alignment_temperature": self.alignment_temperature,
            "temporal_delta": self.temporal_delta,
            "structural_enabled": self.structural_enabled,
        })
        return config


class ModeWiseTextLayer(k.layers.Layer):
    """Inject source/destination/time text embeddings into CoSTCo modes."""

    def __init__(self, source_text_embeddings, destination_text_embeddings,
                 time_text_embeddings, source_text_numeric_features=None,
                 destination_text_numeric_features=None,
                 time_text_numeric_features=None, rank=50, hidden_dim=64, align_dim=64,
                 alpha_init=0.1, text_align_target_ratio=0.0,
                 alignment_temperature=0.2,
                 temporal_delta=2, target_start=0, text_fusion_mode="concat",
                 numeric_alpha_init=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        if text_fusion_mode not in ("concat", "gated_numeric"):
            raise ValueError(
                "text_fusion_mode must be 'concat' or 'gated_numeric'"
            )
        source = np.asarray(source_text_embeddings, dtype="float32")
        destination = np.asarray(destination_text_embeddings, dtype="float32")
        time = np.asarray(time_text_embeddings, dtype="float32")
        if source.ndim not in (2, 3):
            raise ValueError(
                "source_text_embeddings must have shape [node,dim] "
                "or [time,node,dim]"
            )
        if destination.ndim not in (2, 3):
            raise ValueError(
                "destination_text_embeddings must have shape [node,dim] "
                "or [time,node,dim]"
            )
        if time.ndim != 2:
            raise ValueError("time_text_embeddings must have shape [time,dim]")
        if source.shape != destination.shape:
            raise ValueError("source/destination text embeddings must share shape")
        static_source_destination = source.ndim == 2
        if not static_source_destination and source.shape[0] != time.shape[0]:
            raise ValueError("source/destination/time text time lengths disagree")
        source_numeric = None
        destination_numeric = None
        time_numeric = None
        numeric_inputs = [
            source_text_numeric_features,
            destination_text_numeric_features,
            time_text_numeric_features,
        ]
        if any(value is not None for value in numeric_inputs):
            if any(value is None for value in numeric_inputs):
                raise ValueError(
                    "source, destination, and time text numeric features "
                    "must be provided together"
                )
            source_numeric = np.asarray(source_text_numeric_features, dtype="float32")
            destination_numeric = np.asarray(
                destination_text_numeric_features,
                dtype="float32",
            )
            time_numeric = np.asarray(time_text_numeric_features, dtype="float32")
            if (
                source_numeric.ndim != source.ndim or
                source_numeric.shape[:-1] != source.shape[:-1]
            ):
                raise ValueError(
                    "source_text_numeric_features must match source text prefix shape"
                )
            if (
                destination_numeric.ndim != destination.ndim or
                destination_numeric.shape[:-1] != destination.shape[:-1]
            ):
                raise ValueError(
                    "destination_text_numeric_features must match destination text prefix shape"
                )
            if time_numeric.ndim != 2 or time_numeric.shape[0] != time.shape[0]:
                raise ValueError(
                    "time_text_numeric_features must have shape [time,dim]"
                )
            if text_fusion_mode == "concat":
                source = np.concatenate([source, source_numeric], axis=-1)
                destination = np.concatenate(
                    [destination, destination_numeric],
                    axis=-1,
                )
                time = np.concatenate([time, time_numeric], axis=-1)
                source_numeric = None
                destination_numeric = None
                time_numeric = None
        elif text_fusion_mode == "gated_numeric":
            raise ValueError(
                "text_fusion_mode='%s' requires text numeric feature files" %
                text_fusion_mode
            )

        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.align_dim = int(align_dim)
        self.text_fusion_mode = text_fusion_mode
        self.numeric_alpha_init = float(numeric_alpha_init)
        self.alpha_init = float(alpha_init)
        self.static_source_destination_text = bool(static_source_destination)
        self.text_align_target_ratio = float(text_align_target_ratio)
        self.text_align_enabled = self.text_align_target_ratio > 0.0
        self.initial_text_align_weight = 1e-5 if self.text_align_enabled else 0.0
        self.alignment_temperature = float(alignment_temperature)
        self.temporal_delta = int(temporal_delta)
        self.target_start = int(target_start)
        self.text_time_len = int(time.shape[0])
        self.node_count = int(source.shape[0] if static_source_destination else source.shape[1])
        self.source_text_dim = int(source.shape[-1])
        self.destination_text_dim = int(destination.shape[-1])
        self.time_text_dim = int(time.shape[1])
        self.text_dim = self.source_text_dim
        self.source_text_embeddings = tf.constant(source, dtype=tf.float32)
        self.destination_text_embeddings = tf.constant(destination, dtype=tf.float32)
        self.time_text_embeddings = tf.constant(time, dtype=tf.float32)
        self.source_text_numeric_features = (
            None if source_numeric is None else
            tf.constant(source_numeric, dtype=tf.float32)
        )
        self.destination_text_numeric_features = (
            None if destination_numeric is None else
            tf.constant(destination_numeric, dtype=tf.float32)
        )
        self.time_text_numeric_features = (
            None if time_numeric is None else
            tf.constant(time_numeric, dtype=tf.float32)
        )

        self.source_proj_1 = k.layers.Dense(hidden_dim)
        self.source_proj_norm_1 = k.layers.LayerNormalization()
        self.source_proj_act = k.layers.Activation("gelu")
        self.source_proj_2 = k.layers.Dense(rank)
        self.source_proj_norm_2 = k.layers.LayerNormalization()
        self.destination_proj_1 = k.layers.Dense(hidden_dim)
        self.destination_proj_norm_1 = k.layers.LayerNormalization()
        self.destination_proj_act = k.layers.Activation("gelu")
        self.destination_proj_2 = k.layers.Dense(rank)
        self.destination_proj_norm_2 = k.layers.LayerNormalization()
        self.time_proj_1 = k.layers.Dense(hidden_dim)
        self.time_proj_norm_1 = k.layers.LayerNormalization()
        self.time_proj_act = k.layers.Activation("gelu")
        self.time_proj_2 = k.layers.Dense(rank)
        self.time_proj_norm_2 = k.layers.LayerNormalization()

        if self.text_fusion_mode == "gated_numeric":
            self.source_numeric_proj_1 = k.layers.Dense(hidden_dim)
            self.source_numeric_proj_norm_1 = k.layers.LayerNormalization()
            self.source_numeric_proj_act = k.layers.Activation("gelu")
            self.source_numeric_proj_2 = k.layers.Dense(rank)
            self.source_numeric_proj_norm_2 = k.layers.LayerNormalization()
            self.source_numeric_gate = k.layers.Dense(rank, activation="sigmoid")
            self.source_aux_norm = k.layers.LayerNormalization()

            self.destination_numeric_proj_1 = k.layers.Dense(hidden_dim)
            self.destination_numeric_proj_norm_1 = k.layers.LayerNormalization()
            self.destination_numeric_proj_act = k.layers.Activation("gelu")
            self.destination_numeric_proj_2 = k.layers.Dense(rank)
            self.destination_numeric_proj_norm_2 = k.layers.LayerNormalization()
            self.destination_numeric_gate = k.layers.Dense(rank, activation="sigmoid")
            self.destination_aux_norm = k.layers.LayerNormalization()

            self.time_numeric_proj_1 = k.layers.Dense(hidden_dim)
            self.time_numeric_proj_norm_1 = k.layers.LayerNormalization()
            self.time_numeric_proj_act = k.layers.Activation("gelu")
            self.time_numeric_proj_2 = k.layers.Dense(rank)
            self.time_numeric_proj_norm_2 = k.layers.LayerNormalization()
            self.time_numeric_gate = k.layers.Dense(rank, activation="sigmoid")
            self.time_aux_norm = k.layers.LayerNormalization()

        self.source_gate = k.layers.Dense(rank, activation="sigmoid")
        self.destination_gate = k.layers.Dense(rank, activation="sigmoid")
        self.time_gate = k.layers.Dense(rank, activation="sigmoid")
        self.source_norm = k.layers.LayerNormalization()
        self.destination_norm = k.layers.LayerNormalization()
        self.time_norm = k.layers.LayerNormalization()

        self.source_mode_proj = k.layers.Dense(align_dim)
        self.source_text_proj = k.layers.Dense(align_dim)
        self.destination_mode_proj = k.layers.Dense(align_dim)
        self.destination_text_proj = k.layers.Dense(align_dim)
        self.time_mode_proj = k.layers.Dense(align_dim)
        self.time_text_proj = k.layers.Dense(align_dim)

    def build(self, input_shape):
        def init_alpha(value):
            value = np.clip(value / 0.2, 1e-4, 1.0 - 1e-4)
            return float(np.log(value / (1.0 - value)))

        initializer = k.initializers.Constant(init_alpha(self.alpha_init))
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
        if self.text_fusion_mode == "gated_numeric":
            numeric_initializer = k.initializers.Constant(
                init_alpha(self.numeric_alpha_init)
            )
            self.source_numeric_alpha_logit = self.add_weight(
                name="source_numeric_alpha_logit",
                shape=(),
                initializer=numeric_initializer,
                trainable=True,
            )
            self.destination_numeric_alpha_logit = self.add_weight(
                name="destination_numeric_alpha_logit",
                shape=(),
                initializer=numeric_initializer,
                trainable=True,
            )
            self.time_numeric_alpha_logit = self.add_weight(
                name="time_numeric_alpha_logit",
                shape=(),
                initializer=numeric_initializer,
                trainable=True,
            )
        super().build(input_shape)

    def _project_source(self, x):
        x = self.source_proj_1(x)
        x = self.source_proj_norm_1(x)
        x = self.source_proj_act(x)
        x = self.source_proj_2(x)
        return self.source_proj_norm_2(x)

    def _project_destination(self, x):
        x = self.destination_proj_1(x)
        x = self.destination_proj_norm_1(x)
        x = self.destination_proj_act(x)
        x = self.destination_proj_2(x)
        return self.destination_proj_norm_2(x)

    def _project_time(self, x):
        x = self.time_proj_1(x)
        x = self.time_proj_norm_1(x)
        x = self.time_proj_act(x)
        x = self.time_proj_2(x)
        return self.time_proj_norm_2(x)

    def _project_source_numeric(self, x):
        x = self.source_numeric_proj_1(x)
        x = self.source_numeric_proj_norm_1(x)
        x = self.source_numeric_proj_act(x)
        x = self.source_numeric_proj_2(x)
        return self.source_numeric_proj_norm_2(x)

    def _project_destination_numeric(self, x):
        x = self.destination_numeric_proj_1(x)
        x = self.destination_numeric_proj_norm_1(x)
        x = self.destination_numeric_proj_act(x)
        x = self.destination_numeric_proj_2(x)
        return self.destination_numeric_proj_norm_2(x)

    def _project_time_numeric(self, x):
        x = self.time_numeric_proj_1(x)
        x = self.time_numeric_proj_norm_1(x)
        x = self.time_numeric_proj_act(x)
        x = self.time_numeric_proj_2(x)
        return self.time_numeric_proj_norm_2(x)

    def _fuse_gated_source(self, text_x, numeric_x):
        text_x = self._project_source(text_x)
        numeric_x = self._project_source_numeric(numeric_x)
        gate = self.source_numeric_gate(tf.concat([text_x, numeric_x], axis=-1))
        alpha = 0.2 * tf.sigmoid(self.source_numeric_alpha_logit)
        return self.source_aux_norm(text_x + alpha * gate * numeric_x)

    def _fuse_gated_destination(self, text_x, numeric_x):
        text_x = self._project_destination(text_x)
        numeric_x = self._project_destination_numeric(numeric_x)
        gate = self.destination_numeric_gate(
            tf.concat([text_x, numeric_x], axis=-1)
        )
        alpha = 0.2 * tf.sigmoid(self.destination_numeric_alpha_logit)
        return self.destination_aux_norm(text_x + alpha * gate * numeric_x)

    def _fuse_gated_time(self, text_x, numeric_x):
        text_x = self._project_time(text_x)
        numeric_x = self._project_time_numeric(numeric_x)
        gate = self.time_numeric_gate(tf.concat([text_x, numeric_x], axis=-1))
        alpha = 0.2 * tf.sigmoid(self.time_numeric_alpha_logit)
        return self.time_aux_norm(text_x + alpha * gate * numeric_x)

    def _masked_infonce(self, left, right, mask=None):
        left = tf.math.l2_normalize(left, axis=-1)
        right = tf.math.l2_normalize(right, axis=-1)
        logits = tf.matmul(left, right, transpose_b=True)
        logits = logits / self.alignment_temperature
        if mask is not None:
            logits = tf.where(mask, tf.ones_like(logits) * -1e9, logits)
        labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
        loss_a = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        loss_b = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )
        return 0.5 * (tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b))

    def _aggregate_by_key(self, left, right, keys):
        unique_keys, segment_ids = tf.unique(keys)
        count = tf.shape(unique_keys)[0]
        left = tf.math.unsorted_segment_mean(left, segment_ids, count)
        right = tf.math.unsorted_segment_mean(right, segment_ids, count)
        return left, right, unique_keys

    def _add_entity_alignment(self, mode_x, text_x, keys, weight,
                              mode_proj, text_proj, metric_prefix):
        if not self.text_align_enabled:
            return
        mode_x, text_x, unique_keys = self._aggregate_by_key(mode_x, text_x, keys)
        unique_count = tf.shape(unique_keys)[0]
        unique_count_float = tf.cast(unique_count, tf.float32)
        mask = None
        if not self.static_source_destination_text:
            entity_ids = unique_keys // self.text_time_len
            entity_times = unique_keys % self.text_time_len
            same_entity = entity_ids[:, None] == entity_ids[None, :]
            time_dist = tf.abs(entity_times[:, None] - entity_times[None, :])
            near_time = time_dist <= self.temporal_delta
            diagonal = tf.eye(tf.shape(unique_keys)[0], dtype=tf.bool)
            mask = tf.logical_and(
                tf.logical_and(same_entity, near_time),
                tf.logical_not(diagonal),
            )
        loss = tf.cond(
            unique_count < 2,
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: self._masked_infonce(
                mode_proj(mode_x),
                text_proj(text_x),
                mask=mask,
            ),
        )
        self.add_loss(weight * loss)
        self.add_metric(
            loss,
            name="%s_text_align_loss" % metric_prefix,
            aggregation="mean",
        )
        self.add_metric(
            weight * loss,
            name="%s_text_align_weighted_loss" % metric_prefix,
            aggregation="mean",
        )
        self.add_metric(
            unique_count_float,
            name="%s_text_align_unique_count" % metric_prefix,
            aggregation="mean",
        )

    def _add_time_alignment(self, mode_x, text_x, times):
        if not self.text_align_enabled:
            return
        mode_x, text_x, unique_times = self._aggregate_by_key(mode_x, text_x, times)
        unique_count = tf.shape(unique_times)[0]
        unique_count_float = tf.cast(unique_count, tf.float32)
        dist = tf.abs(unique_times[:, None] - unique_times[None, :])
        diagonal = tf.eye(tf.shape(unique_times)[0], dtype=tf.bool)
        near = tf.logical_and(dist <= self.temporal_delta, tf.logical_not(diagonal))
        loss = tf.cond(
            unique_count < 2,
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: self._masked_infonce(
                self.time_mode_proj(mode_x),
                self.time_text_proj(text_x),
                mask=near,
            ),
        )
        self.add_loss(self.time_text_align_weight_var * loss)
        self.add_metric(loss, name="time_text_align_loss", aggregation="mean")
        self.add_metric(
            self.time_text_align_weight_var * loss,
            name="time_text_align_weighted_loss",
            aggregation="mean",
        )
        self.add_metric(
            unique_count_float,
            name="time_text_align_unique_count",
            aggregation="mean",
        )

    def call(self, inputs):
        source_mode, destination_mode, time_mode, src_input, dst_input, time_input = inputs
        source_mode = tf.squeeze(source_mode, axis=1)
        destination_mode = tf.squeeze(destination_mode, axis=1)
        time_mode = tf.squeeze(time_mode, axis=1)
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        text_time_raw = time - self.target_start
        text_time = tf.clip_by_value(text_time_raw, 0, self.text_time_len - 1)
        clipped = tf.not_equal(text_time_raw, text_time)
        self.add_metric(
            tf.reduce_mean(tf.cast(clipped, tf.float32)),
            name="text_time_clipped_ratio",
            aggregation="mean",
        )

        if self.static_source_destination_text:
            source_text_raw = tf.gather(self.source_text_embeddings, src)
            destination_text_raw = tf.gather(self.destination_text_embeddings, dst)
        else:
            source_text_t = tf.gather(self.source_text_embeddings, text_time)
            source_text_raw = tf.gather_nd(
                source_text_t,
                tf.stack([tf.range(tf.shape(time)[0], dtype=tf.int32), src], axis=1),
            )
            destination_text_t = tf.gather(self.destination_text_embeddings, text_time)
            destination_text_raw = tf.gather_nd(
                destination_text_t,
                tf.stack([tf.range(tf.shape(time)[0], dtype=tf.int32), dst], axis=1),
            )
        time_text_raw = tf.gather(self.time_text_embeddings, text_time)

        if self.text_fusion_mode == "gated_numeric":
            if self.static_source_destination_text:
                source_numeric_raw = tf.gather(
                    self.source_text_numeric_features,
                    src,
                )
                destination_numeric_raw = tf.gather(
                    self.destination_text_numeric_features,
                    dst,
                )
            else:
                source_numeric_t = tf.gather(
                    self.source_text_numeric_features,
                    text_time,
                )
                source_numeric_raw = tf.gather_nd(
                    source_numeric_t,
                    tf.stack([tf.range(tf.shape(time)[0], dtype=tf.int32), src], axis=1),
                )
                destination_numeric_t = tf.gather(
                    self.destination_text_numeric_features,
                    text_time,
                )
                destination_numeric_raw = tf.gather_nd(
                    destination_numeric_t,
                    tf.stack([tf.range(tf.shape(time)[0], dtype=tf.int32), dst], axis=1),
                )
            time_numeric_raw = tf.gather(self.time_text_numeric_features, text_time)
            source_text = self._fuse_gated_source(
                source_text_raw,
                source_numeric_raw,
            )
            destination_text = self._fuse_gated_destination(
                destination_text_raw,
                destination_numeric_raw,
            )
            time_text = self._fuse_gated_time(time_text_raw, time_numeric_raw)
        else:
            source_text = self._project_source(source_text_raw)
            destination_text = self._project_destination(destination_text_raw)
            time_text = self._project_time(time_text_raw)
        source_alpha = 0.2 * tf.sigmoid(self.source_alpha_logit)
        destination_alpha = 0.2 * tf.sigmoid(self.destination_alpha_logit)
        time_alpha = 0.2 * tf.sigmoid(self.time_alpha_logit)
        source_gate = self.source_gate(
            tf.concat([source_mode, source_text], axis=-1)
        )
        destination_gate = self.destination_gate(
            tf.concat([destination_mode, destination_text], axis=-1)
        )
        time_gate = self.time_gate(tf.concat([time_mode, time_text], axis=-1))
        source_out = self.source_norm(
            source_mode + source_alpha * source_gate * source_text
        )
        destination_out = self.destination_norm(
            destination_mode + destination_alpha * destination_gate * destination_text
        )
        time_out = self.time_norm(time_mode + time_alpha * time_gate * time_text)

        if self.static_source_destination_text:
            source_keys = src
            destination_keys = dst
        else:
            source_keys = src * self.text_time_len + text_time
            destination_keys = dst * self.text_time_len + text_time
        self._add_entity_alignment(
            source_mode,
            source_text,
            source_keys,
            self.source_text_align_weight_var,
            self.source_mode_proj,
            self.source_text_proj,
            "source",
        )
        self._add_entity_alignment(
            destination_mode,
            destination_text,
            destination_keys,
            self.destination_text_align_weight_var,
            self.destination_mode_proj,
            self.destination_text_proj,
            "destination",
        )
        self._add_time_alignment(time_mode, time_text, text_time)

        return [
            source_out[:, None, :],
            destination_out[:, None, :],
            time_out[:, None, :],
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "hidden_dim": self.hidden_dim,
            "align_dim": self.align_dim,
            "alpha_init": self.alpha_init,
            "static_source_destination_text": self.static_source_destination_text,
            "text_align_target_ratio": self.text_align_target_ratio,
            "initial_text_align_weight": self.initial_text_align_weight,
            "alignment_temperature": self.alignment_temperature,
            "temporal_delta": self.temporal_delta,
            "target_start": self.target_start,
            "text_time_len": self.text_time_len,
            "node_count": self.node_count,
            "text_dim": self.text_dim,
            "text_fusion_mode": self.text_fusion_mode,
            "numeric_alpha_init": self.numeric_alpha_init,
            "source_text_dim": self.source_text_dim,
            "destination_text_dim": self.destination_text_dim,
            "time_text_dim": self.time_text_dim,
        })
        return config


class ODPathFeatureLayer(k.layers.Layer):
    """Inject OD-specific path features into the fused prediction state."""

    def __init__(self, od_path_features, hidden_dim=64, alpha_init=0.05,
                 **kwargs):
        super().__init__(**kwargs)
        features = np.asarray(od_path_features, dtype="float32")
        if features.ndim != 4:
            raise ValueError(
                "od_path_features must have shape [time, source, destination, dim]"
            )
        if features.shape[1] != features.shape[2]:
            raise ValueError("OD path features must use a square node dimension")
        if not np.all(np.isfinite(features)):
            raise ValueError("OD path features contain NaN or Inf values")
        self.od_path_features = tf.constant(features, dtype=tf.float32)
        self.time_len = int(features.shape[0])
        self.node_count = int(features.shape[1])
        self.feature_dim = int(features.shape[3])
        self.hidden_dim = int(hidden_dim)
        self.alpha_init = float(alpha_init)

        self.feature_proj_1 = k.layers.Dense(hidden_dim)
        self.feature_proj_norm_1 = k.layers.LayerNormalization()
        self.feature_proj_act = k.layers.Activation("gelu")
        self.feature_proj_norm_2 = k.layers.LayerNormalization()
        self.gate = None
        self.fuse_norm = k.layers.LayerNormalization()

    def build(self, input_shape):
        fused_dim = int(input_shape[0][-1])
        self.feature_proj_2 = k.layers.Dense(fused_dim)
        self.gate = k.layers.Dense(fused_dim, activation="sigmoid")

        value = np.clip(self.alpha_init / 0.2, 1e-4, 1.0 - 1e-4)
        initializer = k.initializers.Constant(
            float(np.log(value / (1.0 - value)))
        )
        self.alpha_logit = self.add_weight(
            name="od_path_alpha_logit",
            shape=(),
            initializer=initializer,
            trainable=True,
        )
        super().build(input_shape)

    def _project_features(self, features):
        x = self.feature_proj_1(features)
        x = self.feature_proj_norm_1(x)
        x = self.feature_proj_act(x)
        x = self.feature_proj_2(x)
        return self.feature_proj_norm_2(x)

    def call(self, inputs):
        fused, src_input, dst_input, time_input = inputs
        src = tf.cast(tf.reshape(src_input, [-1]), tf.int32)
        dst = tf.cast(tf.reshape(dst_input, [-1]), tf.int32)
        time_raw = tf.cast(tf.reshape(time_input, [-1]), tf.int32)
        time = tf.clip_by_value(time_raw, 0, self.time_len - 1)

        clipped = tf.not_equal(time_raw, time)
        self.add_metric(
            tf.reduce_mean(tf.cast(clipped, tf.float32)),
            name="od_path_time_clipped_ratio",
            aggregation="mean",
        )

        features_t = tf.gather(self.od_path_features, time)
        batch_ids = tf.range(tf.shape(time)[0], dtype=tf.int32)
        path_features = tf.gather_nd(
            features_t,
            tf.stack([batch_ids, src, dst], axis=1),
        )
        path_x = self._project_features(path_features)
        alpha = 0.2 * tf.sigmoid(self.alpha_logit)
        gate = self.gate(tf.concat([fused, path_x], axis=-1))
        out = self.fuse_norm(fused + alpha * gate * path_x)

        self.add_metric(alpha, name="od_path_alpha", aggregation="mean")
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "time_len": self.time_len,
            "node_count": self.node_count,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "alpha_init": self.alpha_init,
        })
        return config


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

        unique_time, inverse_time = tf.unique(time)
        adjacency = tf.gather(self.topology, unique_time)
        node_features = tf.broadcast_to(
            self.node_embeddings[None, :, :],
            [tf.shape(unique_time)[0], self.node_count, self.node_dim],
        )

        hidden = tf.matmul(adjacency, node_features)
        hidden = tf.matmul(hidden, self.gcn_kernel_1)
        hidden = tf.nn.relu(hidden)
        hidden = tf.matmul(adjacency, hidden)
        hidden = tf.matmul(hidden, self.gcn_kernel_2)
        hidden = tf.nn.relu(hidden)
        hidden = tf.gather(hidden, inverse_time)

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
                      gcn_dim=64, node_struct_features=None,
                      time_struct_features=None, structural_hidden_dim=64,
                      structural_align_dim=64, structural_beta=0.1,
                      structural_alpha=0.1, source_align_weight=0.0,
                      destination_align_weight=0.0, time_align_weight=0.0,
                      alignment_temperature=0.2, temporal_delta=2,
                      time_conditioned_modes=False,
                      source_text_embeddings=None,
                      destination_text_embeddings=None,
                      time_text_embeddings=None,
                      source_text_numeric_features=None,
                      destination_text_numeric_features=None,
                      time_text_numeric_features=None,
                      text_fusion_mode="concat",
                      numeric_alpha_init=0.1,
                      text_hidden_dim=64, text_align_dim=64,
                      text_alpha=0.1, text_align_target_ratio=0.0,
                      text_target_start=0,
                      od_path_features=None, od_path_hidden_dim=64,
                      od_path_alpha_init=0.05,
                      output_bias_init=0.0):
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
    if (
        time_conditioned_modes or
        node_struct_features is not None or
        time_struct_features is not None
    ):
        embeds = ModeWiseStructuralLayer(
            node_features=node_struct_features,
            time_features=time_struct_features,
            rank=rank,
            hidden_dim=structural_hidden_dim,
            align_dim=structural_align_dim,
            beta=structural_beta,
            alpha_init=structural_alpha,
            source_align_weight=source_align_weight,
            destination_align_weight=destination_align_weight,
            time_align_weight=time_align_weight,
            alignment_temperature=alignment_temperature,
            temporal_delta=temporal_delta,
            name="mode_wise_structural_alignment",
        )([
            embeds[0],
            embeds[1],
            embeds[2],
            inputs[0],
            inputs[1],
            inputs[2],
        ])
    text_enabled = (
        source_text_embeddings is not None or
        destination_text_embeddings is not None or
        time_text_embeddings is not None
    )
    if text_enabled:
        if (
            source_text_embeddings is None or
            destination_text_embeddings is None or
            time_text_embeddings is None
        ):
            raise ValueError(
                "source, destination, and time text embeddings must be provided together"
            )
        embeds = ModeWiseTextLayer(
            source_text_embeddings=source_text_embeddings,
            destination_text_embeddings=destination_text_embeddings,
            time_text_embeddings=time_text_embeddings,
            source_text_numeric_features=source_text_numeric_features,
            destination_text_numeric_features=destination_text_numeric_features,
            time_text_numeric_features=time_text_numeric_features,
            text_fusion_mode=text_fusion_mode,
            numeric_alpha_init=numeric_alpha_init,
            rank=rank,
            hidden_dim=text_hidden_dim,
            align_dim=text_align_dim,
            alpha_init=text_alpha,
            text_align_target_ratio=text_align_target_ratio,
            alignment_temperature=alignment_temperature,
            temporal_delta=temporal_delta,
            target_start=text_target_start,
            name="mode_wise_text_alignment",
        )([
            embeds[0],
            embeds[1],
            embeds[2],
            inputs[0],
            inputs[1],
            inputs[2],
        ])

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
    if od_path_features is not None:
        fused = ODPathFeatureLayer(
            od_path_features=od_path_features,
            hidden_dim=od_path_hidden_dim,
            alpha_init=od_path_alpha_init,
            name="od_path_feature_fusion",
        )([fused, inputs[0], inputs[1], inputs[2]])
    output = k.layers.Dense(
        1,
        activation=None,
        bias_initializer=k.initializers.Constant(output_bias_init),
        name="traffic_output",
    )(fused)
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
