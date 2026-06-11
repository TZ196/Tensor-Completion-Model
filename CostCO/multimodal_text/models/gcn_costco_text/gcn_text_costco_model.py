import os
import sys

import numpy as np
from tensorflow import keras as k

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(SCRIPT_DIR)
MULTIMODAL_DIR = os.path.dirname(MODELS_DIR)
PROJECT_DIR = os.path.dirname(MULTIMODAL_DIR)
SHARED_DIR = os.path.join(MULTIMODAL_DIR, "shared")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from costco_model import mae, nmae, nrmse, rmse, transform_indices  # noqa: E402
from gcn_costco_model import (  # noqa: E402
    TemporalGCNPairLayer,
    compile_gcn_costco,
)
from mindtext_layers import (  # noqa: E402
    MindTextFusionLayer,
    TemporalSemanticAlignmentLayer,
    create_text_projection,
)


def create_mindtext_gcn_costco(shape, topology, endo_embeddings,
                               exo_embeddings, rank=50, nc=64,
                               node_dim=64, gcn_dim=128,
                               text_projection_dim=128,
                               text_stage="concat",
                               alignment_projection_dim=128,
                               alignment_temperature=0.2,
                               temporal_delta=2,
                               flow_text_loss_weight=0.0,
                               graph_text_loss_weight=0.0,
                               condenser_alpha=1.0,
                               condenser_epsilon=0.05,
                               condenser_loss_weight=0.0,
                               condenser_mu=0.5,
                               output_bias_init=0.0):
    shape = list(shape)
    if len(shape) != 3:
        raise ValueError("MindText-GCN-CoSTCo-Base expects a 3-D tensor")

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

    flow_x = k.layers.Concatenate(axis=1, name="flow_mode_concat")(embeds)
    flow_x = k.layers.Reshape(
        target_shape=(rank, len(shape), 1),
        name="flow_mode_reshape",
    )(flow_x)
    flow_x = k.layers.Conv2D(
        nc,
        kernel_size=(1, len(shape)),
        activation="relu",
        padding="valid",
        name="flow_conv_modes",
    )(flow_x)
    flow_x = k.layers.Conv2D(
        nc,
        kernel_size=(rank, 1),
        activation="relu",
        padding="valid",
        name="flow_conv_rank",
    )(flow_x)
    flow_x = k.layers.Flatten(name="flow_flatten")(flow_x)
    flow_x = k.layers.Dense(nc, activation="relu", name="flow_dense")(flow_x)

    gcn_pair = TemporalGCNPairLayer(
        topology=topology,
        node_dim=node_dim,
        gcn_dim=gcn_dim,
        name="temporal_gcn_pair",
    )(inputs)
    graph_x = k.layers.Dense(nc, activation="relu", name="gcn_pair_dense_1")(
        gcn_pair
    )
    graph_x = k.layers.Dense(nc, activation="relu", name="gcn_pair_dense_2")(
        graph_x
    )

    text_features = MindTextFusionLayer(
        endo_embeddings=endo_embeddings,
        exo_embeddings=exo_embeddings,
        stage=text_stage,
        hidden_dim=text_projection_dim,
        condenser_alpha=condenser_alpha,
        condenser_epsilon=condenser_epsilon,
        condenser_loss_weight=condenser_loss_weight,
        condenser_mu=condenser_mu,
        name="mindtext_fusion",
    )(inputs[2])
    text_x = create_text_projection(
        text_features,
        text_projection_dim,
        name_prefix="text",
    )
    if flow_text_loss_weight > 0.0 or graph_text_loss_weight > 0.0:
        text_x = TemporalSemanticAlignmentLayer(
            projection_dim=alignment_projection_dim,
            temperature=alignment_temperature,
            temporal_delta=temporal_delta,
            flow_text_weight=flow_text_loss_weight,
            graph_text_weight=graph_text_loss_weight,
            name="temporal_semantic_alignment",
        )([flow_x, graph_x, text_x, inputs[2]])

    fused = k.layers.Concatenate(name="flow_graph_text_fusion")([
        flow_x,
        graph_x,
        text_x,
    ])
    fused = k.layers.Dense(nc, activation="relu", name="fusion_dense_1")(
        fused
    )
    fused = k.layers.Dense(
        max(1, nc // 2),
        activation="relu",
        name="fusion_dense_2",
    )(fused)
    output = k.layers.Dense(
        1,
        activation=None,
        bias_initializer=k.initializers.Constant(output_bias_init),
        name="traffic_output",
    )(fused)
    return k.Model(
        inputs=inputs,
        outputs=output,
        name="MindText_GCN_CoSTCo",
    )


def create_mindtext_gcn_costco_base(shape, topology, endo_embeddings,
                                    exo_embeddings, rank=50, nc=64,
                                    node_dim=64, gcn_dim=128,
                                    text_projection_dim=128,
                                    output_bias_init=0.0):
    return create_mindtext_gcn_costco(
        shape,
        topology,
        endo_embeddings,
        exo_embeddings,
        rank=rank,
        nc=nc,
        node_dim=node_dim,
        gcn_dim=gcn_dim,
        text_projection_dim=text_projection_dim,
        text_stage="concat",
        output_bias_init=output_bias_init,
    )


def compile_mindtext_gcn_costco(model, lr=1e-4):
    return compile_gcn_costco(model, lr=lr)


def evaluate_mindtext_gcn_costco(model, indices, values, batch_size=1024,
                                 verbose=1, target_scale=1.0):
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
