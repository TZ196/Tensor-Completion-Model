import os
import random

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras as k


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def nmae(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))
    if denominator == 0.0:
        raise ValueError("Cannot compute NMAE: sum(abs(y_true)) is 0")
    return np.sum(np.abs(y_true - y_pred)) / denominator


def nrmse(y_true, y_pred):
    denominator = np.sum(np.square(y_true))
    if denominator == 0.0:
        raise ValueError("Cannot compute NRMSE: sum(square(y_true)) is 0")
    return np.sqrt(np.sum(np.square(y_true - y_pred)) / denominator)


def transform_indices(indices):
    """Convert an (n_samples, n_modes) index matrix to Keras input arrays."""
    return [indices[:, i] for i in range(indices.shape[1])]


def configure_tensorflow(cpu_only=False, seed=0, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:
            print("Could not enable TensorFlow op determinism:", exc)
        for setter in (
            tf.config.threading.set_inter_op_parallelism_threads,
            tf.config.threading.set_intra_op_parallelism_threads,
        ):
            try:
                setter(1)
            except RuntimeError as exc:
                print("Could not set TensorFlow thread determinism:", exc)

    gpus = tf.config.list_physical_devices("GPU")
    if cpu_only:
        tf.config.set_visible_devices([], "GPU")
        return []

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("Could not set GPU memory growth:", exc)
    return gpus


def create_costco(shape, rank=20, nc=None):
    """Create the CoSTCo neural tensor completion model.

    Args:
        shape: Iterable of tensor mode sizes, e.g. [100, 50, 24].
        rank: Embedding dimension for every tensor mode.
        nc: Number of convolution filters / hidden units. Defaults to rank.

    Returns:
        A compiled-agnostic Keras Model. Call model.compile(...) before train.
    """
    if nc is None:
        nc = rank

    shape = list(shape)
    inputs = [k.Input(shape=(1,), dtype="int32") for _ in range(len(shape))]
    embeds = [
        k.layers.Embedding(output_dim=rank, input_dim=shape[i])(inputs[i])
        for i in range(len(shape))
    ]

    x = k.layers.Concatenate(axis=1)(embeds)
    x = k.layers.Reshape(target_shape=(rank, len(shape), 1))(x)
    x = k.layers.Conv2D(
        nc,
        kernel_size=(1, len(shape)),
        activation="relu",
        padding="valid"
    )(x)
    x = k.layers.Conv2D(
        nc,
        kernel_size=(rank, 1),
        activation="relu",
        padding="valid"
    )(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(nc, activation="relu")(x)
    outputs = k.layers.Dense(1, activation="relu")(x)
    return k.Model(inputs=inputs, outputs=outputs)


def create_topo_costco(shape, rank=20, nc=None, topo_dim=7):
    """Create a topology-aware CoSTCo model.

    The index branch is the original CoSTCo embedding/convolution stack. The
    topology branch learns numeric features extracted from A_t for each
    (source, destination, time) sample.
    """
    if nc is None:
        nc = rank

    shape = list(shape)
    index_inputs = [
        k.Input(shape=(1,), dtype="int32", name="mode_%d_input" % i)
        for i in range(len(shape))
    ]
    embeds = [
        k.layers.Embedding(
            output_dim=rank,
            input_dim=shape[i],
            name="mode_%d_embedding" % i,
        )(index_inputs[i])
        for i in range(len(shape))
    ]

    x = k.layers.Concatenate(axis=1)(embeds)
    x = k.layers.Reshape(target_shape=(rank, len(shape), 1))(x)
    x = k.layers.Conv2D(
        nc,
        kernel_size=(1, len(shape)),
        activation="relu",
        padding="valid"
    )(x)
    x = k.layers.Conv2D(
        nc,
        kernel_size=(rank, 1),
        activation="relu",
        padding="valid"
    )(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(nc, activation="relu")(x)

    topo_input = k.Input(
        shape=(topo_dim,),
        dtype="float32",
        name="topology_features",
    )
    topo_x = k.layers.Dense(nc, activation="relu")(topo_input)
    topo_x = k.layers.Dense(nc, activation="relu")(topo_x)

    fused = k.layers.Concatenate()([x, topo_x])
    fused = k.layers.Dense(nc, activation="relu")(fused)
    fused = k.layers.Dense(max(1, nc // 2), activation="relu")(fused)
    outputs = k.layers.Dense(1, activation="relu")(fused)
    return k.Model(inputs=index_inputs + [topo_input], outputs=outputs)


def compile_costco(model, lr=1e-4):
    optimizer = k.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer, loss="mse", metrics=["mae"])
    return model


def evaluate_costco(model, indices, values, batch_size=1024, verbose=1,
                    target_scale=1.0):
    pred = model.predict(
        transform_indices(indices),
        batch_size=batch_size,
        verbose=verbose
    ).flatten()
    pred = pred * target_scale
    pred = np.maximum(pred, 0.0)
    metrics = {
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
    return metrics


def evaluate_topo_costco(model, indices, values, topo_features,
                         batch_size=1024, verbose=1, target_scale=1.0):
    pred = model.predict(
        transform_indices(indices) + [topo_features],
        batch_size=batch_size,
        verbose=verbose
    ).flatten()
    pred = pred * target_scale
    pred = np.maximum(pred, 0.0)
    metrics = {
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
    return metrics
