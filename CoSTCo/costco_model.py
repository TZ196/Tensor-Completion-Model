import numpy as np
import tensorflow as tf
from tensorflow import keras as k


def mape_keras(y_true, y_pred, threshold=0.1):
    v = k.backend.clip(k.backend.abs(y_true), threshold, None)
    diff = k.backend.abs((y_true - y_pred) / v)
    return 100.0 * k.backend.mean(diff, axis=-1)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def nmae(y_true, y_pred, normalizer):
    return mae(y_true, y_pred) / normalizer


def nrmse(y_true, y_pred, normalizer):
    return rmse(y_true, y_pred) / normalizer


def mape(y_true, y_pred, threshold=0.1):
    v = np.clip(np.abs(y_true), threshold, None)
    diff = np.abs((y_true - y_pred) / v)
    return 100.0 * np.mean(diff, axis=-1).mean()


def transform_indices(indices):
    """Convert an (n_samples, n_modes) index matrix to Keras input arrays."""
    return [indices[:, i] for i in range(indices.shape[1])]


def configure_tensorflow(cpu_only=False, seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    if cpu_only:
        tf.config.set_visible_devices([], "GPU")
        return []

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
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


def compile_costco(model, lr=1e-4):
    optimizer = k.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer, loss="mse", metrics=["mae", mape_keras])
    return model


def evaluate_costco(model, indices, values, batch_size=1024, verbose=1,
                    normalizer=None):
    pred = model.predict(
        transform_indices(indices),
        batch_size=batch_size,
        verbose=verbose
    ).flatten()
    metrics = {
        "rmse": float(rmse(values, pred)),
        "mape": float(mape(values, pred)),
        "mae": float(mae(values, pred))
    }
    if normalizer is not None:
        metrics["nmae"] = float(nmae(values, pred, normalizer))
        metrics["nrmse"] = float(nrmse(values, pred, normalizer))
    return metrics
