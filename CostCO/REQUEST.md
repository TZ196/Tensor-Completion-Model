# CoSTCo Tensor Completion Experiment Request

## Goal

Run the CoSTCo neural tensor completion model on `sat_path_bytes_tensor.npy`.

The dataset is a dense 3-D tensor with shape:

```text
120 x 120 x 60
```

The experiment follows a transductive tensor completion split:

- All non-zero finite entries from the full `N x N x T` tensor are considered.
- Original zero values are excluded from train/validation/test splits.
- `NaN`, `+Inf`, and `-Inf` values are also excluded.
- A random subset is sampled as observed training entries.
- Validation and test entries are sampled from the remaining unobserved entries.
- All time slices may appear in the training set, so time embeddings are trained
  for the same temporal range used by validation and testing.
- The split is generated automatically by `run_sat_tensor_experiment.py`.
- The generated split is saved to `splits/random_observed10_val10_seed_3.npz` for reproducibility.

The observed ratio controls how many tensor entries are used for training:

```text
--observed-ratio 0.1
```

With the default `--observed-ratio 0.1`, 10% of entries are observed and 90%
are unobserved. Of the unobserved entries, `--val-ratio 0.1` is used for
validation and the rest is used for final testing.

The equivalent missing-rate form is:

```text
--missing-rate 0.9
```

The requested test metrics are:

- NMAE
- NRMSE

MAPE is not used because traffic values can be zero or close to zero, which
can make percentage errors unstable or misleading.

The normalized metrics are computed over the held-out missing-entry test set:

```text
NMAE = sum(abs(y_true - y_pred)) / sum(abs(y_true))
NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))
```

Training targets are normalized by default using only the maximum value from
the training split:

```text
target_scale = max(train_values)
y_train_model = y_train_original / target_scale
```

Predictions are multiplied by `target_scale` before computing metrics, so all
reported MAE/RMSE/NMAE/NRMSE values are on the original target scale. To disable
target normalization:

```bash
python run_sat_tensor_experiment.py --target-normalization none
```

## Required Environment

Recommended environment:

```bash
conda create -n costco python=3.10 pip
conda activate costco
pip install -r requirements.txt
```

`requirements.txt` installs:

```text
tensorflow==2.15.0
numpy==1.26.4
```

For GPU execution, use a TensorFlow 2.15 compatible NVIDIA driver/CUDA setup. On Linux, the `tensorflow==2.15.0` pip package includes the needed CUDA runtime dependencies for supported NVIDIA GPU environments.

The code uses `tensorflow.keras`; no separate `keras` package is required.

## Run Command

From this directory:

```bash
python run_sat_tensor_experiment.py
```

CPU-only run:

```bash
python run_sat_tensor_experiment.py --cpu-only
```

Example with explicit hyperparameters:

```bash
python run_sat_tensor_experiment.py \
  --tensor-path sat_path_bytes_tensor.npy \
  --val-ratio 0.1 \
  --observed-ratio 0.1 \
  --rank 20 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-4 \
  --target-normalization max \
  --seed 3
```

## Outputs

The script writes:

```text
splits/random_observed10_val10_seed_3.npz
results/random_observed10_val10_seed3_rank20_nc20_norm_max.json
```

The result JSON contains train, validation, and test metrics. The final test
NMAE/NRMSE should be read from `test`.

Different observed/missing rates are saved to different result files by
default. For example:

```bash
python run_sat_tensor_experiment.py --missing-rate 0.9
python run_sat_tensor_experiment.py --missing-rate 0.8
```

will produce separate files in `results/`.
