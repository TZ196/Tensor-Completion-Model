# CoSTCo Tensor Completion Experiment Request

## Goal

Run the CoSTCo neural tensor completion model on `sat_path_bytes_tensor.npy`.

The dataset is a dense 3-D tensor with shape:

```text
120 x 120 x 60
```

The experiment follows a Satformer-style temporal split plus random masking:

- The first 80% time slices are used as the train/validation period.
- Within that period, the last 10% is used for validation.
- The last 20% time slices are used for testing.
- The split is generated automatically by `run_sat_tensor_experiment.py`.
- The generated split is saved to `splits/temporal_train80_val10_observed10_seed_3.npz` for reproducibility.

Inside train, validation, and test periods, a configurable random mask is
applied to simulate incomplete traffic observations:

```text
--observed-ratio 0.1
```

With the default `--observed-ratio 0.1`, 10% of entries are observed and 90%
are masked. The model is trained using train observed entries. Validation and
test metrics are computed only on the masked/unobserved entries in their
respective time periods.

The equivalent missing-rate form is:

```text
--missing-rate 0.9
```

For the `120 x 120 x 60` tensor, the default split is:

```text
train: t = 0..42
val:   t = 43..47
test:  t = 48..59
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
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --observed-ratio 0.1 \
  --rank 20 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 3
```

## Outputs

The script writes:

```text
splits/temporal_train80_val10_observed10_seed_3.npz
results_sat_costco.json
```

`results_sat_costco.json` contains metrics for train observed, train missing,
validation missing, and test missing entries. The final test NMAE/NRMSE should
be read from `test_missing`.
