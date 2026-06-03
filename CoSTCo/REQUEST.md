# CoSTCo Tensor Completion Experiment Request

## Goal

Run the CoSTCo neural tensor completion model on `sat_path_bytes_tensor.npy`.

The dataset is a dense 3-D tensor with shape:

```text
120 x 120 x 60
```

The experiment uses a 90% missing rate:

- 10% finite tensor entries are used as the training observed entries.
- 90% finite tensor entries are held out as the test missing entries.
- The split is random but reproducible with `--seed 3`.
- The split is saved to `splits/missing_90_seed_3.npz`.
- This repository already includes the prepared split file after running `prepare_sat_split.py`.

The requested test metrics are:

- NMAE
- NRMSE

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

If the split file needs to be regenerated:

```bash
python prepare_sat_split.py
```

CPU-only run:

```bash
python run_sat_tensor_experiment.py --cpu-only
```

Example with explicit hyperparameters:

```bash
python run_sat_tensor_experiment.py \
  --tensor-path sat_path_bytes_tensor.npy \
  --missing-rate 0.9 \
  --rank 20 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 3
```

## Outputs

The script writes:

```text
splits/missing_90_seed_3.npz
results_sat_costco.json
```

`results_sat_costco.json` contains the train/test metrics, including test NMAE and test NRMSE.
