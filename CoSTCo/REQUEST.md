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

Both normalized metrics use the original tensor value range as denominator:

```text
normalizer = max(tensor_values) - min(tensor_values)
NMAE = MAE / normalizer
NRMSE = RMSE / normalizer
```

## Required Environment

This extracted CoSTCo implementation follows the original KDD19 demo, which uses Python 2 and TensorFlow 1.x.

Recommended environment:

```bash
conda create -n costco python=2.7 pip
conda activate costco
pip install -r requirements.txt
```

`requirements.txt` installs:

```text
tensorflow==1.14.0
keras==2.2.4
numpy==1.16.6
h5py==2.10.0
pyyaml==5.4.1
```

For GPU execution, use a TensorFlow 1.14 compatible CUDA/cuDNN stack, typically CUDA 10.0 and cuDNN 7.x. If the server uses GPU TensorFlow, replace `tensorflow==1.14.0` with:

```text
tensorflow-gpu==1.14.0
```

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
