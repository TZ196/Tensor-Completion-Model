# SatFormer Tensor Completion

This project runs SatFormer on the satellite inter-satellite path traffic tensor:

```text
sat_path_bytes_tensor.npy: 120 x 120 x 60
```

The dynamic ISL topology tensor is loaded from:

```text
sat_connectivity_tensor_dynamic_60s_1000ms.npz
```

The topology tensor has the same shape, `120 x 120 x 60`; value `1` means two
satellites are adjacent at that time step, and `0` means no ISL.

## Model

The implementation follows the paper-level SatFormer structure:

```text
sparse observed tensor
-> Encoder Spatio-Temporal Modules
-> Transfer Module
-> Decoder Spatio-Temporal Modules
-> completed tensor
```

Each Spatio-Temporal Module contains:

- two-layer normalized GCN over the dynamic ISL topology;
- SatFormer block with `LN -> ASSIT -> LN -> MLP`;
- residual connections.

ASSIT uses local regions, a center-window mask, multi-head attention, and
adaptive sparse gating. The Transfer Module uses temporal self-attention with
learnable history/current weights.

The distance/time-delay weight matrix `W` is not used here because
`sat_path_bytes_tensor.npy` already stores the real inter-satellite path traffic.

## Split And Metrics

The experiment follows the same random transductive tensor completion protocol
as the CoSTCo reference project:

- finite non-zero entries are eligible samples;
- original zero values are excluded from train/validation/test splits;
- a random observed subset is used as training entries;
- validation and test entries are sampled from the remaining unobserved entries;
- splits are saved under `splits/`;
- metrics are saved under `results/`.

Reported metrics include NMAE and NRMSE:

```text
NMAE  = sum(abs(y_true - y_pred)) / sum(abs(y_true))
NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))
```

Metrics are computed on the original traffic scale. Training targets are
normalized by `max(train_values)` by default.

Training uses the paper-style settings by default:

- Adam optimizer;
- learning rate `0.001`;
- weight decay `1e-5`;
- training updates are grouped by target time step to avoid repeated full-window
  forward passes;
- warmup for the first `5` epochs;
- early stopping with patience `10`;
- no 10-run averaging.

The training loop is grouped by target time step. For one optimizer update, it
builds the history window for one target time step, predicts that time-step
traffic matrix once, and computes loss on all observed training entries from
that target time step. It no longer performs one full `120 x 120 x 60`
reconstruction backward pass per epoch, and it no longer repeats the same
time-step forward pass for many entry batches. The temporal Transfer Module
uses a history window; the default is `8` time steps. Use `--history-window 0`
for full history.

## Environment

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python run_sat_tensor_experiment.py
```

CPU-only:

```bash
python run_sat_tensor_experiment.py --cpu-only
```

Equivalent old entry:

```bash
python Satformer.py
```

Example with explicit settings:

```bash
python run_sat_tensor_experiment.py \
  --tensor-path sat_path_bytes_tensor.npy \
  --adjacency-path sat_connectivity_tensor_dynamic_60s_1000ms.npz \
  --observed-ratio 0.1 \
  --val-ratio 0.1 \
  --feature-dim 128 \
  --gcn-hidden-dim 128 \
  --num-modules 10 \
  --region-size 16 \
  --center-window 16 \
  --heads 8 \
  --batch-size 128 \
  --history-window 8 \
  --warmup-epochs 5 \
  --epochs 200 \
  --lr 0.001 \
  --target-normalization max \
  --seed 3
```

Gradient checkpointing is enabled by default to keep the 10-layer
encoder/decoder model inside GPU memory. To disable it for debugging:

```bash
python run_sat_tensor_experiment.py --no-gradient-checkpointing
```

Quick debug run:

```bash
python run_sat_tensor_experiment.py \
  --missing-rate 0.90 \
  --epochs 1 \
  --max-train-steps-per-epoch 1 \
  --log-every 1
```

Different sampling rates:

```bash
python run_sat_tensor_experiment.py --observed-ratio 0.02
python run_sat_tensor_experiment.py --observed-ratio 0.04
python run_sat_tensor_experiment.py --observed-ratio 0.06
python run_sat_tensor_experiment.py --observed-ratio 0.08
python run_sat_tensor_experiment.py --observed-ratio 0.10
```

Or use missing-rate form:

```bash
python run_sat_tensor_experiment.py --missing-rate 0.9
```

## Outputs

Default outputs look like:

```text
splits/random_observed10_val10_seed_3.npz
results/random_observed10_val10_seed3_dim128_layers10_norm_max.json
```

Read final NMAE and NRMSE from the `test` section of the result JSON.
