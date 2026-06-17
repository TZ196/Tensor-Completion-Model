# ModernTCN Tensor Completion Baseline

This baseline runs ModernTCN as a masked imputation model for satellite path traffic tensor completion.

The default data file is:

```text
data/sat_path_bytes_mb_tensor.npy
```

Expected tensor shape:

```text
source_satellite x destination_satellite x time
```

## Run

From this directory:

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
```

Common options:

```bash
python run_sat_tensor_experiment.py --observed-ratio 0.10
python run_sat_tensor_experiment.py --missing-rate 0.90
python run_sat_tensor_experiment.py --epochs 100 --seed 3
```

Outputs are written to:

```text
results/
splits/
```

Metrics use the same definitions as the other baselines:

```text
NMAE  = sum(abs(y_true - y_pred)) / sum(abs(y_true))
NRMSE = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))
```
