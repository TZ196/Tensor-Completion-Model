# Tensor Completion Model

This repository stores satellite traffic tensor completion experiments in two
parallel project folders:

- `CostCO`: CoSTCo neural tensor completion baseline.
- `Satfomer`: SatFormer model using dynamic ISL topology.

Both projects use:

```text
sat_path_bytes_tensor.npy: 120 x 120 x 60
```

SatFormer additionally uses:

```text
sat_connectivity_tensor_dynamic_60s_1000ms.npz: 120 x 120 x 60
```

## SatFormer

From `Satfomer`:

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
```

Different observed ratios:

```bash
python run_sat_tensor_experiment.py --observed-ratio 0.02
python run_sat_tensor_experiment.py --observed-ratio 0.04
python run_sat_tensor_experiment.py --observed-ratio 0.06
python run_sat_tensor_experiment.py --observed-ratio 0.08
python run_sat_tensor_experiment.py --observed-ratio 0.10
```

## CoSTCo

From `CostCO`:

```bash
pip install -r requirements.txt
python run_sat_tensor_experiment.py
```
