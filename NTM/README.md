# NTM baseline

Runnable Nonlinear Tensor Machine style baseline.

This implementation uses source-destination bilinear tensor interactions plus a
time embedding and nonlinear regression head. It is designed from the NTM
concept because a reliable official implementation is not available in this
repository.

Environment:

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

Run one experiment:

```bash
python run_sat_tensor_experiment.py --tensor-path ../data/iridium5400s/iridium.npy --observed-ratio 0.07 --metrics-path results/vis7_ntm_seed3.json
```
