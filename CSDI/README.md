# CSDI baseline

Runnable lightweight CSDI-like conditional denoising baseline.

The original CSDI is a conditional score-based diffusion model for probabilistic
time-series imputation. This implementation provides a compact denoising
variant adapted to sparse tensor-completion splits. It is intended as a runnable
baseline scaffold; the full official diffusion pipeline can be substituted later
without changing the run protocol.

Environment:

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

Run one experiment:

```bash
python run_sat_tensor_experiment.py --tensor-path ../data/iridium5400s/iridium.npy --observed-ratio 0.07 --metrics-path results/vis7_csdi_seed3.json
```
