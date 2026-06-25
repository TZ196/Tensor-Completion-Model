# PriSTI baseline

Runnable PriSTI-like spatiotemporal imputation baseline.

The original PriSTI is a conditional diffusion framework for spatiotemporal
imputation. This implementation keeps the spatiotemporal conditioning idea via
source, destination, time, and structural context tokens, adapted to the same
random tensor-completion protocol used by this repository.

Environment:

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

Run one experiment:

```bash
python run_sat_tensor_experiment.py --tensor-path ../data/iridium5400s/iridium.npy --observed-ratio 0.07 --metrics-path results/vis7_pristi_seed3.json
```
