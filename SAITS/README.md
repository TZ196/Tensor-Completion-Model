# SAITS baseline

Runnable SAITS-like self-attention imputation baseline adapted to sparse tensor
completion.

The original SAITS is a self-attention imputation model for multivariate time
series. This repository version keeps the self-attention imputation idea but
uses source, destination, and time mode tokens so it can run on the same random
tensor-completion splits as CoSTCo/TimesNet/ModernTCN.

Environment:

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

Run one experiment:

```bash
python run_sat_tensor_experiment.py --tensor-path ../data/iridium5400s/iridium.npy --observed-ratio 0.07 --metrics-path results/vis7_saits_seed3.json
```
