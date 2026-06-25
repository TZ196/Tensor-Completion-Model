# NTC baseline

Runnable Neural Tensor Completion style baseline for sparse traffic tensor completion.

This implementation is a compact supervised tensor-completion model adapted to the
repository experiment protocol. It uses source, destination, and time embeddings
with nonlinear pairwise interactions and an MLP regression head.

Environment:

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

Run one experiment:

```bash
python run_sat_tensor_experiment.py --tensor-path ../data/iridium5400s/iridium.npy --observed-ratio 0.07 --metrics-path results/vis7_ntc_seed3.json
```
