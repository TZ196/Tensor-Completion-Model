# NTF baseline

Runnable Neural Tensor Factorization baseline for sparse traffic tensor completion.

This implementation combines CP-style trilinear factorization, mode biases, and a
nonlinear residual MLP.

Environment:

```bash
conda activate TZ-Satformer
pip install -r requirements.txt
```

Run one experiment:

```bash
python run_sat_tensor_experiment.py --tensor-path ../data/iridium5400s/iridium.npy --observed-ratio 0.07 --metrics-path results/vis7_ntf_seed3.json
```
