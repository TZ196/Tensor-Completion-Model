# SatFormer Local ASSIT

This sibling project keeps the original `Satfomer/` project unchanged and
experiments with a local sparse ASSIT implementation.

The runner reuses `../Satfomer/run_sat_tensor_experiment.py`, but loads this
directory's `satformer_model.py` first. If local data files are absent, it uses:

```text
../Satfomer/sat_path_bytes_mb_tensor.npy
../Satfomer/sat_connectivity_tensor_dynamic_60s_1000ms.npz
```

## Attention Modes

Dense baseline:

```bash
python3 -u run_sat_tensor_experiment.py --attention-mode dense
```

Local sparse ASSIT:

```bash
python3 -u run_sat_tensor_experiment.py \
  --attention-mode local \
  --attention-neighbor-size 3
```

`--attention-neighbor-size 3` means each query token attends to a local
`3 x 3 x 3` time-source-destination neighborhood inside each region.

This is an engineering implementation inspired by the paper's local region and
center-window sparse attention idea. It is not a claim that the paper used this
exact kernel.
