#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-TZ-costco}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "Activated conda environment: $CONDA_ENV"
  else
    echo "conda not found; continuing with current Python."
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-../CostCO/sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-../CostCO/sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
MODE_TEXT_DIR="${MODE_TEXT_DIR:-../CostCO/mode_text_numeric_ablation_data/both}"
SEED="${SEED:-3}"
VISIBLE_RATES_RAW="${VISIBLE_RATES:-7 10 20}"
VARIANTS_RAW="${VARIANTS:-M7_dense M8_dense M9_dense}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-10}"
CHUNK_LEN="${CHUNK_LEN:-4}"
D_MODEL="${D_MODEL:-64}"
NODE_DIM="${NODE_DIM:-64}"
GCN_DIM="${GCN_DIM:-128}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
FF_DIM="${FF_DIM:-128}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-3}"
TEXT_ALIGN_WEIGHT="${TEXT_ALIGN_WEIGHT:-0.0001}"

read -r -a VISIBLE_RATES <<< "$VISIBLE_RATES_RAW"
read -r -a VARIANTS <<< "$VARIANTS_RAW"

mkdir -p logs results splits checkpoints histories
MASTER_LOG="logs/dense_text_tokens_$(date +%Y%m%d_%H%M%S).log"

run_one() {
  local rate="$1"
  local variant="$2"
  local ratio
  ratio="$("$PYTHON_BIN" - <<PY
print(float($rate) / 100.0)
PY
)"
  local name="vis${rate}_${variant}_gt_mst_dense_seed${SEED}"
  local split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  local metrics_path="results/${name}.json"
  local log_path="logs/${name}.log"

  echo
  echo "----- Running ${name} -----"
  echo "Log: ${log_path}"
  "$PYTHON_BIN" run_dense_experiment.py \
    --tensor-path "$TENSOR_PATH" \
    --topology-path "$TOPOLOGY_PATH" \
    --mode-text-dir "$MODE_TEXT_DIR" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --variant "$variant" \
    --chunk-len "$CHUNK_LEN" \
    --d-model "$D_MODEL" \
    --node-dim "$NODE_DIM" \
    --gcn-dim "$GCN_DIM" \
    --transformer-layers "$TRANSFORMER_LAYERS" \
    --num-heads "$NUM_HEADS" \
    --ff-dim "$FF_DIM" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --early-stopping-patience "$PATIENCE" \
    --target-normalization max \
    --seed "$SEED" \
    --text-align-weight "$TEXT_ALIGN_WEIGHT" \
    --metrics-path "$metrics_path" \
    > "$log_path" 2>&1
  echo "Finished ${name}"
}

{
  echo "===== Dense Independent Text Token GT-MST Experiments ====="
  echo "Start time: $(date)"
  echo "Work dir: $PROJECT_DIR"
  echo "Python: $PYTHON_BIN"
  echo "Tensor: $TENSOR_PATH"
  echo "Topology: $TOPOLOGY_PATH"
  echo "Mode text dir: $MODE_TEXT_DIR"
  echo "Seed: $SEED"
  echo "Visible rates: ${VISIBLE_RATES[*]}"
  echo "Variants: ${VARIANTS[*]}"
  echo "Chunk length: $CHUNK_LEN"
  echo "Epochs: $EPOCHS"
  echo "Patience: $PATIENCE"

  for rate in "${VISIBLE_RATES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      run_one "$rate" "$variant"
    done
  done

  echo
  echo "===== Dense run finished at $(date) ====="
  "$PYTHON_BIN" - <<PY
import json, os

rates = "${VISIBLE_RATES_RAW}".split()
variants = "${VARIANTS_RAW}".split()
labels = {
    "M7_dense": "IndependentTextTokens",
    "M8_dense": "TextTokens+NumericControl",
    "M9_dense": "TextTokens+NumericControl+TextAlign",
}
print(f"{'Visible':>7s}  {'Variant':>9s}  {'Model':38s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 82)
for rate in rates:
    for variant in variants:
        path = f"results/vis{rate}_{variant}_gt_mst_dense_seed${SEED}.json"
        label = labels.get(variant, variant)
        if not os.path.exists(path):
            print(f"{rate + '%':>7s}  {variant:>9s}  {label:38s}  {'missing':>10s}  {'missing':>10s}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        test = payload.get("test", {})
        print(f"{rate + '%':>7s}  {variant:>9s}  {label:38s}  {test.get('nmae', float('nan')):10.6f}  {test.get('nrmse', float('nan')):10.6f}")
    print("-" * 82)
PY
} | tee "$MASTER_LOG"

if [[ "${SKIP_CONDA_DEACTIVATE:-0}" != "1" && "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  conda deactivate || true
fi
