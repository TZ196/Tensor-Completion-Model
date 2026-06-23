#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-../CostCO/sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-../CostCO/sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
MODE_TEXT_DIR="${MODE_TEXT_DIR:-../CostCO/mode_text_numeric_ablation_data/both}"

SEED="${SEED:-3}"
D_MODEL="${D_MODEL:-64}"
NODE_DIM="${NODE_DIM:-64}"
GCN_DIM="${GCN_DIM:-128}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
FF_DIM="${FF_DIM:-128}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
TEXT_ALIGN_TARGET_RATIO="${TEXT_ALIGN_TARGET_RATIO:-0.01}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"

VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})
VARIANTS=(${VARIANTS:-M0 M1 M2 M3 M4 M5 M6})

mkdir -p logs results splits checkpoints histories

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/gt_mst_visible_rate_ablation_${timestamp}.log"
pid_file="logs/gt_mst_visible_rate_ablation.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started GT-MST visible-rate ablation in background."
  echo "PID: $pid"
  echo "Master log: $PROJECT_DIR/$master_log"
  echo "PID file: $PROJECT_DIR/$pid_file"
  echo "Watch: tail -f $PROJECT_DIR/$master_log"
  exit 0
fi

export PYTHONHASHSEED="$SEED"
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_ENABLE_ONEDNN_OPTS=0

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

rate_ratio() {
  local rate="$1"
  awk "BEGIN { printf \"%.2f\", ${rate}/100 }"
}

run_step() {
  local name="$1"
  shift
  local log_file="logs/${name}.log"
  echo "----- Running ${name} -----"
  echo "Command: $*"
  echo "Log: ${log_file}"
  "$@" > "$log_file" 2>&1
  echo "Finished ${name} at $(date)"
  echo
}

require_file "$TENSOR_PATH"
require_file "$TOPOLOGY_PATH"
for name in source_text_embeddings.npy destination_text_embeddings.npy time_text_embeddings.npy; do
  require_file "$MODE_TEXT_DIR/$name"
done

echo "===== GT-MST Visible-Rate Ablation ====="
echo "Start time: $(date)"
echo "Work dir: $PROJECT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Topology: $TOPOLOGY_PATH"
echo "Mode text dir: $MODE_TEXT_DIR"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Variants: ${VARIANTS[*]}"
echo

common_args=(
  --tensor-path "$TENSOR_PATH"
  --topology-path "$TOPOLOGY_PATH"
  --mode-text-dir "$MODE_TEXT_DIR"
  --d-model "$D_MODEL"
  --node-dim "$NODE_DIM"
  --gcn-dim "$GCN_DIM"
  --transformer-layers "$TRANSFORMER_LAYERS"
  --num-heads "$NUM_HEADS"
  --ff-dim "$FF_DIM"
  --dropout "$DROPOUT"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --target-normalization "$TARGET_NORMALIZATION"
  --seed "$SEED"
  --text-align-target-ratio "$TEXT_ALIGN_TARGET_RATIO"
  --alignment-temperature "$ALIGNMENT_TEMPERATURE"
  --temporal-delta "$TEMPORAL_DELTA"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
)

for rate in "${VISIBLE_RATES[@]}"; do
  ratio="$(rate_ratio "$rate")"
  split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  for variant in "${VARIANTS[@]}"; do
    name="vis${rate}_${variant}_gt_mst_seed${SEED}"
    run_step "$name" \
      "$PYTHON_BIN" run_experiment.py \
      "${common_args[@]}" \
      --variant "$variant" \
      --observed-ratio "$ratio" \
      --split-path "$split_path" \
      --metrics-path "results/${name}.json"
  done
done

echo "===== GT-MST visible-rate ablation finished at $(date) ====="
"$PYTHON_BIN" - <<'PY'
import json
import os

rates = [int(v) for v in os.environ.get("VISIBLE_RATES", "1 3 5 7 10 20").split()]
variants = os.environ.get("VARIANTS", "M0 M1 M2 M3 M4 M5 M6").split()
seed = os.environ.get("SEED", "3")
print(f"{'Visible':>7s}  {'Variant':>7s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 42)
for rate in rates:
    for variant in variants:
        path = f"results/vis{rate}_{variant}_gt_mst_seed{seed}.json"
        if not os.path.exists(path):
            print(f"{rate:>6d}%  {variant:>7s}  {'missing':>10s}  {'missing':>10s}")
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        test = payload["test"]
        print(f"{rate:>6d}%  {variant:>7s}  {test['nmae']:10.6f}  {test['nrmse']:10.6f}")
    print("-" * 42)
PY

