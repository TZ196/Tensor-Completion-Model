#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-data/sat_path_bytes_mb_tensor.npy}"

SEED="${SEED:-3}"
VAL_RATIO="${VAL_RATIO:-0.1}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"

D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
TOP_K="${TOP_K:-2}"
NUM_KERNELS="${NUM_KERNELS:-6}"
DROPOUT="${DROPOUT:-0.1}"

VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})

mkdir -p logs results splits

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/timesnet_visible_rate_${timestamp}.log"
pid_file="logs/timesnet_visible_rate.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started TimesNet visible-rate experiments in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
  exit 0
fi

export PYTHONHASHSEED="$SEED"

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

echo "===== TimesNet Visible-Rate Experiments ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Epochs: $EPOCHS"
echo

for rate in "${VISIBLE_RATES[@]}"; do
  ratio="$(rate_ratio "$rate")"
  split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  metrics_path="results/vis${rate}_timesnet_seed${SEED}.json"

  run_step "vis${rate}_timesnet_seed${SEED}" \
    "$PYTHON_BIN" run_sat_tensor_experiment.py \
    --tensor-path "$TENSOR_PATH" \
    --observed-ratio "$ratio" \
    --val-ratio "$VAL_RATIO" \
    --split-path "$split_path" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --target-normalization "$TARGET_NORMALIZATION" \
    --seed "$SEED" \
    --d-model "$D_MODEL" \
    --d-ff "$D_FF" \
    --e-layers "$E_LAYERS" \
    --top-k "$TOP_K" \
    --num-kernels "$NUM_KERNELS" \
    --dropout "$DROPOUT" \
    --metrics-path "$metrics_path"
done

echo "===== TimesNet visible-rate experiments finished at $(date) ====="
"$PYTHON_BIN" - <<'PY'
import json
import os

rates = [int(v) for v in os.environ.get("VISIBLE_RATES", "1 3 5 7 10 20").split()]
seed = os.environ.get("SEED", "3")
print(f"{'Visible':>7s}  {'Model':12s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 47)
for rate in rates:
    path = f"results/vis{rate}_timesnet_seed{seed}.json"
    if not os.path.exists(path):
        print(f"{rate:>6d}%  {'TimesNet':12s}  {'missing':>10s}  {'missing':>10s}")
        continue
    with open(path) as f:
        payload = json.load(f)
    test = payload.get("test", payload)
    print(f"{rate:>6d}%  {'TimesNet':12s}  {test['nmae']:10.6f}  {test['nrmse']:10.6f}")
PY
