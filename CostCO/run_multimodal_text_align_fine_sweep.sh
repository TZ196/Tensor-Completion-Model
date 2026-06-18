#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
MODE_TEXT_DIR="${MODE_TEXT_DIR:-mode_text_numeric_ablation_data/both}"

SEED="${SEED:-3}"
RANK="${RANK:-50}"
NC="${NC:-64}"
NODE_DIM="${NODE_DIM:-64}"
GCN_DIM="${GCN_DIM:-128}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"

VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})

mkdir -p logs results splits checkpoints histories

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/multimodal_text_align_fine_sweep_${timestamp}.log"
pid_file="logs/multimodal_text_align_fine_sweep.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started multimodal text-align fine sweep in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
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

align_tag() {
  local ratio="$1"
  "$PYTHON_BIN" - "$ratio" <<'PY'
import sys
ratio = float(sys.argv[1])
print("align%04d" % round(ratio * 10000))
PY
}

align_ratios_for_rate() {
  local rate="$1"
  case "$rate" in
    1)
      echo "0.001 0.010 0.015 0.020"
      ;;
    3)
      echo "0.0025 0.005 0.0075"
      ;;
    5)
      echo "0.0075 0.010 0.015"
      ;;
    7)
      echo "0.0075 0.010 0.015"
      ;;
    10)
      echo "0.010 0.015 0.020"
      ;;
    20)
      echo "0.030 0.035 0.040"
      ;;
    *)
      echo "Unsupported visible rate: $rate" >&2
      exit 1
      ;;
  esac
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
for name in \
  source_text_embeddings.npy \
  destination_text_embeddings.npy \
  time_text_embeddings.npy
do
  require_file "$MODE_TEXT_DIR/$name"
done

echo "===== Multimodal TextAlign Fine Sweep ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Topology: $TOPOLOGY_PATH"
echo "Mode text dir: $MODE_TEXT_DIR"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Alignment temperature: $ALIGNMENT_TEMPERATURE"
echo "Temporal delta: $TEMPORAL_DELTA"
echo

common_args=(
  --tensor-path "$TENSOR_PATH"
  --topology-path "$TOPOLOGY_PATH"
  --rank "$RANK"
  --nc "$NC"
  --node-dim "$NODE_DIM"
  --gcn-dim "$GCN_DIM"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --target-normalization "$TARGET_NORMALIZATION"
  --seed "$SEED"
  --use-mode-text
  --mode-text-dir "$MODE_TEXT_DIR"
  --text-fusion-mode concat
  --alignment-temperature "$ALIGNMENT_TEMPERATURE"
  --temporal-delta "$TEMPORAL_DELTA"
)

for rate in "${VISIBLE_RATES[@]}"; do
  ratio="$(rate_ratio "$rate")"
  split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  for text_align_ratio in $(align_ratios_for_rate "$rate"); do
    tag="$(align_tag "$text_align_ratio")"
    run_step "vis${rate}_gcn_costco_text_fine_${tag}_seed${SEED}" \
      "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
      "${common_args[@]}" \
      --observed-ratio "$ratio" \
      --split-path "$split_path" \
      --text-align-target-ratio "$text_align_ratio" \
      --metrics-path "results/vis${rate}_gcn_costco_text_fine_${tag}_seed${SEED}.json"
  done
done

echo "===== Multimodal TextAlign fine sweep finished at $(date) ====="
"$PYTHON_BIN" - <<'PY'
import json
import os

rates = [int(v) for v in os.environ.get("VISIBLE_RATES", "1 3 5 7 10 20").split()]
ratio_map = {
    1: [0.001, 0.010, 0.015, 0.020],
    3: [0.0025, 0.005, 0.0075],
    5: [0.0075, 0.010, 0.015],
    7: [0.0075, 0.010, 0.015],
    10: [0.010, 0.015, 0.020],
    20: [0.030, 0.035, 0.040],
}
seed = os.environ.get("SEED", "3")
print(f"{'Visible':>7s}  {'Align':>8s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 44)
for rate in rates:
    for align in ratio_map[rate]:
        tag = "align%04d" % round(align * 10000)
        path = f"results/vis{rate}_gcn_costco_text_fine_{tag}_seed{seed}.json"
        if not os.path.exists(path):
            print(f"{rate:>6d}%  {align:8.4f}  {'missing':>10s}  {'missing':>10s}")
            continue
        with open(path) as f:
            payload = json.load(f)
        test = payload.get("test", payload)
        print(f"{rate:>6d}%  {align:8.4f}  {test['nmae']:10.6f}  {test['nrmse']:10.6f}")
PY
