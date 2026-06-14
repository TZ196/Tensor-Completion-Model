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

SOURCE_TEXT_ALIGN_WEIGHT="${SOURCE_TEXT_ALIGN_WEIGHT:-0.0005}"
DESTINATION_TEXT_ALIGN_WEIGHT="${DESTINATION_TEXT_ALIGN_WEIGHT:-0.0005}"
TIME_TEXT_ALIGN_WEIGHT="${TIME_TEXT_ALIGN_WEIGHT:-0.001}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"

VISIBLE_RATES=(${VISIBLE_RATES:-0.01 0.03 0.05 0.07 0.10 0.20})

mkdir -p logs results checkpoints histories

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/visible_rate_18_experiments_${timestamp}.log"
pid_file="logs/visible_rate_18_experiments.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started 18 visible-rate experiments in background."
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

rate_tag() {
  local rate="$1"
  "$PYTHON_BIN" - "$rate" <<'PY'
import sys
rate = float(sys.argv[1])
print("vis%02d" % round(rate * 100))
PY
}

run_step() {
  local name="$1"
  shift
  local log_file="logs/${name}.log"
  echo "----- Running ${name} -----"
  echo "Log: ${log_file}"
  "$@" > "$log_file" 2>&1
  echo "Finished ${name} at $(date)"
  echo
}

echo "===== Visible-Rate 18 Experiments ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Topology: $TOPOLOGY_PATH"
echo "Mode text dir: $MODE_TEXT_DIR"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Text align weights: source=$SOURCE_TEXT_ALIGN_WEIGHT destination=$DESTINATION_TEXT_ALIGN_WEIGHT time=$TIME_TEXT_ALIGN_WEIGHT"
echo

for rate in "${VISIBLE_RATES[@]}"; do
  tag="$(rate_tag "$rate")"

  run_step "${tag}_costco_seed${SEED}" \
    "$PYTHON_BIN" run_sat_tensor_experiment.py \
    --tensor-path "$TENSOR_PATH" \
    --observed-ratio "$rate" \
    --rank "$RANK" \
    --nc "$NC" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --target-normalization "$TARGET_NORMALIZATION" \
    --seed "$SEED" \
    --metrics-path "results/${tag}_costco_seed${SEED}.json"

  run_step "${tag}_gcn_costco_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    --tensor-path "$TENSOR_PATH" \
    --topology-path "$TOPOLOGY_PATH" \
    --observed-ratio "$rate" \
    --struct-feature-group none \
    --rank "$RANK" \
    --nc "$NC" \
    --node-dim "$NODE_DIM" \
    --gcn-dim "$GCN_DIM" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --target-normalization "$TARGET_NORMALIZATION" \
    --seed "$SEED" \
    --metrics-path "results/${tag}_gcn_costco_seed${SEED}.json"

  run_step "${tag}_gcn_costco_text_align_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    --tensor-path "$TENSOR_PATH" \
    --topology-path "$TOPOLOGY_PATH" \
    --observed-ratio "$rate" \
    --struct-feature-group none \
    --use-mode-text \
    --mode-text-dir "$MODE_TEXT_DIR" \
    --text-fusion-mode concat \
    --source-text-align-weight "$SOURCE_TEXT_ALIGN_WEIGHT" \
    --destination-text-align-weight "$DESTINATION_TEXT_ALIGN_WEIGHT" \
    --time-text-align-weight "$TIME_TEXT_ALIGN_WEIGHT" \
    --alignment-temperature "$ALIGNMENT_TEMPERATURE" \
    --temporal-delta "$TEMPORAL_DELTA" \
    --rank "$RANK" \
    --nc "$NC" \
    --node-dim "$NODE_DIM" \
    --gcn-dim "$GCN_DIM" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --target-normalization "$TARGET_NORMALIZATION" \
    --seed "$SEED" \
    --metrics-path "results/${tag}_gcn_costco_text_align_seed${SEED}.json"
done

echo "Pipeline finished at $(date)"
echo "Results:"
ls -1 results/vis*_costco_seed"${SEED}".json \
      results/vis*_gcn_costco_seed"${SEED}".json \
      results/vis*_gcn_costco_text_align_seed"${SEED}".json 2>/dev/null || true
