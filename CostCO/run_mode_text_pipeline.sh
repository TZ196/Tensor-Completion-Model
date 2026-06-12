#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
TEXT_DIR="${TEXT_DIR:-mode_text_data}"
MODEL_PATH="${MODEL_PATH:-all-MiniLM-L6-v2}"
HISTORY_LEN="${HISTORY_LEN:-30}"
TARGET_START="${TARGET_START:-0}"
SEED="${SEED:-3}"
RANK="${RANK:-50}"
NC="${NC:-64}"
NODE_DIM="${NODE_DIM:-64}"
GCN_DIM="${GCN_DIM:-128}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-32}"
SOURCE_TEXT_ALIGN_WEIGHT="${SOURCE_TEXT_ALIGN_WEIGHT:-0.0005}"
DESTINATION_TEXT_ALIGN_WEIGHT="${DESTINATION_TEXT_ALIGN_WEIGHT:-0.0005}"
TIME_TEXT_ALIGN_WEIGHT="${TIME_TEXT_ALIGN_WEIGHT:-0.001}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"
RUN_PLAIN_TEXT="${RUN_PLAIN_TEXT:-1}"
RUN_ALIGNED_TEXT="${RUN_ALIGNED_TEXT:-1}"
ALLOW_REMOTE_MODEL="${ALLOW_REMOTE_MODEL:-0}"

mkdir -p logs results "$TEXT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/mode_text_pipeline_${timestamp}.log"
pid_file="logs/mode_text_pipeline.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started mode text pipeline in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
  exit 0
fi

echo "===== Mode Text GCN-CoSTCo Pipeline ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Tensor: $TENSOR_PATH"
echo "Topology: $TOPOLOGY_PATH"
echo "Text dir: $TEXT_DIR"
echo "Model path: $MODEL_PATH"
echo "Seed: $SEED"
echo

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

if [[ ! -f "$TENSOR_PATH" ]]; then
  echo "Missing tensor file: $TENSOR_PATH" >&2
  exit 1
fi
if [[ ! -f "$TOPOLOGY_PATH" ]]; then
  echo "Missing topology file: $TOPOLOGY_PATH" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" && "$ALLOW_REMOTE_MODEL" != "1" ]]; then
  echo "Missing local text encoder directory: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/path/to/all-MiniLM-L6-v2 or ALLOW_REMOTE_MODEL=1." >&2
  exit 1
fi

run_step "mode_text_build_records_seed${SEED}" \
  "$PYTHON_BIN" build_mode_texts.py \
  --tensor-path "$TENSOR_PATH" \
  --topology-path "$TOPOLOGY_PATH" \
  --output-dir "$TEXT_DIR" \
  --history-len "$HISTORY_LEN" \
  --target-start "$TARGET_START"

encode_cmd=(
  "$PYTHON_BIN" encode_mode_texts.py
  --text-dir "$TEXT_DIR"
  --model-path "$MODEL_PATH"
  --batch-size "$TEXT_BATCH_SIZE"
)
if [[ "$ALLOW_REMOTE_MODEL" == "1" ]]; then
  encode_cmd+=(--allow-remote-model)
fi
run_step "mode_text_encode_seed${SEED}" "${encode_cmd[@]}"

if [[ "$RUN_PLAIN_TEXT" == "1" ]]; then
  run_step "mode_text_gcn_costco_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    --use-mode-text \
    --mode-text-dir "$TEXT_DIR" \
    --rank "$RANK" --nc "$NC" \
    --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM" \
    --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --target-normalization max --seed "$SEED" \
    --metrics-path "results/mode_text_gcn_costco_seed${SEED}.json"
fi

if [[ "$RUN_ALIGNED_TEXT" == "1" ]]; then
  run_step "mode_text_gcn_costco_align_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    --use-mode-text \
    --mode-text-dir "$TEXT_DIR" \
    --source-text-align-weight "$SOURCE_TEXT_ALIGN_WEIGHT" \
    --destination-text-align-weight "$DESTINATION_TEXT_ALIGN_WEIGHT" \
    --time-text-align-weight "$TIME_TEXT_ALIGN_WEIGHT" \
    --alignment-temperature "$ALIGNMENT_TEMPERATURE" \
    --temporal-delta "$TEMPORAL_DELTA" \
    --rank "$RANK" --nc "$NC" \
    --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM" \
    --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --target-normalization max --seed "$SEED" \
    --metrics-path "results/mode_text_gcn_costco_align_seed${SEED}.json"
fi

echo "Pipeline finished at $(date)"
echo "Results:"
ls -1 results/mode_text_gcn_costco*seed"${SEED}".json 2>/dev/null || true
