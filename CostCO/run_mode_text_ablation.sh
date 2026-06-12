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
ALLOW_REMOTE_MODEL="${ALLOW_REMOTE_MODEL:-0}"

mkdir -p logs results "$TEXT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/mode_text_ablation_${timestamp}.log"
pid_file="logs/mode_text_ablation.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started mode text ablation in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
  exit 0
fi

echo "===== Mode Text Ablation Pipeline ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
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

ensure_text_embeddings() {
  if [[ -f "$TEXT_DIR/source_text_embeddings.npy" \
        && -f "$TEXT_DIR/destination_text_embeddings.npy" \
        && -f "$TEXT_DIR/time_text_embeddings.npy" ]]; then
    echo "Text embeddings already exist in $TEXT_DIR"
    return
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
}

base_args=(
  --rank "$RANK" --nc "$NC"
  --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM"
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --target-normalization max --seed "$SEED"
)

text_args=(
  --use-mode-text
  --mode-text-dir "$TEXT_DIR"
)

ensure_text_embeddings

run_step "text_ablation_T0_gcn_baseline_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  --struct-feature-group none \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T0_gcn_baseline_seed${SEED}.json"

run_step "text_ablation_T1_text_no_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${text_args[@]}" \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T1_text_no_align_seed${SEED}.json"

run_step "text_ablation_T2_time_text_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${text_args[@]}" \
  --time-text-align-weight 0.001 \
  --alignment-temperature 0.2 \
  --temporal-delta 2 \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T2_time_text_align_seed${SEED}.json"

run_step "text_ablation_T3_source_text_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${text_args[@]}" \
  --source-text-align-weight 0.0005 \
  --alignment-temperature 0.2 \
  --temporal-delta 2 \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T3_source_text_align_seed${SEED}.json"

run_step "text_ablation_T4_destination_text_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${text_args[@]}" \
  --destination-text-align-weight 0.0005 \
  --alignment-temperature 0.2 \
  --temporal-delta 2 \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T4_destination_text_align_seed${SEED}.json"

run_step "text_ablation_T5_all_text_align_small_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${text_args[@]}" \
  --source-text-align-weight 0.0005 \
  --destination-text-align-weight 0.0005 \
  --time-text-align-weight 0.001 \
  --alignment-temperature 0.2 \
  --temporal-delta 2 \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T5_all_text_align_small_seed${SEED}.json"

run_step "text_ablation_T6_all_text_align_strong_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${text_args[@]}" \
  --source-text-align-weight 0.001 \
  --destination-text-align-weight 0.001 \
  --time-text-align-weight 0.003 \
  --alignment-temperature 0.2 \
  --temporal-delta 2 \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T6_all_text_align_strong_seed${SEED}.json"

run_step "text_ablation_T7_struct_A5_plus_time_text_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  --struct-feature-group full \
  --time-align-weight 0.001 \
  "${text_args[@]}" \
  --time-text-align-weight 0.001 \
  --alignment-temperature 0.2 \
  --temporal-delta 2 \
  "${base_args[@]}" \
  --metrics-path "results/text_ablation_T7_struct_A5_plus_time_text_seed${SEED}.json"

echo "Ablation finished at $(date)"
echo "Result files:"
ls -1 results/text_ablation_T*seed"${SEED}".json 2>/dev/null || true
