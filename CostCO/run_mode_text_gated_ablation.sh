#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TEXT_DIR="${TEXT_DIR:-mode_text_numeric_ablation_data/both}"
SEED="${SEED:-3}"
RANK="${RANK:-50}"
NC="${NC:-64}"
NODE_DIM="${NODE_DIM:-64}"
GCN_DIM="${GCN_DIM:-128}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUMERIC_ALPHA_INIT="${NUMERIC_ALPHA_INIT:-0.02}"
SOURCE_TEXT_ALIGN_WEIGHT="${SOURCE_TEXT_ALIGN_WEIGHT:-0.0005}"
DESTINATION_TEXT_ALIGN_WEIGHT="${DESTINATION_TEXT_ALIGN_WEIGHT:-0.0001}"
TIME_TEXT_ALIGN_WEIGHT="${TIME_TEXT_ALIGN_WEIGHT:-0.0008}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"

mkdir -p logs results

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/mode_text_gated_ablation_${timestamp}.log"
pid_file="logs/mode_text_gated_ablation.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started mode text gated ablation in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
  exit 0
fi

echo "===== Mode Text Gated Numeric Ablation ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Text dir: $TEXT_DIR"
echo "Seed: $SEED"
echo "Numeric alpha init: $NUMERIC_ALPHA_INIT"
echo

required=(
  source_text_embeddings.npy
  destination_text_embeddings.npy
  time_text_embeddings.npy
  source_text_numeric_features.npy
  destination_text_numeric_features.npy
  time_text_numeric_features.npy
)
for name in "${required[@]}"; do
  if [[ ! -f "$TEXT_DIR/$name" ]]; then
    echo "Missing $TEXT_DIR/$name" >&2
    echo "Run run_mode_text_numeric_ablation.sh once to prepare text/numeric features." >&2
    exit 1
  fi
done

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

base_args=(
  --struct-feature-group none
  --rank "$RANK" --nc "$NC"
  --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM"
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --target-normalization max --seed "$SEED"
)

text_args=(
  --use-mode-text
  --mode-text-dir "$TEXT_DIR"
)

align_args=(
  --source-text-align-weight "$SOURCE_TEXT_ALIGN_WEIGHT"
  --destination-text-align-weight "$DESTINATION_TEXT_ALIGN_WEIGHT"
  --time-text-align-weight "$TIME_TEXT_ALIGN_WEIGHT"
  --alignment-temperature "$ALIGNMENT_TEMPERATURE"
  --temporal-delta "$TEMPORAL_DELTA"
)

run_step "text_gated_G0_gcn_baseline_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${base_args[@]}" \
  --metrics-path "results/text_gated_G0_gcn_baseline_seed${SEED}.json"

run_step "text_gated_G1_concat_conservative_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${base_args[@]}" \
  "${text_args[@]}" \
  --text-fusion-mode concat \
  "${align_args[@]}" \
  --metrics-path "results/text_gated_G1_concat_conservative_align_seed${SEED}.json"

run_step "text_gated_G2_gated_conservative_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${base_args[@]}" \
  "${text_args[@]}" \
  --text-fusion-mode gated_numeric \
  --numeric-alpha-init "$NUMERIC_ALPHA_INIT" \
  "${align_args[@]}" \
  --metrics-path "results/text_gated_G2_gated_conservative_align_seed${SEED}.json"

echo "Gated numeric ablation finished at $(date)"
echo "Result files:"
ls -1 results/text_gated_G*seed"${SEED}".json 2>/dev/null || true
