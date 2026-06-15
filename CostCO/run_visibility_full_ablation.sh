#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
TEXT_DIR="${TEXT_DIR:-mode_text_refined_data}"
TEXT_ONLY_DIR="${TEXT_ONLY_DIR:-${TEXT_DIR}_text_only}"
OD_PATH_FEATURE_PATH="${OD_PATH_FEATURE_PATH:-mode_od_path_features.npz}"
SEED="${SEED:-3}"
VAL_RATIO="${VAL_RATIO:-0.1}"
RANK="${RANK:-50}"
NC="${NC:-64}"
NODE_DIM="${NODE_DIM:-64}"
GCN_DIM="${GCN_DIM:-128}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
TEXT_FUSION_MODE="${TEXT_FUSION_MODE:-gated_numeric}"
TEXT_ALIGN_TARGET_RATIO="${TEXT_ALIGN_TARGET_RATIO:-0}"
OD_PATH_ALPHA_INIT="${OD_PATH_ALPHA_INIT:-0.05}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"
PLANES="${PLANES:-10}"
NODE_COUNT="${NODE_COUNT:-120}"
TIME_LEN="${TIME_LEN:-60}"

VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})

mkdir -p logs results splits checkpoints histories

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/visibility_full_ablation_${timestamp}.log"
pid_file="logs/visibility_full_ablation.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started full visibility ablation in background."
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

echo "===== Visibility Full Ablation ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Topology: $TOPOLOGY_PATH"
echo "Text dir: $TEXT_DIR"
echo "Text-only dir: $TEXT_ONLY_DIR"
echo "OD path feature path: $OD_PATH_FEATURE_PATH"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$TENSOR_PATH"
require_file "$TOPOLOGY_PATH"

if [[ ! -f "$OD_PATH_FEATURE_PATH" ]]; then
  echo "OD path feature file not found; building $OD_PATH_FEATURE_PATH"
  "$PYTHON_BIN" build_od_path_features.py \
    --topology-path "$TOPOLOGY_PATH" \
    --output-path "$OD_PATH_FEATURE_PATH" \
    --planes "$PLANES" \
    --node-count "$NODE_COUNT" \
    --time-len "$TIME_LEN"
fi

for name in \
  source_text_embeddings.npy \
  destination_text_embeddings.npy \
  time_text_embeddings.npy \
  source_text_numeric_features.npy \
  destination_text_numeric_features.npy \
  time_text_numeric_features.npy
do
  require_file "$TEXT_DIR/$name"
done

prepare_text_only_dir() {
  mkdir -p "$TEXT_ONLY_DIR"
  cp "$TEXT_DIR/source_text_embeddings.npy" "$TEXT_ONLY_DIR/source_text_embeddings.npy"
  cp "$TEXT_DIR/destination_text_embeddings.npy" "$TEXT_ONLY_DIR/destination_text_embeddings.npy"
  cp "$TEXT_DIR/time_text_embeddings.npy" "$TEXT_ONLY_DIR/time_text_embeddings.npy"
  if [[ -f "$TEXT_DIR/text_embedding_metadata.json" ]]; then
    cp "$TEXT_DIR/text_embedding_metadata.json" "$TEXT_ONLY_DIR/text_embedding_metadata.json"
  fi
  rm -f "$TEXT_ONLY_DIR/source_text_numeric_features.npy"
  rm -f "$TEXT_ONLY_DIR/destination_text_numeric_features.npy"
  rm -f "$TEXT_ONLY_DIR/time_text_numeric_features.npy"
}

prepare_text_only_dir

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

prepare_split() {
  local rate="$1"
  local ratio="$2"
  local split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  echo "Preparing split ${split_path}"
  "$PYTHON_BIN" -c "from run_sat_tensor_experiment import create_random_completion_split; create_random_completion_split('${TENSOR_PATH}', '${split_path}', ${ratio}, ${VAL_RATIO}, ${SEED}); print('${split_path}')"
}

common_costco_args=(
  --tensor-path "$TENSOR_PATH"
  --rank "$RANK" --nc "$NC"
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --target-normalization "$TARGET_NORMALIZATION"
  --seed "$SEED"
)

common_gcn_args=(
  --tensor-path "$TENSOR_PATH"
  --topology-path "$TOPOLOGY_PATH"
  --rank "$RANK" --nc "$NC"
  --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM"
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --target-normalization "$TARGET_NORMALIZATION"
  --seed "$SEED"
  --alignment-temperature "$ALIGNMENT_TEMPERATURE"
  --temporal-delta "$TEMPORAL_DELTA"
)

for rate in "${VISIBLE_RATES[@]}"; do
  ratio="$(awk "BEGIN { printf \"%.2f\", ${rate}/100 }")"
  split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  prepare_split "$rate" "$ratio"

  run_step "vis${rate}_E0_costco_seed${SEED}" \
    "$PYTHON_BIN" run_sat_tensor_experiment.py \
    "${common_costco_args[@]}" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --metrics-path "results/vis${rate}_E0_costco_seed${SEED}.json"

  run_step "vis${rate}_E1_gcn_costco_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    "${common_gcn_args[@]}" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --metrics-path "results/vis${rate}_E1_gcn_costco_seed${SEED}.json"

  run_step "vis${rate}_E2_text_only_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    "${common_gcn_args[@]}" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --use-mode-text \
    --mode-text-dir "$TEXT_ONLY_DIR" \
    --text-fusion-mode concat \
    --text-align-target-ratio "$TEXT_ALIGN_TARGET_RATIO" \
    --metrics-path "results/vis${rate}_E2_text_only_seed${SEED}.json"

  run_step "vis${rate}_E3_text_numeric_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    "${common_gcn_args[@]}" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --use-mode-text \
    --mode-text-dir "$TEXT_DIR" \
    --text-fusion-mode "$TEXT_FUSION_MODE" \
    --text-align-target-ratio "$TEXT_ALIGN_TARGET_RATIO" \
    --metrics-path "results/vis${rate}_E3_text_numeric_seed${SEED}.json"

  run_step "vis${rate}_E4_od_path_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    "${common_gcn_args[@]}" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --use-od-path-features \
    --od-path-feature-path "$OD_PATH_FEATURE_PATH" \
    --od-path-alpha-init "$OD_PATH_ALPHA_INIT" \
    --metrics-path "results/vis${rate}_E4_od_path_seed${SEED}.json"

  run_step "vis${rate}_E5_text_numeric_od_path_seed${SEED}" \
    "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
    "${common_gcn_args[@]}" \
    --observed-ratio "$ratio" \
    --split-path "$split_path" \
    --use-mode-text \
    --mode-text-dir "$TEXT_DIR" \
    --text-fusion-mode "$TEXT_FUSION_MODE" \
    --text-align-target-ratio "$TEXT_ALIGN_TARGET_RATIO" \
    --use-od-path-features \
    --od-path-feature-path "$OD_PATH_FEATURE_PATH" \
    --od-path-alpha-init "$OD_PATH_ALPHA_INIT" \
    --metrics-path "results/vis${rate}_E5_text_numeric_od_path_seed${SEED}.json"
done

echo "===== Full ablation finished at $(date) ====="
"$PYTHON_BIN" - <<'PY'
import json
import os

rates = [1, 3, 5, 7, 10, 20]
models = [
    ("E0", "CoSTCo", "costco"),
    ("E1", "GCN-CoSTCo", "gcn_costco"),
    ("E2", "GCN+TextOnly", "text_only"),
    ("E3", "GCN+TextNumeric", "text_numeric"),
    ("E4", "GCN+ODPath", "od_path"),
    ("E5", "GCN+TextNumeric+ODPath", "text_numeric_od_path"),
]
seed = os.environ.get("SEED", "3")
print(f"{'Visible':>7s}  {'Exp':4s}  {'Model':26s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 68)
for rate in rates:
    for exp, label, stem in models:
        path = f"results/vis{rate}_{exp}_{stem}_seed{seed}.json"
        if not os.path.exists(path):
            print(f"{rate:>6d}%  {exp:4s}  {label:26s}  {'missing':>10s}  {'missing':>10s}")
            continue
        with open(path) as f:
            payload = json.load(f)
        test = payload.get("test", payload)
        print(f"{rate:>6d}%  {exp:4s}  {label:26s}  {test['nmae']:10.6f}  {test['nrmse']:10.6f}")
    print("-" * 68)
PY
