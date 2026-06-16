#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
TEXT_DIR="${TEXT_DIR:-mode_text_refined_data}"
WORK_TEXT_ROOT="${WORK_TEXT_ROOT:-mode_text_residual_verification_data}"
VISIBLE_RATE="${VISIBLE_RATE:-7}"
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
TEXT_RESIDUAL_ALPHA_INIT="${TEXT_RESIDUAL_ALPHA_INIT:-0.005}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"

mkdir -p logs results splits "$WORK_TEXT_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/text_residual_verification_${timestamp}.log"
pid_file="logs/text_residual_verification.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started text residual verification in background."
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

rate_tag="vis${VISIBLE_RATE}"
alpha_tag="$(printf "%s" "$TEXT_RESIDUAL_ALPHA_INIT" | sed 's/\./p/g')"
observed_ratio="$(awk "BEGIN { printf \"%.2f\", ${VISIBLE_RATE}/100 }")"
split_path="splits/random_observed${VISIBLE_RATE}_val10_seed_${SEED}.npz"
text_only_dir="$WORK_TEXT_ROOT/text_only"
numeric_only_dir="$WORK_TEXT_ROOT/numeric_only"

echo "===== Text/Numeric Residual Verification ====="
echo "Start time: $(date)"
echo "Visible rate: ${VISIBLE_RATE}%"
echo "Observed ratio: $observed_ratio"
echo "Seed: $SEED"
echo "Text dir: $TEXT_DIR"
echo "Residual alpha init: $TEXT_RESIDUAL_ALPHA_INIT"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
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

"$PYTHON_BIN" - <<PY
import json
import os
import shutil
import numpy as np

base = "$TEXT_DIR"
text_only = "$text_only_dir"
numeric_only = "$numeric_only_dir"
for path in (text_only, numeric_only):
    os.makedirs(path, exist_ok=True)

embedding_names = [
    "source_text_embeddings.npy",
    "destination_text_embeddings.npy",
    "time_text_embeddings.npy",
]
numeric_names = [
    "source_text_numeric_features.npy",
    "destination_text_numeric_features.npy",
    "time_text_numeric_features.npy",
]
record_names = [
    "source_text_records.json",
    "destination_text_records.json",
    "time_text_records.json",
    "text_embedding_metadata.json",
]

for name in embedding_names:
    arr = np.load(os.path.join(base, name)).astype("float32")
    np.save(os.path.join(text_only, name), arr)
    np.save(os.path.join(numeric_only, name), np.zeros_like(arr))

for name in numeric_names:
    numeric = np.load(os.path.join(base, name)).astype("float32")
    np.save(os.path.join(numeric_only, name), numeric)
    target = os.path.join(text_only, name)
    if os.path.exists(target):
        os.remove(target)

for name in record_names:
    src = os.path.join(base, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(text_only, name))
        shutil.copy2(src, os.path.join(numeric_only, name))

for variant, path in (("text_only", text_only), ("numeric_only", numeric_only)):
    meta_path = os.path.join(path, "text_embedding_metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata["residual_verification_variant"] = variant
    metadata["text_embedding_zeroed"] = variant == "numeric_only"
    metadata["numeric_features_present"] = variant == "numeric_only"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Prepared:", text_only)
print("Prepared:", numeric_only)
PY

"$PYTHON_BIN" -c "from run_sat_tensor_experiment import create_random_completion_split; create_random_completion_split('${TENSOR_PATH}', '${split_path}', ${observed_ratio}, ${VAL_RATIO}, ${SEED}); print('${split_path}')"

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

common_gcn_args=(
  --tensor-path "$TENSOR_PATH"
  --topology-path "$TOPOLOGY_PATH"
  --observed-ratio "$observed_ratio"
  --split-path "$split_path"
  --rank "$RANK" --nc "$NC"
  --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM"
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --target-normalization "$TARGET_NORMALIZATION"
  --seed "$SEED"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
)

run_step "${rate_tag}_R0_gcn_costco_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${common_gcn_args[@]}" \
  --metrics-path "results/${rate_tag}_R0_gcn_costco_seed${SEED}.json"

run_step "${rate_tag}_R1_text_residual_alpha${alpha_tag}_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${common_gcn_args[@]}" \
  --use-mode-text \
  --mode-text-dir "$text_only_dir" \
  --text-application residual \
  --text-fusion-mode concat \
  --text-residual-alpha-init "$TEXT_RESIDUAL_ALPHA_INIT" \
  --metrics-path "results/${rate_tag}_R1_text_residual_alpha${alpha_tag}_seed${SEED}.json"

run_step "${rate_tag}_R2_numeric_gate_residual_alpha${alpha_tag}_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${common_gcn_args[@]}" \
  --use-mode-text \
  --mode-text-dir "$numeric_only_dir" \
  --text-application residual \
  --text-fusion-mode gated_numeric \
  --text-residual-alpha-init "$TEXT_RESIDUAL_ALPHA_INIT" \
  --metrics-path "results/${rate_tag}_R2_numeric_gate_residual_alpha${alpha_tag}_seed${SEED}.json"

run_step "${rate_tag}_R3_text_numeric_residual_alpha${alpha_tag}_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  "${common_gcn_args[@]}" \
  --use-mode-text \
  --mode-text-dir "$TEXT_DIR" \
  --text-application residual \
  --text-fusion-mode gated_numeric \
  --text-residual-alpha-init "$TEXT_RESIDUAL_ALPHA_INIT" \
  --metrics-path "results/${rate_tag}_R3_text_numeric_residual_alpha${alpha_tag}_seed${SEED}.json"

echo "===== Residual verification finished at $(date) ====="
"$PYTHON_BIN" - <<PY
import json
import os

rate_tag = "$rate_tag"
seed = "$SEED"
alpha_tag = "$alpha_tag"
rows = [
    ("R0", "GCN-CoSTCo", f"results/{rate_tag}_R0_gcn_costco_seed{seed}.json"),
    ("R1", "TextResidual", f"results/{rate_tag}_R1_text_residual_alpha{alpha_tag}_seed{seed}.json"),
    ("R2", "NumericGateResidual", f"results/{rate_tag}_R2_numeric_gate_residual_alpha{alpha_tag}_seed{seed}.json"),
    ("R3", "TextNumericResidual", f"results/{rate_tag}_R3_text_numeric_residual_alpha{alpha_tag}_seed{seed}.json"),
]
print(f"{'Exp':4s}  {'Model':24s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 58)
for exp, label, path in rows:
    if not os.path.exists(path):
        print(f"{exp:4s}  {label:24s}  {'missing':>10s}  {'missing':>10s}")
        continue
    with open(path) as f:
        payload = json.load(f)
    test = payload.get("test", payload)
    print(f"{exp:4s}  {label:24s}  {test['nmae']:10.6f}  {test['nrmse']:10.6f}")
PY
