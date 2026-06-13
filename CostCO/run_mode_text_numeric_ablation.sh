#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-sat_path_bytes_mb_tensor.npy}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-sat_connectivity_tensor_dynamic_60s_1000ms.npz}"
BASE_TEXT_DIR="${BASE_TEXT_DIR:-mode_text_data}"
WORK_TEXT_ROOT="${WORK_TEXT_ROOT:-mode_text_numeric_ablation_data}"
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
REBUILD_TEXT="${REBUILD_TEXT:-1}"

SOURCE_TEXT_ALIGN_WEIGHT="${SOURCE_TEXT_ALIGN_WEIGHT:-0.0005}"
DESTINATION_TEXT_ALIGN_WEIGHT="${DESTINATION_TEXT_ALIGN_WEIGHT:-0.0005}"
TIME_TEXT_ALIGN_WEIGHT="${TIME_TEXT_ALIGN_WEIGHT:-0.001}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"

mkdir -p logs results "$BASE_TEXT_DIR" "$WORK_TEXT_ROOT"
export BASE_TEXT_DIR WORK_TEXT_ROOT

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/mode_text_numeric_ablation_${timestamp}.log"
pid_file="logs/mode_text_numeric_ablation.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started mode text numeric ablation in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
  exit 0
fi

echo "===== Mode Text Numeric Side-Channel Ablation ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Seed: $SEED"
echo "Base text dir: $BASE_TEXT_DIR"
echo "Work text root: $WORK_TEXT_ROOT"
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

ensure_base_text_features() {
  if [[ "$REBUILD_TEXT" != "1" \
        && -f "$BASE_TEXT_DIR/source_text_embeddings.npy" \
        && -f "$BASE_TEXT_DIR/destination_text_embeddings.npy" \
        && -f "$BASE_TEXT_DIR/time_text_embeddings.npy" \
        && -f "$BASE_TEXT_DIR/source_text_numeric_features.npy" \
        && -f "$BASE_TEXT_DIR/destination_text_numeric_features.npy" \
        && -f "$BASE_TEXT_DIR/time_text_numeric_features.npy" ]]; then
    echo "Reusing existing text embeddings and numeric features in $BASE_TEXT_DIR"
    return
  fi

  if [[ ! -d "$MODEL_PATH" && "$ALLOW_REMOTE_MODEL" != "1" ]]; then
    echo "Missing local text encoder directory: $MODEL_PATH" >&2
    echo "Set MODEL_PATH=/path/to/all-MiniLM-L6-v2 or ALLOW_REMOTE_MODEL=1." >&2
    exit 1
  fi

  run_step "numeric_ablation_build_records_seed${SEED}" \
    "$PYTHON_BIN" build_mode_texts.py \
    --tensor-path "$TENSOR_PATH" \
    --topology-path "$TOPOLOGY_PATH" \
    --output-dir "$BASE_TEXT_DIR" \
    --history-len "$HISTORY_LEN" \
    --target-start "$TARGET_START"

  encode_cmd=(
    "$PYTHON_BIN" encode_mode_texts.py
    --text-dir "$BASE_TEXT_DIR"
    --model-path "$MODEL_PATH"
    --batch-size "$TEXT_BATCH_SIZE"
  )
  if [[ "$ALLOW_REMOTE_MODEL" == "1" ]]; then
    encode_cmd+=(--allow-remote-model)
  fi
  run_step "numeric_ablation_encode_seed${SEED}" "${encode_cmd[@]}"
}

prepare_variant_dirs() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
import numpy as np

base = os.environ.get("BASE_TEXT_DIR", "mode_text_data")
root = os.environ.get("WORK_TEXT_ROOT", "mode_text_numeric_ablation_data")
variants = {
    "embedding_only": os.path.join(root, "embedding_only"),
    "numeric_only": os.path.join(root, "numeric_only"),
    "both": os.path.join(root, "both"),
}

required = [
    "source_text_embeddings.npy",
    "destination_text_embeddings.npy",
    "time_text_embeddings.npy",
    "source_text_numeric_features.npy",
    "destination_text_numeric_features.npy",
    "time_text_numeric_features.npy",
    "text_embedding_metadata.json",
]
missing = [name for name in required if not os.path.exists(os.path.join(base, name))]
if missing:
    raise SystemExit("Missing base text feature files: %s" % missing)

for path in variants.values():
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            os.remove(full)

copy_names = [
    "source_text_records.json",
    "destination_text_records.json",
    "time_text_records.json",
    "text_embedding_metadata.json",
]
for variant, path in variants.items():
    for name in copy_names:
        src = os.path.join(base, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(path, name))

for name in [
    "source_text_embeddings.npy",
    "destination_text_embeddings.npy",
    "time_text_embeddings.npy",
]:
    arr = np.load(os.path.join(base, name)).astype("float32")
    np.save(os.path.join(variants["embedding_only"], name), arr)
    np.save(os.path.join(variants["both"], name), arr)
    np.save(os.path.join(variants["numeric_only"], name), np.zeros_like(arr))

for name in [
    "source_text_numeric_features.npy",
    "destination_text_numeric_features.npy",
    "time_text_numeric_features.npy",
]:
    arr = np.load(os.path.join(base, name)).astype("float32")
    np.save(os.path.join(variants["numeric_only"], name), arr)
    np.save(os.path.join(variants["both"], name), arr)

for variant, path in variants.items():
    metadata_path = os.path.join(path, "text_embedding_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["numeric_ablation_variant"] = variant
    metadata["numeric_side_channel"] = variant in ("numeric_only", "both")
    metadata["text_embedding_zeroed"] = variant == "numeric_only"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Prepared variant text dirs:")
for variant, path in variants.items():
    print("  %s: %s" % (variant, path))
PY
}

base_args=(
  --rank "$RANK" --nc "$NC"
  --node-dim "$NODE_DIM" --gcn-dim "$GCN_DIM"
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --target-normalization max --seed "$SEED"
)

align_args=(
  --source-text-align-weight "$SOURCE_TEXT_ALIGN_WEIGHT"
  --destination-text-align-weight "$DESTINATION_TEXT_ALIGN_WEIGHT"
  --time-text-align-weight "$TIME_TEXT_ALIGN_WEIGHT"
  --alignment-temperature "$ALIGNMENT_TEMPERATURE"
  --temporal-delta "$TEMPORAL_DELTA"
)

ensure_base_text_features
prepare_variant_dirs

run_step "text_numeric_B0_gcn_baseline_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  --struct-feature-group none \
  "${base_args[@]}" \
  --metrics-path "results/text_numeric_B0_gcn_baseline_seed${SEED}.json"

run_step "text_numeric_B1_embedding_only_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  --struct-feature-group none \
  --use-mode-text \
  --mode-text-dir "$WORK_TEXT_ROOT/embedding_only" \
  "${align_args[@]}" \
  "${base_args[@]}" \
  --metrics-path "results/text_numeric_B1_embedding_only_align_seed${SEED}.json"

run_step "text_numeric_B2_numeric_only_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  --struct-feature-group none \
  --use-mode-text \
  --mode-text-dir "$WORK_TEXT_ROOT/numeric_only" \
  "${align_args[@]}" \
  "${base_args[@]}" \
  --metrics-path "results/text_numeric_B2_numeric_only_align_seed${SEED}.json"

run_step "text_numeric_B3_both_align_seed${SEED}" \
  "$PYTHON_BIN" run_gcn_sat_tensor_experiment.py \
  --struct-feature-group none \
  --use-mode-text \
  --mode-text-dir "$WORK_TEXT_ROOT/both" \
  "${align_args[@]}" \
  "${base_args[@]}" \
  --metrics-path "results/text_numeric_B3_both_align_seed${SEED}.json"

echo "Numeric side-channel ablation finished at $(date)"
echo "Result files:"
ls -1 results/text_numeric_B*seed"${SEED}".json 2>/dev/null || true
