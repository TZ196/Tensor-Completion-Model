#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-TZ-costco}"
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
USER_BATCH_SIZE="${BATCH_SIZE:-}"
BATCH_SIZE="${USER_BATCH_SIZE:-256}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
TEXT_ALIGN_TARGET_RATIO="${TEXT_ALIGN_TARGET_RATIO:-0.01}"
ALIGNMENT_TEMPERATURE="${ALIGNMENT_TEMPERATURE:-0.2}"
TEMPORAL_DELTA="${TEMPORAL_DELTA:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
MAX_GRAPH_ATTENTION_BIAS="${MAX_GRAPH_ATTENTION_BIAS:-2.0}"
FAST_MODE="${FAST_MODE:-0}"
USER_TEXT_ALIGN_SAMPLE_SIZE="${TEXT_ALIGN_SAMPLE_SIZE:-}"
USER_DISABLE_TEXT_METRICS="${DISABLE_TEXT_METRICS:-}"
TEXT_ALIGN_SAMPLE_SIZE="${USER_TEXT_ALIGN_SAMPLE_SIZE:-0}"
DISABLE_TEXT_METRICS="${USER_DISABLE_TEXT_METRICS:-0}"

if [[ "$FAST_MODE" == "1" ]]; then
  if [[ -z "$USER_BATCH_SIZE" ]]; then
    BATCH_SIZE=1024
  fi
  if [[ -z "$USER_TEXT_ALIGN_SAMPLE_SIZE" ]]; then
    TEXT_ALIGN_SAMPLE_SIZE=128
  fi
  if [[ -z "$USER_DISABLE_TEXT_METRICS" ]]; then
    DISABLE_TEXT_METRICS=1
  fi
fi

VISIBLE_RATES_RAW="${VISIBLE_RATES:-7 10 20}"
VARIANTS_RAW="${VARIANTS:-M2 M3 M4 M5 M6}"
VISIBLE_RATES=($VISIBLE_RATES_RAW)
VARIANTS=($VARIANTS_RAW)

mkdir -p logs results splits checkpoints histories

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/gt_mst_graph_bias_v3_${timestamp}.log"
pid_file="logs/gt_mst_graph_bias_v3.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup env \
    PYTHON_BIN="$PYTHON_BIN" \
    CONDA_ENV="$CONDA_ENV" \
    TENSOR_PATH="$TENSOR_PATH" \
    TOPOLOGY_PATH="$TOPOLOGY_PATH" \
    MODE_TEXT_DIR="$MODE_TEXT_DIR" \
    SEED="$SEED" \
    D_MODEL="$D_MODEL" \
    NODE_DIM="$NODE_DIM" \
    GCN_DIM="$GCN_DIM" \
    TRANSFORMER_LAYERS="$TRANSFORMER_LAYERS" \
    NUM_HEADS="$NUM_HEADS" \
    FF_DIM="$FF_DIM" \
    DROPOUT="$DROPOUT" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    TARGET_NORMALIZATION="$TARGET_NORMALIZATION" \
    TEXT_ALIGN_TARGET_RATIO="$TEXT_ALIGN_TARGET_RATIO" \
    ALIGNMENT_TEMPERATURE="$ALIGNMENT_TEMPERATURE" \
    TEMPORAL_DELTA="$TEMPORAL_DELTA" \
    EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
    MAX_GRAPH_ATTENTION_BIAS="$MAX_GRAPH_ATTENTION_BIAS" \
    FAST_MODE="$FAST_MODE" \
    TEXT_ALIGN_SAMPLE_SIZE="$TEXT_ALIGN_SAMPLE_SIZE" \
    DISABLE_TEXT_METRICS="$DISABLE_TEXT_METRICS" \
    VISIBLE_RATES="$VISIBLE_RATES_RAW" \
    VARIANTS="$VARIANTS_RAW" \
    bash "$SCRIPT_PATH" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started GT-MST V3 graph-bias validation in background."
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

activate_conda_env() {
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    echo "Skipping conda activation because SKIP_CONDA_ACTIVATE=1"
    return
  fi

  if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
  elif command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.conda/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh. Set SKIP_CONDA_ACTIVATE=1 if the environment is already active." >&2
    exit 1
  fi

  conda activate "$CONDA_ENV"
  CONDA_ENV_ACTIVATED=1
  echo "Activated conda environment: $CONDA_ENV"
}

deactivate_conda_env() {
  if [[ "${CONDA_ENV_ACTIVATED:-0}" == "1" ]]; then
    conda deactivate || true
    echo "Deactivated conda environment: $CONDA_ENV"
  fi
}

trap deactivate_conda_env EXIT
activate_conda_env

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

echo "===== GT-MST V3 Graph-Bias Validation ====="
echo "Start time: $(date)"
echo "Work dir: $PROJECT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Topology: $TOPOLOGY_PATH"
echo "Mode text dir: $MODE_TEXT_DIR"
echo "Conda env: $CONDA_ENV"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "Fast mode: $FAST_MODE"
echo "Max graph attention bias: $MAX_GRAPH_ATTENTION_BIAS"
echo "Text align sample size: $TEXT_ALIGN_SAMPLE_SIZE"
echo "Disable auxiliary text metrics: $DISABLE_TEXT_METRICS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
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
  --max-graph-attention-bias "$MAX_GRAPH_ATTENTION_BIAS"
  --text-align-sample-size "$TEXT_ALIGN_SAMPLE_SIZE"
)

if [[ "$DISABLE_TEXT_METRICS" == "1" ]]; then
  common_args+=(--disable-text-metrics)
fi

for rate in "${VISIBLE_RATES[@]}"; do
  ratio="$(rate_ratio "$rate")"
  split_path="splits/random_observed${rate}_val10_seed_${SEED}.npz"
  for variant in "${VARIANTS[@]}"; do
    name="vis${rate}_${variant}_gt_mst_graph_bias_v3_seed${SEED}"
    run_step "$name" \
      "$PYTHON_BIN" run_experiment.py \
      "${common_args[@]}" \
      --variant "$variant" \
      --observed-ratio "$ratio" \
      --split-path "$split_path" \
      --metrics-path "results/${name}.json"
  done
done

echo "===== GT-MST V3 graph-bias validation finished at $(date) ====="
"$PYTHON_BIN" - <<'PY'
import json
import os

rates = [int(v) for v in os.environ.get("VISIBLE_RATES", "7 10 20").split()]
variants = os.environ.get("VARIANTS", "M2 M3 M4 M5 M6").split()
seed = os.environ.get("SEED", "3")

print(f"{'Visible':>7s}  {'Variant':>7s}  {'Model':34s}  {'NMAE':>10s}  {'NRMSE':>10s}")
print("-" * 78)
names = {
    "M2": "GraphToken+GraphBias",
    "M3": "M2+TextOnly",
    "M4": "M2+NumericOnly",
    "M5": "M2+TextNumeric",
    "M6": "M5+TextAlign",
}
for rate in rates:
    for variant in variants:
        path = f"results/vis{rate}_{variant}_gt_mst_graph_bias_v3_seed{seed}.json"
        label = names.get(variant, variant)
        if not os.path.exists(path):
            print(f"{rate:>6d}%  {variant:>7s}  {label:34s}  {'missing':>10s}  {'missing':>10s}")
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        test = payload["test"]
        print(f"{rate:>6d}%  {variant:>7s}  {label:34s}  {test['nmae']:10.6f}  {test['nrmse']:10.6f}")
    print("-" * 78)
PY
