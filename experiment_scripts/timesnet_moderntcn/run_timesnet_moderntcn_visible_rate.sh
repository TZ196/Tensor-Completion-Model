#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-TZ-Satformer}"
TENSOR_PATH="${TENSOR_PATH:-$REPO_DIR/data/iridium5400s/iridium.npy}"
DATASET_NAME="${DATASET_NAME:-iridium5400s}"

TIMESNET_DIR="${TIMESNET_DIR:-$REPO_DIR/TimesNet}"
if [[ -n "${MODERNTCN_DIR:-}" ]]; then
  TCN_DIR="$MODERNTCN_DIR"
elif [[ -d "$REPO_DIR/ModelnTCN" ]]; then
  TCN_DIR="$REPO_DIR/ModelnTCN"
else
  TCN_DIR="$REPO_DIR/ModernTCN-imputation"
fi

SEED="${SEED:-3}"
VAL_RATIO="${VAL_RATIO:-0.1}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"

# TimesNet hyperparameters.
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
TOP_K="${TOP_K:-2}"
NUM_KERNELS="${NUM_KERNELS:-6}"
TIMESNET_DROPOUT="${TIMESNET_DROPOUT:-0.1}"

# ModernTCN hyperparameters.
FFN_RATIO="${FFN_RATIO:-1}"
PATCH_SIZE="${PATCH_SIZE:-1}"
PATCH_STRIDE="${PATCH_STRIDE:-1}"
NUM_BLOCKS="${NUM_BLOCKS:-1}"
LARGE_SIZE="${LARGE_SIZE:-31}"
SMALL_SIZE="${SMALL_SIZE:-5}"
DIMS="${DIMS:-64}"
DW_DIMS="${DW_DIMS:-64}"
TCN_DROPOUT="${TCN_DROPOUT:-0.1}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"

# These are observed/visible rates in percent.
VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})
MODELS=(${MODELS:-TimesNet ModernTCN})

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/results" "$SCRIPT_DIR/splits"

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="$SCRIPT_DIR/logs/${DATASET_NAME}_timesnet_moderntcn_visible_rate_${timestamp}.log"
pid_file="$SCRIPT_DIR/logs/${DATASET_NAME}_timesnet_moderntcn_visible_rate.pid"
summary_csv="$SCRIPT_DIR/results/${DATASET_NAME}_timesnet_moderntcn_visible_rate_nrmse_seed${SEED}.csv"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup env \
    PYTHON_BIN="$PYTHON_BIN" \
    CONDA_ENV="$CONDA_ENV" \
    SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}" \
    TENSOR_PATH="$TENSOR_PATH" \
    DATASET_NAME="$DATASET_NAME" \
    TIMESNET_DIR="$TIMESNET_DIR" \
    MODERNTCN_DIR="$TCN_DIR" \
    SEED="$SEED" \
    VAL_RATIO="$VAL_RATIO" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    PATIENCE="$PATIENCE" \
    TARGET_NORMALIZATION="$TARGET_NORMALIZATION" \
    D_MODEL="$D_MODEL" \
    D_FF="$D_FF" \
    E_LAYERS="$E_LAYERS" \
    TOP_K="$TOP_K" \
    NUM_KERNELS="$NUM_KERNELS" \
    TIMESNET_DROPOUT="$TIMESNET_DROPOUT" \
    FFN_RATIO="$FFN_RATIO" \
    PATCH_SIZE="$PATCH_SIZE" \
    PATCH_STRIDE="$PATCH_STRIDE" \
    NUM_BLOCKS="$NUM_BLOCKS" \
    LARGE_SIZE="$LARGE_SIZE" \
    SMALL_SIZE="$SMALL_SIZE" \
    DIMS="$DIMS" \
    DW_DIMS="$DW_DIMS" \
    TCN_DROPOUT="$TCN_DROPOUT" \
    HEAD_DROPOUT="$HEAD_DROPOUT" \
    VISIBLE_RATES="${VISIBLE_RATES[*]}" \
    MODELS="${MODELS[*]}" \
    bash "$SCRIPT_PATH" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started TimesNet/ModernTCN visible-rate run in background."
  echo "PID: $pid"
  echo "Master log: $master_log"
  echo "PID file: $pid_file"
  echo "Watch: tail -f $master_log"
  exit 0
fi

export PYTHONHASHSEED="$SEED"

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

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
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
  local log_file="$SCRIPT_DIR/logs/${name}.log"
  echo "----- Running ${name} -----"
  echo "Command: $*"
  echo "Log: ${log_file}"
  "$@" > "$log_file" 2>&1
  echo "Finished ${name} at $(date)"
  echo
}

run_timesnet() {
  local rate="$1"
  local ratio="$2"
  local split_path="$SCRIPT_DIR/splits/${DATASET_NAME}_visible${rate}_val10_seed_${SEED}.npz"
  local metrics_path="$SCRIPT_DIR/results/${DATASET_NAME}_timesnet_visible${rate}_seed${SEED}.json"
  local name="${DATASET_NAME}_timesnet_visible${rate}_seed${SEED}"
  (
    cd "$TIMESNET_DIR"
    run_step "$name" \
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
      --dropout "$TIMESNET_DROPOUT" \
      --metrics-path "$metrics_path"
  )
}

run_moderntcn() {
  local rate="$1"
  local ratio="$2"
  local split_path="$SCRIPT_DIR/splits/${DATASET_NAME}_visible${rate}_val10_seed_${SEED}.npz"
  local metrics_path="$SCRIPT_DIR/results/${DATASET_NAME}_moderntcn_visible${rate}_seed${SEED}.json"
  local name="${DATASET_NAME}_moderntcn_visible${rate}_seed${SEED}"
  (
    cd "$TCN_DIR"
    run_step "$name" \
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
      --ffn-ratio "$FFN_RATIO" \
      --patch-size "$PATCH_SIZE" \
      --patch-stride "$PATCH_STRIDE" \
      --num-blocks "$NUM_BLOCKS" \
      --large-size "$LARGE_SIZE" \
      --small-size "$SMALL_SIZE" \
      --dims "$DIMS" \
      --dw-dims "$DW_DIMS" \
      --dropout "$TCN_DROPOUT" \
      --head-dropout "$HEAD_DROPOUT" \
      --metrics-path "$metrics_path"
  )
}

require_file "$TENSOR_PATH"
require_dir "$TIMESNET_DIR"
require_dir "$TCN_DIR"
require_file "$TIMESNET_DIR/run_sat_tensor_experiment.py"
require_file "$TCN_DIR/run_sat_tensor_experiment.py"

echo "===== TimesNet / ModernTCN Iridium Visible-Rate Experiments ====="
echo "Start time: $(date)"
echo "Script dir: $SCRIPT_DIR"
echo "Repo dir: $REPO_DIR"
echo "Conda env: $CONDA_ENV"
echo "Tensor: $TENSOR_PATH"
echo "Dataset: $DATASET_NAME"
echo "TimesNet dir: $TIMESNET_DIR"
echo "ModernTCN dir: $TCN_DIR"
echo "Models: ${MODELS[*]}"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "Summary CSV: $summary_csv"
echo

for visible in "${VISIBLE_RATES[@]}"; do
  observed_ratio="$(rate_ratio "$visible")"
  for model in "${MODELS[@]}"; do
    case "$model" in
      TimesNet|timesnet)
        run_timesnet "$visible" "$observed_ratio"
        ;;
      ModernTCN|modernTCN|moderntcn|ModelnTCN|modelntcn)
        run_moderntcn "$visible" "$observed_ratio"
        ;;
      *)
        echo "Unsupported model in MODELS: $model" >&2
        exit 1
        ;;
    esac
  done
done

"$PYTHON_BIN" - "$SCRIPT_DIR/results" "$DATASET_NAME" "$SEED" "$summary_csv" "${MODELS[*]}" -- "${VISIBLE_RATES[@]}" <<'PY'
import csv
import json
import os
import sys

results_dir = sys.argv[1]
dataset = sys.argv[2]
seed = sys.argv[3]
summary_csv = sys.argv[4]
models_raw = sys.argv[5].split()
visible_rates = sys.argv[7:]

def normalize_model(name):
    low = name.lower()
    if low == "timesnet":
        return "timesnet"
    if low in ("moderntcn", "modelntcn"):
        return "moderntcn"
    return low

models = [normalize_model(name) for name in models_raw]

rows = []
for visible in visible_rates:
    for key in models:
        name = f"{dataset}_{key}_visible{visible}_seed{seed}"
        path = os.path.join(results_dir, name + ".json")
        if not os.path.exists(path):
            rows.append((name, "missing"))
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        test = payload.get("test", payload)
        rows.append((name, test["nrmse"]))

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "nrmse"])
    for name, nrmse in rows:
        writer.writerow([name, nrmse])

print("Saved CSV:", summary_csv)
print(f"{'name':60s}  {'nrmse':>10s}")
print("-" * 73)
for name, nrmse in rows:
    if isinstance(nrmse, float):
        print(f"{name:60s}  {nrmse:10.6f}")
    else:
        print(f"{name:60s}  {nrmse:>10s}")
PY

echo "===== TimesNet / ModernTCN visible-rate run finished at $(date) ====="
