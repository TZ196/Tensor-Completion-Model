#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COSTCO_DIR="$REPO_DIR/CostCO"
cd "$COSTCO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PATH="${TENSOR_PATH:-$REPO_DIR/data/iridium5400s/iridium.npy}"
DATASET_NAME="${DATASET_NAME:-iridium5400s}"
CONDA_ENV="${CONDA_ENV:-TZ-costco}"

SEED="${SEED:-3}"
RANK="${RANK:-50}"
NC="${NC:-64}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# These are observed/visible rates in percent.
VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})

mkdir -p logs results splits

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/${DATASET_NAME}_costco_visible_rate_${timestamp}.log"
pid_file="logs/${DATASET_NAME}_costco_visible_rate.pid"
summary_csv="results/${DATASET_NAME}_costco_visible_rate_nrmse_seed${SEED}.csv"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup env \
    PYTHON_BIN="$PYTHON_BIN" \
    TENSOR_PATH="$TENSOR_PATH" \
    DATASET_NAME="$DATASET_NAME" \
    CONDA_ENV="$CONDA_ENV" \
    SEED="$SEED" \
    RANK="$RANK" \
    NC="$NC" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    TARGET_NORMALIZATION="$TARGET_NORMALIZATION" \
    VAL_RATIO="$VAL_RATIO" \
    VISIBLE_RATES="${VISIBLE_RATES[*]}" \
    bash "$SCRIPT_PATH" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started CoSTCo iridium visible-rate run in background."
  echo "PID: $pid"
  echo "Master log: $COSTCO_DIR/$master_log"
  echo "PID file: $COSTCO_DIR/$pid_file"
  echo "Watch: tail -f $COSTCO_DIR/$master_log"
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

echo "===== CoSTCo Iridium Visible-Rate Experiments ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Repo dir: $REPO_DIR"
echo "CoSTCo dir: $COSTCO_DIR"
echo "Conda env: $CONDA_ENV"
echo "Tensor: $TENSOR_PATH"
echo "Dataset: $DATASET_NAME"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Rank: $RANK"
echo "NC: $NC"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Summary CSV: $summary_csv"
echo

for visible in "${VISIBLE_RATES[@]}"; do
  observed_ratio="$(rate_ratio "$visible")"
  missing_ratio="$(awk "BEGIN { printf \"%.2f\", 1 - ${observed_ratio} }")"
  name="${DATASET_NAME}_costco_visible${visible}_seed${SEED}"
  split_path="splits/${DATASET_NAME}_visible${visible}_val$(printf '%.0f' "$(awk "BEGIN { print ${VAL_RATIO} * 100 }")")_seed_${SEED}.npz"
  metrics_path="results/${name}.json"
  run_step "$name" \
    "$PYTHON_BIN" run_sat_tensor_experiment.py \
    --tensor-path "$TENSOR_PATH" \
    --missing-rate "$missing_ratio" \
    --observed-ratio "$observed_ratio" \
    --val-ratio "$VAL_RATIO" \
    --split-path "$split_path" \
    --rank "$RANK" \
    --nc "$NC" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --target-normalization "$TARGET_NORMALIZATION" \
    --seed "$SEED" \
    --metrics-path "$metrics_path"
done

"$PYTHON_BIN" - "$DATASET_NAME" "$SEED" "$summary_csv" "${VISIBLE_RATES[@]}" <<'PY'
import csv
import json
import os
import sys

dataset = sys.argv[1]
seed = sys.argv[2]
summary_csv = sys.argv[3]
visible_rates = sys.argv[4:]

rows = []
for visible in visible_rates:
    name = f"{dataset}_costco_visible{visible}_seed{seed}"
    path = os.path.join("results", name + ".json")
    if not os.path.exists(path):
        rows.append((name, "missing"))
        continue
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows.append((name, payload["test"]["nrmse"]))

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "nrmse"])
    for name, nrmse in rows:
        writer.writerow([name, nrmse])

print("Saved CSV:", summary_csv)
print(f"{'name':55s}  {'nrmse':>10s}")
print("-" * 68)
for name, nrmse in rows:
    if isinstance(nrmse, float):
        print(f"{name:55s}  {nrmse:10.6f}")
    else:
        print(f"{name:55s}  {nrmse:>10s}")
PY

echo "===== CoSTCo iridium visible-rate run finished at $(date) ====="
