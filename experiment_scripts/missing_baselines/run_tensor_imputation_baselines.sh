#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-TZ-Satformer}"
TENSOR_PATH="${TENSOR_PATH:-$REPO_DIR/data/iridium5400s/iridium.npy}"
DATASET_NAME="${DATASET_NAME:-iridium5400s}"
SEED="${SEED:-3}"
VISIBLE_RATES="${VISIBLE_RATES:-1 3 5 7 10 20}"
MODELS="${MODELS:-NTC NTF NTM SAITS CSDI PriSTI}"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/results"

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="$SCRIPT_DIR/logs/${DATASET_NAME}_missing_baselines_${timestamp}.log"
pid_file="$SCRIPT_DIR/logs/${DATASET_NAME}_missing_baselines.pid"
summary_csv="$SCRIPT_DIR/results/${DATASET_NAME}_missing_baselines_nrmse_seed${SEED}.csv"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup env \
    PYTHON_BIN="$PYTHON_BIN" \
    CONDA_ENV="$CONDA_ENV" \
    SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}" \
    TENSOR_PATH="$TENSOR_PATH" \
    DATASET_NAME="$DATASET_NAME" \
    SEED="$SEED" \
    VISIBLE_RATES="$VISIBLE_RATES" \
    MODELS="$MODELS" \
    bash "$SCRIPT_PATH" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started tensor imputation baselines in background."
  echo "PID: $pid"
  echo "Master log: $master_log"
  echo "PID file: $pid_file"
  echo "Watch: tail -f $master_log"
  exit 0
fi

echo "===== Tensor Imputation Baselines ====="
echo "Start time: $(date)"
echo "Repo dir: $REPO_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Conda env: $CONDA_ENV"
echo "Models: $MODELS"
echo "Visible rates: $VISIBLE_RATES"
echo "Summary CSV: $summary_csv"
echo

for model in $MODELS; do
  model_dir="$REPO_DIR/$model"
  if [[ ! -d "$model_dir" ]]; then
    echo "Missing model directory: $model_dir" >&2
    exit 1
  fi
  echo "----- Running $model -----"
  (
    cd "$model_dir"
    SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}" \
    CONDA_ENV="$CONDA_ENV" \
    PYTHON_BIN="$PYTHON_BIN" \
    TENSOR_PATH="$TENSOR_PATH" \
    DATASET_NAME="$DATASET_NAME" \
    SEED="$SEED" \
    VISIBLE_RATES="$VISIBLE_RATES" \
    ./run_visible_rate_experiments.sh --foreground
  )
done

"$PYTHON_BIN" - "$REPO_DIR" "$DATASET_NAME" "$SEED" "$summary_csv" "$MODELS" -- $VISIBLE_RATES <<'PY'
import csv
import json
import os
import sys

repo_dir = sys.argv[1]
dataset = sys.argv[2]
seed = sys.argv[3]
summary_csv = sys.argv[4]
models = sys.argv[5].split()
rates = sys.argv[7:]

rows = []
for model in models:
    key = model.lower()
    results_dir = os.path.join(repo_dir, model, "results")
    for rate in rates:
        name = f"{dataset}_{key}_visible{rate}_seed{seed}"
        path = os.path.join(results_dir, name + ".json")
        if not os.path.exists(path):
            rows.append((name, "missing"))
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        rows.append((name, payload["test"]["nrmse"]))

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "nrmse"])
    writer.writerows(rows)

print("Saved CSV:", summary_csv)
print(f"{'name':55s}  {'nrmse':>10s}")
print("-" * 68)
for name, nrmse in rows:
    if isinstance(nrmse, float):
        print(f"{name:55s}  {nrmse:10.6f}")
    else:
        print(f"{name:55s}  {nrmse:>10s}")
PY

echo "===== Tensor imputation baselines finished at $(date) ====="
