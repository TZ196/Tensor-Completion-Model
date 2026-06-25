set -euo pipefail

cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-TZ-Satformer}"
TENSOR_PATH="${TENSOR_PATH:-$REPO_DIR/data/iridium5400s/iridium.npy}"
DATASET_NAME="${DATASET_NAME:-iridium5400s}"

SEED="${SEED:-3}"
VAL_RATIO="${VAL_RATIO:-0.1}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16384}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
RANK="${RANK:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
MLP_DEPTH="${MLP_DEPTH:-2}"
TENSOR_RANK="${TENSOR_RANK:-32}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-16}"

VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})

mkdir -p logs results splits

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/${MODEL_NAME}_visible_rate_${timestamp}.log"
pid_file="logs/${MODEL_NAME}_visible_rate.pid"
summary_csv="results/${DATASET_NAME}_${MODEL_NAME}_visible_rate_nrmse_seed${SEED}.csv"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup env \
    PYTHON_BIN="$PYTHON_BIN" \
    CONDA_ENV="$CONDA_ENV" \
    SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}" \
    TENSOR_PATH="$TENSOR_PATH" \
    DATASET_NAME="$DATASET_NAME" \
    SEED="$SEED" \
    VAL_RATIO="$VAL_RATIO" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    PATIENCE="$PATIENCE" \
    BATCH_SIZE="$BATCH_SIZE" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    TARGET_NORMALIZATION="$TARGET_NORMALIZATION" \
    RANK="$RANK" \
    HIDDEN_DIM="$HIDDEN_DIM" \
    MLP_DEPTH="$MLP_DEPTH" \
    TENSOR_RANK="$TENSOR_RANK" \
    LAYERS="$LAYERS" \
    HEADS="$HEADS" \
    DROPOUT="$DROPOUT" \
    DIFFUSION_STEPS="$DIFFUSION_STEPS" \
    VISIBLE_RATES="${VISIBLE_RATES[*]}" \
    bash "$SCRIPT_DIR/run_visible_rate_experiments.sh" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started ${MODEL_LABEL} visible-rate experiments in background."
  echo "PID: $pid"
  echo "Master log: $SCRIPT_DIR/$master_log"
  echo "PID file: $SCRIPT_DIR/$pid_file"
  echo "Watch: tail -f $SCRIPT_DIR/$master_log"
  exit 0
fi

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

export PYTHONHASHSEED="$SEED"

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

echo "===== ${MODEL_LABEL} Visible-Rate Experiments ====="
echo "Start time: $(date)"
echo "Work dir: $SCRIPT_DIR"
echo "Tensor: $TENSOR_PATH"
echo "Conda env: $CONDA_ENV"
echo "Seed: $SEED"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch size: $BATCH_SIZE"
echo "Summary CSV: $summary_csv"
echo

for rate in "${VISIBLE_RATES[@]}"; do
  ratio="$(rate_ratio "$rate")"
  split_path="splits/${DATASET_NAME}_visible${rate}_val10_seed_${SEED}.npz"
  metrics_path="results/${DATASET_NAME}_${MODEL_NAME}_visible${rate}_seed${SEED}.json"
  run_step "${DATASET_NAME}_${MODEL_NAME}_visible${rate}_seed${SEED}" \
    "$PYTHON_BIN" run_sat_tensor_experiment.py \
    --model-name "$MODEL_NAME" \
    --tensor-path "$TENSOR_PATH" \
    --observed-ratio "$ratio" \
    --val-ratio "$VAL_RATIO" \
    --split-path "$split_path" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --target-normalization "$TARGET_NORMALIZATION" \
    --seed "$SEED" \
    --rank "$RANK" \
    --hidden-dim "$HIDDEN_DIM" \
    --mlp-depth "$MLP_DEPTH" \
    --tensor-rank "$TENSOR_RANK" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --dropout "$DROPOUT" \
    --diffusion-steps "$DIFFUSION_STEPS" \
    --metrics-path "$metrics_path"
done

"$PYTHON_BIN" - "$MODEL_NAME" "$DATASET_NAME" "$SEED" "$summary_csv" "${VISIBLE_RATES[@]}" <<'PY'
import csv
import json
import os
import sys

model = sys.argv[1]
dataset = sys.argv[2]
seed = sys.argv[3]
summary_csv = sys.argv[4]
visible_rates = sys.argv[5:]

rows = []
for visible in visible_rates:
    name = f"{dataset}_{model}_visible{visible}_seed{seed}"
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

echo "===== ${MODEL_LABEL} visible-rate experiments finished at $(date) ====="
