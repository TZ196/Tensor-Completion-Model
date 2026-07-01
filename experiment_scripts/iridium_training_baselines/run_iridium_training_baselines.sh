#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/data_results}"

COSTCO_ENV="${COSTCO_ENV:-TZ-costco}"
SATFORMER_ENV="${SATFORMER_ENV:-TZ-Satformer}"

COSTCO_DIR="${COSTCO_DIR:-$REPO_DIR/CostCO}"
TIMESNET_DIR="${TIMESNET_DIR:-$REPO_DIR/TimesNet}"
MODERNTCN_DIR="${MODERNTCN_DIR:-$REPO_DIR/ModernTCN-imputation}"

SEED="${SEED:-3}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VISIBLE_RATES=(${VISIBLE_RATES:-1 3 5 7 10 20})
MODELS=(${MODELS:-costco timesnet moderntcn})

# Shared training defaults.
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-max}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"

# CoSTCo defaults.
COSTCO_LR="${COSTCO_LR:-1e-4}"
COSTCO_RANK="${COSTCO_RANK:-50}"
COSTCO_NC="${COSTCO_NC:-64}"
COSTCO_BATCH_SIZE="${COSTCO_BATCH_SIZE:-256}"

# TimesNet defaults.
TIMESNET_LR="${TIMESNET_LR:-1e-3}"
TIMESNET_D_MODEL="${TIMESNET_D_MODEL:-64}"
TIMESNET_D_FF="${TIMESNET_D_FF:-128}"
TIMESNET_E_LAYERS="${TIMESNET_E_LAYERS:-2}"
TIMESNET_TOP_K="${TIMESNET_TOP_K:-2}"
TIMESNET_NUM_KERNELS="${TIMESNET_NUM_KERNELS:-6}"
TIMESNET_DROPOUT="${TIMESNET_DROPOUT:-0.1}"

# ModernTCN defaults.
MODERNTCN_LR="${MODERNTCN_LR:-1e-3}"
MODERNTCN_FFN_RATIO="${MODERNTCN_FFN_RATIO:-1}"
MODERNTCN_PATCH_SIZE="${MODERNTCN_PATCH_SIZE:-1}"
MODERNTCN_PATCH_STRIDE="${MODERNTCN_PATCH_STRIDE:-1}"
MODERNTCN_NUM_BLOCKS="${MODERNTCN_NUM_BLOCKS:-1}"
MODERNTCN_LARGE_SIZE="${MODERNTCN_LARGE_SIZE:-31}"
MODERNTCN_SMALL_SIZE="${MODERNTCN_SMALL_SIZE:-5}"
MODERNTCN_DIMS="${MODERNTCN_DIMS:-64}"
MODERNTCN_DW_DIMS="${MODERNTCN_DW_DIMS:-64}"
MODERNTCN_DROPOUT="${MODERNTCN_DROPOUT:-0.1}"
MODERNTCN_HEAD_DROPOUT="${MODERNTCN_HEAD_DROPOUT:-0.0}"

mkdir -p "$SCRIPT_DIR/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="$SCRIPT_DIR/logs/iridium_training_baselines_${timestamp}.log"
pid_file="$SCRIPT_DIR/logs/iridium_training_baselines.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup env \
    PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    COSTCO_ENV="$COSTCO_ENV" \
    SATFORMER_ENV="$SATFORMER_ENV" \
    SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}" \
    COSTCO_DIR="$COSTCO_DIR" \
    TIMESNET_DIR="$TIMESNET_DIR" \
    MODERNTCN_DIR="$MODERNTCN_DIR" \
    SEED="$SEED" \
    VAL_RATIO="$VAL_RATIO" \
    VISIBLE_RATES="${VISIBLE_RATES[*]}" \
    MODELS="${MODELS[*]}" \
    TARGET_NORMALIZATION="$TARGET_NORMALIZATION" \
    EPOCHS="$EPOCHS" \
    PATIENCE="$PATIENCE" \
    COSTCO_LR="$COSTCO_LR" \
    COSTCO_RANK="$COSTCO_RANK" \
    COSTCO_NC="$COSTCO_NC" \
    COSTCO_BATCH_SIZE="$COSTCO_BATCH_SIZE" \
    TIMESNET_LR="$TIMESNET_LR" \
    TIMESNET_D_MODEL="$TIMESNET_D_MODEL" \
    TIMESNET_D_FF="$TIMESNET_D_FF" \
    TIMESNET_E_LAYERS="$TIMESNET_E_LAYERS" \
    TIMESNET_TOP_K="$TIMESNET_TOP_K" \
    TIMESNET_NUM_KERNELS="$TIMESNET_NUM_KERNELS" \
    TIMESNET_DROPOUT="$TIMESNET_DROPOUT" \
    MODERNTCN_LR="$MODERNTCN_LR" \
    MODERNTCN_FFN_RATIO="$MODERNTCN_FFN_RATIO" \
    MODERNTCN_PATCH_SIZE="$MODERNTCN_PATCH_SIZE" \
    MODERNTCN_PATCH_STRIDE="$MODERNTCN_PATCH_STRIDE" \
    MODERNTCN_NUM_BLOCKS="$MODERNTCN_NUM_BLOCKS" \
    MODERNTCN_LARGE_SIZE="$MODERNTCN_LARGE_SIZE" \
    MODERNTCN_SMALL_SIZE="$MODERNTCN_SMALL_SIZE" \
    MODERNTCN_DIMS="$MODERNTCN_DIMS" \
    MODERNTCN_DW_DIMS="$MODERNTCN_DW_DIMS" \
    MODERNTCN_DROPOUT="$MODERNTCN_DROPOUT" \
    MODERNTCN_HEAD_DROPOUT="$MODERNTCN_HEAD_DROPOUT" \
    bash "$SCRIPT_PATH" --foreground > "$master_log" 2>&1 &
  pid="$!"
  echo "$pid" > "$pid_file"
  echo "Started Iridium training baseline run in background."
  echo "PID: $pid"
  echo "Master log: $master_log"
  echo "PID file: $pid_file"
  echo "Watch: tail -f $master_log"
  exit 0
fi

export PYTHONHASHSEED="$SEED"

activate_conda_env() {
  local env_name="$1"
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    echo "Skipping conda activation; expected active env for: $env_name"
    return
  fi

  if ! command -v conda >/dev/null 2>&1 && [[ -z "${CONDA_EXE:-}" ]]; then
    if [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$HOME/.conda/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
      echo "Could not find conda. Set SKIP_CONDA_ACTIVATE=1 if the right env is already active." >&2
      exit 1
    fi
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
  else
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
  fi

  conda activate "$env_name"
  echo "Activated conda environment: $env_name"
}

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

run_logged() {
  local log_file="$1"
  shift
  echo "Command: $*"
  echo "Log: $log_file"
  "$@" > "$log_file" 2>&1
}

run_costco() {
  local tensor_path="$1"
  local result_dir="$2"
  local dataset_key="$3"
  local visible="$4"
  local ratio="$5"
  local missing=$((100 - visible))
  local name="${dataset_key}_costco_visible${visible}_missing${missing}_seed${SEED}"
  local split_path="$result_dir/splits/visible${visible}_missing${missing}_val10_seed_${SEED}.npz"
  local metrics_path="$result_dir/json/${name}.json"
  local log_file="$result_dir/logs/${name}.log"

  activate_conda_env "$COSTCO_ENV"
  (
    cd "$COSTCO_DIR"
    run_logged "$log_file" \
      "$PYTHON_BIN" run_sat_tensor_experiment.py \
      --tensor-path "$tensor_path" \
      --observed-ratio "$ratio" \
      --val-ratio "$VAL_RATIO" \
      --split-path "$split_path" \
      --rank "$COSTCO_RANK" \
      --nc "$COSTCO_NC" \
      --lr "$COSTCO_LR" \
      --epochs "$EPOCHS" \
      --batch-size "$COSTCO_BATCH_SIZE" \
      --target-normalization "$TARGET_NORMALIZATION" \
      --seed "$SEED" \
      --metrics-path "$metrics_path"
  )
}

run_timesnet() {
  local tensor_path="$1"
  local result_dir="$2"
  local dataset_key="$3"
  local visible="$4"
  local ratio="$5"
  local missing=$((100 - visible))
  local name="${dataset_key}_timesnet_visible${visible}_missing${missing}_seed${SEED}"
  local split_path="$result_dir/splits/visible${visible}_missing${missing}_val10_seed_${SEED}.npz"
  local metrics_path="$result_dir/json/${name}.json"
  local log_file="$result_dir/logs/${name}.log"

  activate_conda_env "$SATFORMER_ENV"
  (
    cd "$TIMESNET_DIR"
    run_logged "$log_file" \
      "$PYTHON_BIN" run_sat_tensor_experiment.py \
      --tensor-path "$tensor_path" \
      --observed-ratio "$ratio" \
      --val-ratio "$VAL_RATIO" \
      --split-path "$split_path" \
      --lr "$TIMESNET_LR" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --target-normalization "$TARGET_NORMALIZATION" \
      --seed "$SEED" \
      --d-model "$TIMESNET_D_MODEL" \
      --d-ff "$TIMESNET_D_FF" \
      --e-layers "$TIMESNET_E_LAYERS" \
      --top-k "$TIMESNET_TOP_K" \
      --num-kernels "$TIMESNET_NUM_KERNELS" \
      --dropout "$TIMESNET_DROPOUT" \
      --metrics-path "$metrics_path"
  )
}

run_moderntcn() {
  local tensor_path="$1"
  local result_dir="$2"
  local dataset_key="$3"
  local visible="$4"
  local ratio="$5"
  local missing=$((100 - visible))
  local name="${dataset_key}_moderntcn_visible${visible}_missing${missing}_seed${SEED}"
  local split_path="$result_dir/splits/visible${visible}_missing${missing}_val10_seed_${SEED}.npz"
  local metrics_path="$result_dir/json/${name}.json"
  local log_file="$result_dir/logs/${name}.log"

  activate_conda_env "$SATFORMER_ENV"
  (
    cd "$MODERNTCN_DIR"
    run_logged "$log_file" \
      "$PYTHON_BIN" run_sat_tensor_experiment.py \
      --tensor-path "$tensor_path" \
      --observed-ratio "$ratio" \
      --val-ratio "$VAL_RATIO" \
      --split-path "$split_path" \
      --lr "$MODERNTCN_LR" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --target-normalization "$TARGET_NORMALIZATION" \
      --seed "$SEED" \
      --ffn-ratio "$MODERNTCN_FFN_RATIO" \
      --patch-size "$MODERNTCN_PATCH_SIZE" \
      --patch-stride "$MODERNTCN_PATCH_STRIDE" \
      --num-blocks "$MODERNTCN_NUM_BLOCKS" \
      --large-size "$MODERNTCN_LARGE_SIZE" \
      --small-size "$MODERNTCN_SMALL_SIZE" \
      --dims "$MODERNTCN_DIMS" \
      --dw-dims "$MODERNTCN_DW_DIMS" \
      --dropout "$MODERNTCN_DROPOUT" \
      --head-dropout "$MODERNTCN_HEAD_DROPOUT" \
      --metrics-path "$metrics_path"
  )
}

write_dataset_csv() {
  local result_dir="$1"
  local dataset_key="$2"
  local dataset_label="$3"
  local csv_path="$result_dir/csv/${dataset_key}_visible_rate_metrics_seed${SEED}.csv"
  "$PYTHON_BIN" - "$result_dir/json" "$csv_path" "$dataset_key" "$dataset_label" "$SEED" "${MODELS[*]}" -- "${VISIBLE_RATES[@]}" <<'PY'
import csv
import json
import os
import sys

json_dir, csv_path, dataset_key, dataset_label, seed, models_raw = sys.argv[1:7]
visible_rates = sys.argv[8:]
models = [item.lower() for item in models_raw.split()]

rows = []
for visible in visible_rates:
    missing = str(100 - int(visible))
    for model in models:
        name = f"{dataset_key}_{model}_visible{visible}_missing{missing}_seed{seed}"
        path = os.path.join(json_dir, name + ".json")
        row = {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "model": model,
            "visible_rate_percent": visible,
            "missing_rate_percent": missing,
            "seed": seed,
            "metrics_json": path,
            "status": "missing",
            "mae": "",
            "rmse": "",
            "nmae": "",
            "nrmse": "",
            "test_entries": "",
        }
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            test = payload.get("test", payload)
            row.update({
                "status": "ok",
                "mae": test.get("mae", ""),
                "rmse": test.get("rmse", ""),
                "nmae": test.get("nmae", ""),
                "nrmse": test.get("nrmse", ""),
                "test_entries": test.get("entries", ""),
            })
        rows.append(row)

os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fields = [
    "dataset_key", "dataset_label", "model", "visible_rate_percent",
    "missing_rate_percent", "seed", "status", "mae", "rmse", "nmae",
    "nrmse", "test_entries", "metrics_json",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("Saved dataset CSV:", csv_path)
PY
}

write_master_csv() {
  local master_csv="$OUTPUT_ROOT/iridium/training_data_all_baselines_seed${SEED}.csv"
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$master_csv" <<'PY'
import csv
import glob
import os
import sys

output_root, master_csv = sys.argv[1:3]
csv_files = [
    path for path in glob.glob(os.path.join(output_root, "iridium", "*", "training_data", "csv", "*.csv"))
    if not os.path.basename(path).startswith("training_data_all")
]
rows = []
fields = None
for path in sorted(csv_files):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fields is None:
            fields = reader.fieldnames
        rows.extend(reader)
if fields is None:
    fields = [
        "dataset_key", "dataset_label", "model", "visible_rate_percent",
        "missing_rate_percent", "seed", "status", "mae", "rmse", "nmae",
        "nrmse", "test_entries", "metrics_json",
    ]
os.makedirs(os.path.dirname(master_csv), exist_ok=True)
with open(master_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("Saved master CSV:", master_csv)
PY
}

require_dir "$COSTCO_DIR"
require_dir "$TIMESNET_DIR"
require_dir "$MODERNTCN_DIR"
require_file "$COSTCO_DIR/run_sat_tensor_experiment.py"
require_file "$TIMESNET_DIR/run_sat_tensor_experiment.py"
require_file "$MODERNTCN_DIR/run_sat_tensor_experiment.py"

DATASETS=(
  "iridium_flow|流级仿真|$REPO_DIR/data/iridium/流级仿真/training_data/traffic7200s.npy|$OUTPUT_ROOT/iridium/流级仿真/training_data"
  "iridium_packet|包级仿真|$REPO_DIR/data/iridium/包级仿真/training_data/traffic120s.npy|$OUTPUT_ROOT/iridium/包级仿真/training_data"
)

echo "===== Iridium training-data baselines ====="
echo "Start time: $(date)"
echo "Repo: $REPO_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Models: ${MODELS[*]}"
echo "Visible rates: ${VISIBLE_RATES[*]}"
echo "CostCO env: $COSTCO_ENV"
echo "TimesNet/ModernTCN env: $SATFORMER_ENV"
echo "Seed: $SEED"
echo "Epochs: $EPOCHS"
echo

for item in "${DATASETS[@]}"; do
  IFS='|' read -r dataset_key dataset_label tensor_path result_dir <<< "$item"
  require_file "$tensor_path"
  mkdir -p "$result_dir/json" "$result_dir/csv" "$result_dir/logs" "$result_dir/splits"

  echo "===== Dataset: $dataset_label ($dataset_key) ====="
  echo "Tensor: $tensor_path"
  echo "Results: $result_dir"
  echo

  for visible in "${VISIBLE_RATES[@]}"; do
    ratio="$(rate_ratio "$visible")"
    missing=$((100 - visible))
    echo "----- Visible ${visible}% / Missing ${missing}% -----"
    for model in "${MODELS[@]}"; do
      case "${model,,}" in
        costco)
          run_costco "$tensor_path" "$result_dir" "$dataset_key" "$visible" "$ratio"
          ;;
        timesnet)
          run_timesnet "$tensor_path" "$result_dir" "$dataset_key" "$visible" "$ratio"
          ;;
        moderntcn|modelntcn)
          run_moderntcn "$tensor_path" "$result_dir" "$dataset_key" "$visible" "$ratio"
          ;;
        *)
          echo "Unsupported model: $model" >&2
          exit 1
          ;;
      esac
      echo "Finished ${dataset_key}/${model}/visible${visible} at $(date)"
      echo
    done
  done
  write_dataset_csv "$result_dir" "$dataset_key" "$dataset_label"
done

write_master_csv
echo "===== Finished at $(date) ====="
