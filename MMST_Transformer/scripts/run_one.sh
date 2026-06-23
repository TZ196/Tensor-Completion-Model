#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
VISIBLE_RATE="${VISIBLE_RATE:-7}"
VARIANT="${VARIANT:-M2}"
SEED="${SEED:-3}"

ratio="$(awk "BEGIN { printf \"%.2f\", ${VISIBLE_RATE}/100 }")"
mkdir -p logs results splits checkpoints histories

export PYTHONHASHSEED="$SEED"
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_ENABLE_ONEDNN_OPTS=0

name="vis${VISIBLE_RATE}_${VARIANT}_gt_mst_seed${SEED}"

"$PYTHON_BIN" run_experiment.py \
  --tensor-path "${TENSOR_PATH:-../CostCO/sat_path_bytes_mb_tensor.npy}" \
  --topology-path "${TOPOLOGY_PATH:-../CostCO/sat_connectivity_tensor_dynamic_60s_1000ms.npz}" \
  --mode-text-dir "${MODE_TEXT_DIR:-../CostCO/mode_text_numeric_ablation_data/both}" \
  --variant "$VARIANT" \
  --observed-ratio "$ratio" \
  --split-path "splits/random_observed${VISIBLE_RATE}_val10_seed_${SEED}.npz" \
  --d-model "${D_MODEL:-64}" \
  --node-dim "${NODE_DIM:-64}" \
  --gcn-dim "${GCN_DIM:-128}" \
  --transformer-layers "${TRANSFORMER_LAYERS:-2}" \
  --num-heads "${NUM_HEADS:-4}" \
  --ff-dim "${FF_DIM:-128}" \
  --dropout "${DROPOUT:-0.1}" \
  --lr "${LR:-1e-3}" \
  --epochs "${EPOCHS:-200}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --target-normalization "${TARGET_NORMALIZATION:-max}" \
  --seed "$SEED" \
  --text-align-target-ratio "${TEXT_ALIGN_TARGET_RATIO:-0.01}" \
  --alignment-temperature "${ALIGNMENT_TEMPERATURE:-0.2}" \
  --temporal-delta "${TEMPORAL_DELTA:-2}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-20}" \
  --metrics-path "results/${name}.json"

