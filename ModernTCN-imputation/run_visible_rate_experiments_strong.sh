#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export EPOCHS="${EPOCHS:-200}"
export NUM_BLOCKS="${NUM_BLOCKS:-1}"
export LARGE_SIZE="${LARGE_SIZE:-31}"
export SMALL_SIZE="${SMALL_SIZE:-5}"
export DIMS="${DIMS:-96}"
export DW_DIMS="${DW_DIMS:-96}"
export DROPOUT="${DROPOUT:-0.2}"
export HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"

exec bash "$SCRIPT_DIR/run_visible_rate_experiments.sh" "$@"
