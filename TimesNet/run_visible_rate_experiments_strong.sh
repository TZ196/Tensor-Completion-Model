#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export EPOCHS="${EPOCHS:-200}"
export TOP_K="${TOP_K:-4}"
export NUM_KERNELS="${NUM_KERNELS:-10}"
export DROPOUT="${DROPOUT:-0.2}"

exec bash "$SCRIPT_DIR/run_visible_rate_experiments.sh" "$@"
