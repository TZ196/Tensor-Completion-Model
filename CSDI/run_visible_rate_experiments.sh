#!/usr/bin/env bash
MODEL_NAME="csdi"
MODEL_LABEL="CSDI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/tensor_baseline_lib/run_visible_rate_template.sh"
