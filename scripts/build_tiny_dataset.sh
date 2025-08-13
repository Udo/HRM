#!/usr/bin/env bash
set -euo pipefail

# Build a minimal synthetic dataset for smoke testing.
: "${OUTPUT_DIR:=data/tiny}"
: "${TRAIN_SIZE:=8}"
: "${TEST_SIZE:=4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv
python dataset/build_tiny_dataset.py --output-dir "$OUTPUT_DIR" --train-size "$TRAIN_SIZE" --test-size "$TEST_SIZE"
