#!/usr/bin/env bash
set -euo pipefail

# Download + build a 1k augmented Sudoku dataset (fast subset)
: "${OUTPUT_DIR:=data/sudoku-extreme-1k-aug-10}"
: "${SUBSAMPLE:=1000}"
: "${AUG:=10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv
python dataset/build_sudoku_dataset.py --output-dir "$OUTPUT_DIR" --subsample-size "$SUBSAMPLE" --num-aug "$AUG"
