#!/usr/bin/env bash
set -euo pipefail

# Build maze dataset (30x30 hard 1k) with optional subsample + augmentation.
# Usage:
#   ./scripts/build_maze_1k.sh [SUBSAMPLE=...] [AUG=1] [OUTPUT_DIR=...]
# Env Vars:
#   OUTPUT_DIR   (default data/maze-30x30-hard-1k)
#   SUBSAMPLE    (optional integer; if set, limits train examples)
#   AUG          (0|1 enable 8x dihedral augmentation on train set)

: "${OUTPUT_DIR:=data/maze-30x30-hard-1k}"
: "${SUBSAMPLE:=}"    # empty for full
: "${AUG:=0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

ARGS=( --output-dir "$OUTPUT_DIR" )
if [ -n "$SUBSAMPLE" ]; then
  ARGS+=( --subsample-size "$SUBSAMPLE" )
fi
if [ "$AUG" = "1" ]; then
  ARGS+=( --aug )
fi
python dataset/build_maze_dataset.py "${ARGS[@]}"
