#!/usr/bin/env bash
set -euo pipefail

# Train a small HRM on the 1k Sudoku dataset with modest model size (good for MPS / single GPU)
# Override any var by exporting before calling or passing KEY=VALUE pairs (Hydra style) after --

# Default to small augmented subset; allow user to point at full dataset (data/sudoku-extreme-full) or any other.
: "${DATA_PATH:=data/sudoku-extreme-1k-aug-10}"
: "${EPOCHS:=2000}"           # keep small for quick iterations
: "${EVAL_INTERVAL:=200}"     # evaluate every N epochs
: "${GBS:=128}"
: "${HIDDEN:=256}"
: "${HEADS:=8}"
: "${H_LAYERS:=2}"
: "${L_LAYERS:=2}"
: "${H_CYC:=2}"
: "${L_CYC:=2}"
: "${LR:=5e-4}"
: "${PUZ_LR:=5e-4}"
: "${WD:=0.1}"
: "${PUZ_WD:=0.1}"

# Disable compile & wandb by default for portability unless user overrides
: "${DISABLE_COMPILE:=1}"
: "${HRM_DISABLE_WANDB:=1}"
# Optionally disable sparse embedding optimizer if running on MPS
: "${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:=1}"

export DISABLE_COMPILE HRM_DISABLE_WANDB HRM_DISABLE_SPARSE_EMB_OPTIMIZER

# Backwards compat: parse leading VAR=VAL args (uppercase) & export
while [ $# -gt 0 ]; do
  case "$1" in
    [A-Z]*=*) export "$1"; shift;;
    *) break;;
  esac
done
# Re-sync locals from env (in case updated)
EPOCHS=${EPOCHS:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
GBS=${GBS:-128}
HIDDEN=${HIDDEN:-256}
HEADS=${HEADS:-8}
H_LAYERS=${H_LAYERS:-2}
L_LAYERS=${L_LAYERS:-2}
H_CYC=${H_CYC:-2}
L_CYC=${L_CYC:-2}
LR=${LR:-5e-4}
PUZ_LR=${PUZ_LR:-5e-4}
WD=${WD:-0.1}
PUZ_WD=${PUZ_WD:-0.1}

# Ensure venv PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

# Auto-build small subset if pointing at default path and it is missing.
if [ ! -f "$DATA_PATH/train/dataset.json" ]; then
  if [ "$DATA_PATH" = "data/sudoku-extreme-1k-aug-10" ]; then
    echo "[INFO] Small Sudoku subset missing; building (subsample=1000 aug=10) -> $DATA_PATH" >&2
    python dataset/build_sudoku_dataset.py --output-dir "$DATA_PATH" --subsample-size 1000 --num-aug 10 || {
      echo "[WARN] Failed to build small subset; will attempt to fall back to full dataset if present." >&2
    }
  fi
fi

# Fallback: if still missing, but full dataset exists, switch to it.
if [ ! -f "$DATA_PATH/train/dataset.json" ] && [ -f data/sudoku-extreme-full/train/dataset.json ]; then
  echo "[INFO] Falling back to full Sudoku dataset at data/sudoku-extreme-full" >&2
  DATA_PATH=data/sudoku-extreme-full
fi

if [ ! -f "$DATA_PATH/train/dataset.json" ]; then
  cat >&2 <<EOF
[ERROR] Sudoku dataset not found at $DATA_PATH
Run one of:
  ./scripts/fetch_all_datasets.sh BUILD_SUDOKU_1K=1   # build both full + 1k subset
  python dataset/build_sudoku_dataset.py --output-dir $DATA_PATH --subsample-size 1000 --num-aug 10
Or point DATA_PATH at an existing dataset directory.
EOF
  exit 2
fi

# Unified invocation via _hrm_run_pretrain wrapper
OVERRIDES=(
  data_path="$DATA_PATH"
  epochs="$EPOCHS"
  eval_interval="$EVAL_INTERVAL"
  global_batch_size="$GBS"
  arch.hidden_size="$HIDDEN"
  arch.num_heads="$HEADS"
  arch.H_layers="$H_LAYERS"
  arch.L_layers="$L_LAYERS"
  arch.H_cycles="$H_CYC"
  arch.L_cycles="$L_CYC"
  lr="$LR"
  puzzle_emb_lr="$PUZ_LR"
  weight_decay="$WD"
  puzzle_emb_weight_decay="$PUZ_WD"
  arch.loss.loss_type=softmax_cross_entropy
)
_hrm_run_pretrain "${OVERRIDES[@]}" "$@"
