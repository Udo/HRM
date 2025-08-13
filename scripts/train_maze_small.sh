#!/usr/bin/env bash
set -euo pipefail

# Train a small model on maze dataset.
# Usage: ./scripts/train_maze_small.sh [VAR=val ...] [-- hydra overrides]

: "${DATA_PATH:=data/maze-30x30-hard-1k}"
: "${EPOCHS:=800}"
: "${EVAL_INTERVAL:=80}"
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
: "${WARMUP_STEPS:=50}"

# Parse leading VAR=VAL args
while [ $# -gt 0 ]; do
  case "$1" in
    [A-Z]*=*) export "$1"; shift;;
    --) break;;
    *) break;;
  esac
done

EPOCHS=${EPOCHS:-800}
EVAL_INTERVAL=${EVAL_INTERVAL:-80}
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

export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

OVERRIDES=(
  data_path="$DATA_PATH"
  epochs="$EPOCHS"
  eval_interval="$EVAL_INTERVAL"
  global_batch_size="$GBS"
  lr_warmup_steps="$WARMUP_STEPS"
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
