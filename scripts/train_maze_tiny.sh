#!/usr/bin/env bash
set -euo pipefail
# Ultra small maze training for smoke tests.
: "${DATASET_DIR:=data/maze-30x30-hard-1k}"
: "${EPOCHS:=40}"
: "${EVAL_INTERVAL:=10}"
: "${GBS:=64}"
: "${HIDDEN:=128}"
: "${HEADS:=4}"
: "${H_LAYERS:=1}"
: "${L_LAYERS:=1}"
: "${H_CYC:=1}"
: "${L_CYC:=1}"
: "${LR:=1e-3}"
: "${PUZ_LR:=1e-3}"
: "${WD:=0.01}"
: "${PUZ_WD:=0.01}"
: "${CKPT_EVERY:=0}"
: "${WARMUP_STEPS:=20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv
TRAIN_EXTRA=()
if [ "$CKPT_EVERY" = "1" ]; then TRAIN_EXTRA+=(checkpoint_every_eval=true); fi
OVERRIDES=(
  data_path="$DATASET_DIR"
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
_hrm_run_pretrain "${OVERRIDES[@]}" ${TRAIN_EXTRA:+"${TRAIN_EXTRA[@]}"}
