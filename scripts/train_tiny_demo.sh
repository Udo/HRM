#!/usr/bin/env bash
set -euo pipefail

# End-to-end tiny demo: env check, dataset build (if missing), short training.
# Usage: ./scripts/train_tiny_demo.sh [extra hydra overrides]

# Defaults (resolved lazily after parsing positional KEY=VAL assignments)
: "${OUTPUT_DIR:=data/tiny}"
: "${EPOCHS:=50}"
: "${EVAL_INTERVAL:=10}"
: "${GBS:=16}"
: "${HIDDEN:=128}"
: "${HEADS:=4}"
: "${H_LAYERS:=1}"
: "${L_LAYERS:=1}"
: "${H_CYC:=1}"
: "${L_CYC:=1}"
: "${LR:=1e-3}"
: "${PUZ_LR:=1e-3}"
: "${CKPT_EVERY:=0}"  # 1 to enable checkpoint_every_eval

# Disable heavier features by default for portability
export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

# Ensure venv on PATH (no full activation needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

############################################################
# Backwards compat: allow passing VAR=VAL (uppercase) as args
# e.g. ./scripts/train_tiny_demo.sh EPOCHS=20 EVAL_INTERVAL=5
############################################################
PREF_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    [A-Z]*=*)
      export "$1"  # updates env; value after '='
      shift
      ;;
    *)
      break;;
  esac
done

# Re-sync local vars from (possibly updated) environment
EPOCHS=${EPOCHS:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
GBS=${GBS:-16}
HIDDEN=${HIDDEN:-128}
HEADS=${HEADS:-4}
H_LAYERS=${H_LAYERS:-1}
L_LAYERS=${L_LAYERS:-1}
H_CYC=${H_CYC:-1}
L_CYC=${L_CYC:-1}
LR=${LR:-1e-3}
PUZ_LR=${PUZ_LR:-1e-3}
CKPT_EVERY=${CKPT_EVERY:-0}

if [ ! -f "$OUTPUT_DIR/train/dataset.json" ]; then
  echo "[INFO] Building tiny dataset at $OUTPUT_DIR" >&2
  python dataset/build_tiny_dataset.py --output-dir "$OUTPUT_DIR" --train-size 16 --test-size 8
fi

EXTRA_ARGS=()
if [ "$CKPT_EVERY" = "1" ]; then
  EXTRA_ARGS+=(checkpoint_every_eval=true)
fi

# Unified invocation via _hrm_run_pretrain (DRY with other train scripts)
OVERRIDES=(
  data_path="$OUTPUT_DIR"
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
  weight_decay=0.0
  puzzle_emb_weight_decay=0.0
  arch.loss.loss_type=softmax_cross_entropy
)
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  _hrm_run_pretrain "${OVERRIDES[@]}" "${EXTRA_ARGS[@]}" "$@"
else
  _hrm_run_pretrain "${OVERRIDES[@]}" "$@"
fi
