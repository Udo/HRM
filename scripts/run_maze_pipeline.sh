#!/usr/bin/env bash
set -euo pipefail

# Maze pipeline: build dataset (optional), train, evaluate summary, benchmark.
# Usage:
#   ./scripts/run_maze_pipeline.sh [-- hydra.overrides]
# Env Vars:
#   DATASET_DIR (default data/maze-30x30-hard-1k)
#   SUBSAMPLE   (optional int)
#   AUG         (0|1 dihedral aug)
#   SKIP_BUILD  (0|1)
#   EPOCHS/EVAL_INTERVAL/GBS/HIDDEN/HEADS/H_LAYERS/L_LAYERS/H_CYC/L_CYC/LR/PUZ_LR/WD/PUZ_WD
#   CKPT_EVERY SUMMARY BENCHMARK (same semantics as other pipelines)

: "${DATASET_DIR:=data/maze-30x30-hard-1k}"
: "${SUBSAMPLE:=}"  # empty -> full
: "${AUG:=0}"
: "${SKIP_BUILD:=0}"
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
: "${CKPT_EVERY:=0}"
: "${SUMMARY:=1}"
: "${BENCHMARK:=1}"

HYDRA_OVERRIDES=()
for a in "$@"; do
  if [ "$a" = "--" ]; then
    shift
    HYDRA_OVERRIDES=("$@")
    break
  fi
  shift || true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

start_ts=$(date +%s)

if [ "$SKIP_BUILD" != "1" ]; then
  if [ -f "$DATASET_DIR/train/dataset.json" ]; then
    echo "[INFO] Maze dataset exists -> skip build" >&2
  else
    echo "[INFO] Building maze dataset -> $DATASET_DIR" >&2
    SUB_ARGS=()
    if [ -n "$SUBSAMPLE" ]; then SUB_ARGS+=( --subsample-size "$SUBSAMPLE" ); fi
    if [ "$AUG" = "1" ]; then SUB_ARGS+=( --aug ); fi
    python dataset/build_maze_dataset.py --output-dir "$DATASET_DIR" "${SUB_ARGS[@]}"
  fi
fi

export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

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

echo "[INFO] Training maze model" >&2
_hrm_run_pretrain "${OVERRIDES[@]}" ${TRAIN_EXTRA:+"${TRAIN_EXTRA[@]}"} ${HYDRA_OVERRIDES:+"${HYDRA_OVERRIDES[@]}"}

echo "[INFO] Training complete" >&2
LATEST_CKPT=$(bash "$SCRIPT_DIR/eval_latest.sh" --print-path || true)
if [ -z "$LATEST_CKPT" ]; then
  echo "[WARN] No checkpoint found" >&2
  exit 0
fi

echo "[INFO] Latest checkpoint: $LATEST_CKPT" >&2
if [ "$SUMMARY" = "1" ]; then
  bash "$SCRIPT_DIR/eval_summary.sh" "$LATEST_CKPT" || echo "[WARN] eval_summary failed" >&2
fi
if [ "$BENCHMARK" = "1" ]; then
  bash "$SCRIPT_DIR/benchmark_eval.sh" "$LATEST_CKPT" || echo "[WARN] benchmark failed" >&2
fi

end_ts=$(date +%s); echo "MAZE_PIPELINE_DURATION_SEC=$(( end_ts - start_ts ))" >&2
