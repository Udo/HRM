#!/usr/bin/env bash
set -euo pipefail

# Orchestrated pipeline: build (subset) Sudoku dataset, train model, evaluate & benchmark latest checkpoint.
# Usage:
#   ./scripts/run_sudoku_pipeline.sh [-- no-pass-through hydra overrides]
# Environment variables (override to customize):
#   DATASET_DIR           Path for dataset root (default: data/sudoku-extreme-1k-aug-10)
#   SUBSAMPLE             Train set subsample size (default: 1000)
#   AUG                   Number of augmentations per puzzle (train set) (default: 10)
#   MIN_DIFFICULTY        Minimum difficulty rating (optional)
#   SKIP_BUILD            If set to 1, skip dataset build step
#   EPOCHS                Total epochs (default: 2000 as in train_sudoku_small.sh)
#   EVAL_INTERVAL         Eval interval (default: 200)
#   GBS                   Global batch size (default: 128)
#   HIDDEN, HEADS, H_LAYERS, L_LAYERS, H_CYC, L_CYC, LR, PUZ_LR, WD, PUZ_WD  (mirrors train_sudoku_small.sh)
#   BENCHMARK             If 1 (default), run benchmark on latest checkpoint
#   SUMMARY               If 1 (default), run eval_summary.sh on latest checkpoint
#   CKPT_EVERY            If 1, enable per-eval checkpointing (checkpoint_every_eval=true)
#   EXTRA_BUILD_FLAGS     Extra flags passed to dataset/build_sudoku_dataset.py
# Pass-through Hydra overrides: appended after a literal -- (e.g. -- arch.expansion=3)
#
# Example (quick smoke run):
#   EPOCHS=20 EVAL_INTERVAL=5 SUBSAMPLE=200 AUG=2 ./scripts/run_sudoku_pipeline.sh
#   EPOCHS=20 EVAL_INTERVAL=5 SUBSAMPLE=200 AUG=2 ./scripts/run_sudoku_pipeline.sh -- arch.hidden_size=384 arch.num_heads=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

: "${DATASET_DIR:=data/sudoku-extreme-1k-aug-10}"
: "${SUBSAMPLE:=1000}"
: "${AUG:=10}"
: "${MIN_DIFFICULTY:=}"
: "${SKIP_BUILD:=0}"
: "${EPOCHS:=2000}"
: "${EVAL_INTERVAL:=200}"
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
: "${BENCHMARK:=1}"
: "${SUMMARY:=1}"
: "${CKPT_EVERY:=0}"
: "${EXTRA_BUILD_FLAGS:=}"

# Split out pass-through hydra overrides after -- (if present)
HYDRA_OVERRIDES=()
for i in "$@"; do
  if [ "$i" = "--" ]; then
    shift
    HYDRA_OVERRIDES=("$@")
    break
  fi
  shift || true
done

start_ts=$(date +%s)

############################
# 1. Build dataset (optional)
############################
if [ "$SKIP_BUILD" != "1" ]; then
  if [ -f "$DATASET_DIR/train/dataset.json" ] && [ -f "$DATASET_DIR/test/dataset.json" ]; then
    echo "[INFO] Dataset already exists at $DATASET_DIR (skipping build). Set SKIP_BUILD=0 and remove to rebuild." >&2
  else
    echo "[INFO] Building Sudoku dataset -> $DATASET_DIR (subsample=$SUBSAMPLE aug=$AUG)" >&2
    BUILD_CMD=(python dataset/build_sudoku_dataset.py \
      --output-dir "$DATASET_DIR" \
      --subsample-size "$SUBSAMPLE" \
      --num-aug "$AUG")
    if [ -n "$MIN_DIFFICULTY" ]; then
      BUILD_CMD+=(--min-difficulty "$MIN_DIFFICULTY")
    fi
    if [ -n "$EXTRA_BUILD_FLAGS" ]; then
      # shellcheck disable=SC2206
      EXTRA_FLAGS_ARRAY=($EXTRA_BUILD_FLAGS)
      BUILD_CMD+=("${EXTRA_FLAGS_ARRAY[@]}")
    fi
    "${BUILD_CMD[@]}"
    echo "[INFO] Dataset build complete." >&2
  fi
else
  echo "[INFO] SKIP_BUILD=1 -> Skipping dataset build." >&2
fi

############################
# 2. Train
############################
export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

echo "[INFO] Starting training (epochs=$EPOCHS eval_interval=$EVAL_INTERVAL hidden=$HIDDEN heads=$HEADS H_layers=$H_LAYERS L_layers=$L_LAYERS)" >&2
TRAIN_EXTRA=()
if [ "$CKPT_EVERY" = "1" ]; then
  TRAIN_EXTRA+=(checkpoint_every_eval=true)
fi
OVERRIDES=(
  data_path="$DATASET_DIR"
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
_hrm_run_pretrain "${OVERRIDES[@]}" ${TRAIN_EXTRA:+"${TRAIN_EXTRA[@]}"} ${HYDRA_OVERRIDES:+"${HYDRA_OVERRIDES[@]}"}

echo "[INFO] Training completed." >&2

############################
# 3. Locate latest checkpoint
############################
LATEST_CKPT=$(bash "$SCRIPT_DIR/eval_latest.sh" --print-path || true)
if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
  echo "[WARN] Could not locate latest checkpoint; skipping evaluation & benchmark." >&2
  exit 0
fi
echo "[INFO] Latest checkpoint: $LATEST_CKPT" >&2

############################
# 4. Evaluation summary (optional)
############################
if [ "$SUMMARY" = "1" ]; then
  echo "[INFO] Running eval_summary.sh" >&2
  bash "$SCRIPT_DIR/eval_summary.sh" "$LATEST_CKPT" || echo "[WARN] eval_summary failed" >&2
fi

############################
# 5. Benchmark (optional)
############################
if [ "$BENCHMARK" = "1" ]; then
  echo "[INFO] Running benchmark_eval.sh" >&2
  bash "$SCRIPT_DIR/benchmark_eval.sh" "$LATEST_CKPT" || echo "[WARN] benchmark failed" >&2
fi

end_ts=$(date +%s)
dur=$(( end_ts - start_ts ))
echo "PIPELINE_DURATION_SEC=$dur" >&2
echo "[INFO] Pipeline complete in ${dur}s" >&2
