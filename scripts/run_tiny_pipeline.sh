#!/usr/bin/env bash
set -euo pipefail

# Orchestrated tiny pipeline: build tiny synthetic dataset (if missing), train a small model,
# then run evaluation summary and (optional) benchmark on the latest checkpoint.
#
# Usage:
#   ./scripts/run_tiny_pipeline.sh [-- hydra.overrides=values]
#
# Environment Variables (override inline like VAR=val ./scripts/run_tiny_pipeline.sh ...):
#   OUTPUT_DIR        Dataset + run root (default: data/tiny)
#   TRAIN_SIZE        Tiny train examples (default: 16)
#   TEST_SIZE         Tiny test examples  (default: 8)
#   EPOCHS            Training epochs (default: 50)
#   EVAL_INTERVAL     Eval interval (default: 10)
#   GBS               Global batch size (default: 16)
#   HIDDEN, HEADS     Model width & heads (defaults: 128, 4)
#   H_LAYERS, L_LAYERS  High/Low transformer layer counts (default: 1,1)
#   H_CYC, L_CYC      Cycles (default: 1,1)
#   LR, PUZ_LR        Learning rates (default: 1e-3, 1e-3)
#   CKPT_EVERY        1 => save checkpoint every eval (default: 0)
#   SUMMARY           1 => run eval_summary (default: 1)
#   BENCHMARK         1 => run benchmark_eval (default: 1)
#   DISABLE_COMPILE   1 disables torch.compile (default: 1)
#   HRM_DISABLE_WANDB 1 disables wandb logging (default: 1)
#   HRM_DISABLE_SPARSE_EMB_OPTIMIZER 1 disables custom sparse optimizer (default: 1)
#
# Pass-through Hydra overrides: after a literal --, e.g.
#   ./scripts/run_tiny_pipeline.sh -- arch.expansion=3 arch.loss.loss_type=softmax_cross_entropy

# Defaults
: "${OUTPUT_DIR:=data/tiny}"
: "${TRAIN_SIZE:=16}"
: "${TEST_SIZE:=8}"
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
: "${CKPT_EVERY:=0}"
: "${SUMMARY:=1}"
: "${BENCHMARK:=1}"

# Backwards compat: absorb leading VAR=VAL args (uppercase env style)
while [ $# -gt 0 ]; do
  case "$1" in
    [A-Z]*=*) export "$1"; shift;;
    --) break;;
    *) break;;
  esac
done

# Split hydra overrides after -- (if present)
HYDRA_OVERRIDES=()
for arg in "$@"; do
  if [ "$arg" = "--" ]; then
    shift
    HYDRA_OVERRIDES=("$@")
    break
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

# Export feature toggles (respect user overrides if already set)
export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

start_ts=$(date +%s)

########################################
# 1. Build tiny dataset (idempotent)
########################################
if [ ! -f "$OUTPUT_DIR/train/dataset.json" ]; then
  echo "[INFO] Building tiny dataset -> $OUTPUT_DIR (train=$TRAIN_SIZE test=$TEST_SIZE)" >&2
  python dataset/build_tiny_dataset.py --output-dir "$OUTPUT_DIR" --train-size "$TRAIN_SIZE" --test-size "$TEST_SIZE" || {
    echo "[ERR] Tiny dataset build failed" >&2; exit 3; }
else
  echo "[INFO] Tiny dataset already present at $OUTPUT_DIR (skip build)" >&2
fi

########################################
# 2. Train
########################################
EXTRA_ARGS=()
if [ "$CKPT_EVERY" = "1" ]; then
  EXTRA_ARGS+=(checkpoint_every_eval=true)
fi

echo "[INFO] Training tiny model epochs=$EPOCHS eval_interval=$EVAL_INTERVAL hidden=$HIDDEN heads=$HEADS" >&2
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
  arch.loss.loss_type=stablemax_cross_entropy
)

_hrm_run_pretrain "${OVERRIDES[@]}" ${EXTRA_ARGS:+"${EXTRA_ARGS[@]}"} ${HYDRA_OVERRIDES:+"${HYDRA_OVERRIDES[@]}"}
echo "[INFO] Training finished" >&2

########################################
# 3. Resolve latest checkpoint
########################################
LATEST_CKPT=$(bash "$SCRIPT_DIR/eval_latest.sh" --print-path || true)
if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
  echo "[WARN] No checkpoint found after training (skipping eval & benchmark)" >&2
  end_ts=$(date +%s); echo "TINY_PIPELINE_DURATION_SEC=$(( end_ts - start_ts ))"; exit 0
fi
echo "[INFO] Latest checkpoint: $LATEST_CKPT" >&2

########################################
# 4. Evaluation summary (optional)
########################################
if [ "$SUMMARY" = "1" ]; then
  echo "[INFO] Running eval_summary.sh" >&2
  # Quote path; wrapper can reconstruct but we pass as single arg to avoid split issues
  bash "$SCRIPT_DIR/eval_summary.sh" "$LATEST_CKPT" || echo "[WARN] eval_summary failed" >&2
fi

########################################
# 5. Benchmark (optional)
########################################
if [ "$BENCHMARK" = "1" ]; then
  echo "[INFO] Running benchmark_eval.sh" >&2
  bash "$SCRIPT_DIR/benchmark_eval.sh" "$LATEST_CKPT" || echo "[WARN] benchmark failed" >&2
fi

end_ts=$(date +%s)
 dur=$(( end_ts - start_ts ))
 echo "TINY_PIPELINE_DURATION_SEC=$dur"
 echo "[INFO] Tiny pipeline complete in ${dur}s" >&2
