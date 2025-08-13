#!/usr/bin/env bash
set -euo pipefail

# ARC pipeline: (optionally) build ARC dataset, train model, evaluate summary, benchmark.
# Usage:
#   ./scripts/run_arc_pipeline.sh [-- hydra.overrides]
# Env Vars (override to customize):
#   DATASET_DIR      (default: data/arc-aug-1000)  # must match build_arc_dataset.py default
#   SKIP_BUILD       (0|1) default 0
#   AUTO_FETCH_ARC   (0|1) attempt to run scripts/fetch_arc_raw.sh if raw data missing (builder will retry)
#   EPOCHS           (default 800 like maze small)
#   EVAL_INTERVAL    (default 80)
#   GBS              (global batch size, default 128)
#   HIDDEN, HEADS, H_LAYERS, L_LAYERS, H_CYC, L_CYC
#   LR, PUZ_LR, WD, PUZ_WD, WARMUP_STEPS
#   CKPT_EVERY       (0|1) if 1 enable checkpoint_every_eval
#   SUMMARY          (0|1) run eval_summary (default 1)
#   BENCHMARK        (0|1) run benchmark (default 1)
# Pass additional hydra overrides after a literal --

# Allow ARC_DATASET_DIR (from fetch_all_datasets.sh) to alias DATASET_DIR if user set only that.
if [ -z "${DATASET_DIR:-}" ] && [ -n "${ARC_DATASET_DIR:-}" ]; then
  DATASET_DIR="$ARC_DATASET_DIR"
fi
: "${DATASET_DIR:=data/arc-aug-1000}"
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
    echo "[INFO] ARC dataset exists -> skip build" >&2
  else
    # Sanity check raw data presence (at least one non-empty source dir)
    RAW_OK=0
    for d in dataset/raw-data/ARC-AGI/data dataset/raw-data/ConceptARC/corpus dataset/raw-data/ARC-AGI-2/data; do
      if [ -d "$d" ] && find "$d" -type f -name '*.json' -maxdepth 2 | head -n 1 >/dev/null 2>&1; then
        RAW_OK=1; break
      fi
    done
    if [ "$RAW_OK" != 1 ]; then
      if [ "${ALLOW_TINY_ARC:-1}" = "1" ]; then
        echo "[WARN] No real ARC raw data found; building tiny synthetic ARC dataset for smoke test." >&2
        python dataset/build_arc_tiny_smoke.py --output-dir "$DATASET_DIR"
      else
        cat >&2 <<'EOF'
[ERROR] ARC raw data not found (expected json files under dataset/raw-data/ARC-AGI/data or ConceptARC/corpus).
        The repository currently has empty raw-data directories, so build_arc_dataset.py cannot proceed.
        Fix options:
          1. Place ARC dataset JSON files in the expected folders.
          2. Override source dirs, e.g.: python dataset/build_arc_dataset.py --dataset-dirs path1 path2 --output-dir "$DATASET_DIR"
          3. Skip build (SKIP_BUILD=1) if you already have a prepared dataset.
          4. Set ALLOW_TINY_ARC=1 to auto-generate a tiny synthetic dataset for smoke testing.
        After adding data re-run this script.
EOF
        exit 2
      fi
    else
      echo "[INFO] Building ARC dataset -> $DATASET_DIR" >&2
      python dataset/build_arc_dataset.py --output-dir "$DATASET_DIR"
    fi
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

echo "[INFO] Training ARC model" >&2
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

end_ts=$(date +%s); echo "ARC_PIPELINE_DURATION_SEC=$(( end_ts - start_ts ))" >&2
