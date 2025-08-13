#!/usr/bin/env bash
set -euo pipefail

# Auto-pick the most recent Sudoku checkpoint (latest modified model_step_*.pt)
# and invoke the interactive / GIF visualization.
#
# Usage examples:
#   ./scripts/viz_sudoku_latest.sh --data-dir data/sudoku-extreme-1k-aug-10 --puzzle-index 5
#   ./scripts/viz_sudoku_latest.sh --data-dir data/sudoku-extreme-1k-aug-10 \
#       --gif solve.gif --gif-delay 0.5 --auto 0.5
#   DATA_DIR=data/sudoku-extreme-1k-aug-10 ./scripts/viz_sudoku_latest.sh --auto 0.4
#
# Pass any visualize_sudoku_cli.py flags directly; this script only resolves the --checkpoint-dir.
# If you want to override search root: CHECKPOINT_ROOT=checkpoints ./scripts/viz_sudoku_latest.sh ...
# Restrict to a project name pattern (e.g. 'Sudoku*ACT-torch'): PROJECT_GLOB='checkpoints/Sudoku*ACT-torch/*' ./scripts/viz_sudoku_latest.sh ...
#
# Environment variables:
#   DATA_DIR         (required unless --data-dir provided)
#   CHECKPOINT_ROOT  Root directory to search (default: checkpoints)
#   PROJECT_GLOB     Glob (relative or absolute) overriding default recursive search for model_step_*.pt
#
# Implementation detail: chooses newest by file modification time (ls -1t). If tie, first returned.

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

# Collect passthrough flags (we'll resolve data-dir later if missing)
DATA_DIR_ENV=${DATA_DIR:-}
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      [[ $# -lt 2 ]] && { echo "[ERROR] --data-dir needs a value" >&2; exit 1; }
      DATA_DIR_ENV=$2; shift 2;;
    --data-dir=*)
      DATA_DIR_ENV=${1#*=}; shift;;
    *) PASSTHRU+=("$1"); shift;;
  esac
done

if [[ ${#PASSTHRU[@]} -gt 0 && -z "${PASSTHRU[0]}" ]]; then
  PASSTHRU=("${PASSTHRU[@]:1}")
fi

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}
PROJECT_GLOB=${PROJECT_GLOB:-}

SEARCH_DESC=""
CKPT_FILES=()
if [[ -n "$PROJECT_GLOB" ]]; then
  SEARCH_DESC="$PROJECT_GLOB"
  while IFS= read -r -d '' f; do CKPT_FILES+=("$f"); done < <(find $PROJECT_GLOB -type f -name 'model_step_*.pt' -print0 2>/dev/null || true)
else
  SEARCH_DESC="$CHECKPOINT_ROOT (recursive)"
  while IFS= read -r -d '' f; do CKPT_FILES+=("$f"); done < <(find "$CHECKPOINT_ROOT" -type f -name 'model_step_*.pt' -print0 2>/dev/null || true)
fi

if [[ ${#CKPT_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoint files (model_step_*.pt) found under: $SEARCH_DESC" >&2
  echo "Hint: Train a Sudoku model first (e.g. scripts/train_sudoku_small.sh)" >&2
  exit 1
fi

# Sort by modification time descending using stat (portable-ish for macOS & Linux)
# macOS stat: stat -f '%m %N' file ; Linux: stat -c '%Y %n'
get_mtime() { if stat -f '%m' "$1" >/dev/null 2>&1; then stat -f '%m' "$1"; else stat -c '%Y' "$1"; fi; }

best_ckpt=""
best_time=0
fallback_ckpt=""
fallback_time=0
for f in "${CKPT_FILES[@]}"; do
  [[ -f "$f" ]] || continue
  dir=$(dirname "$f")
  cfg="$dir/all_config.yaml"
  data_ok=0
  if [[ -f "$cfg" ]]; then
    dp=$(grep -E '^data_path:' "$cfg" | head -n1 | sed 's/data_path:[ ]*//') || true
    if [[ -n "$dp" && "$dp" == *sudoku* ]]; then
      data_ok=1
    fi
  fi
  # Require sudoku; skip non-sudoku entirely
  (( data_ok == 0 )) && continue
  mt=$(get_mtime "$f") || mt=0
  if (( mt > best_time )); then
    best_time=$mt; best_ckpt="$f"
  fi
done

latest_ckpt="$best_ckpt"
if [[ -z "$latest_ckpt" ]]; then
  echo "[ERROR] No Sudoku checkpoints found (looked for data_path containing 'sudoku')." >&2
  exit 1
fi

ckpt_dir="$(dirname "$latest_ckpt")"
echo "[INFO] Selected latest checkpoint: $latest_ckpt" >&2
echo "[INFO] Run directory: $ckpt_dir" >&2

# Resolve data dir if not provided
if [[ -z "$DATA_DIR_ENV" ]]; then
  cfg_file="$ckpt_dir/all_config.yaml"
  if [[ -f "$cfg_file" ]]; then
    DATA_DIR_ENV=$(python - <<'PY'
import sys, yaml, json, os
cfg_path=sys.argv[1]
try:
    with open(cfg_path,'r') as f: cfg=yaml.safe_load(f)
    dp=cfg.get('data_path')
    if dp:
        print(dp)
except Exception as e:
    pass
PY
"$cfg_file")
    if [[ -n "$DATA_DIR_ENV" ]]; then
      echo "[INFO] Inferred data-dir from checkpoint config: $DATA_DIR_ENV" >&2
    fi
  fi
fi

# If still empty, search for a sudoku dataset (seq_len 81) heuristically
if [[ -z "$DATA_DIR_ENV" ]]; then
  candidate=$(ls -1d data/sudoku* 2>/dev/null | head -n1 || true)
  if [[ -n "$candidate" && -f "$candidate/test/dataset.json" ]]; then
    DATA_DIR_ENV=$candidate
    echo "[INFO] Auto-selected dataset directory: $DATA_DIR_ENV" >&2
  fi
fi

if [[ -z "$DATA_DIR_ENV" || ! -f "$DATA_DIR_ENV/test/dataset.json" ]]; then
  echo "[ERROR] Could not determine Sudoku dataset directory (no --data-dir provided, inference failed)." >&2
  echo "Hint: Provide --data-dir or set DATA_DIR env." >&2
  exit 1
fi

exec python scripts/visualize_sudoku_cli.py \
  --checkpoint-dir "$ckpt_dir" \
  --data-dir "$DATA_DIR_ENV" \
  $(
    wants_gif=1
    for a in "${PASSTHRU[@]}"; do
      case "$a" in
        --gif|--gif=*|--no-gif) wants_gif=0;;
      esac
    done
    if (( wants_gif )); then
      mkdir -p visualizations
      ts=$(date +%Y%m%d_%H%M%S)
      base=$(basename "$ckpt_dir" | tr ' ' '_')
      echo --gif "visualizations/sudoku_${base}_${ts}.gif"
    fi
  ) \
  "${PASSTHRU[@]}"
