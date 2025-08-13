#!/usr/bin/env bash
set -euo pipefail

# Auto-pick newest ARC checkpoint (config data_path containing 'arc') and visualize.
# Runs without arguments by inferring dataset from config or existing data/arc* directory.
#
# Examples:
#   ./scripts/viz_arc_latest.sh --puzzle-index 3 --auto 0.3
#   DATA_DIR=data/arc-aug-1000 ./scripts/viz_arc_latest.sh --no-color
#
# Pass any visualize_arc_cli.py flags; this script only resolves --checkpoint-dir and --data-dir.

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

DATA_DIR_ENV=${DATA_DIR:-}
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      [[ $# -lt 2 ]] && { echo "[ERROR] --data-dir needs a value" >&2; exit 1; }
      DATA_DIR_ENV=$2; shift 2;;
    --data-dir=*) DATA_DIR_ENV=${1#*=}; shift;;
    *) PASSTHRU+=("$1"); shift;;
  esac
done

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}

# Gather checkpoints robustly (handle spaces) using embedded Python (portable to old bash)
CKPTS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && CKPTS+=("$line")
done < <(python - <<'PY'
import os
root=os.environ.get('CHECKPOINT_ROOT','checkpoints')
for r,_,fs in os.walk(root):
    for f in fs:
        if f.startswith('model_step_') and f.endswith('.pt'):
            print(os.path.join(r,f))
PY
)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoint files found under $CHECKPOINT_ROOT" >&2
  exit 1
fi

get_mtime() { if stat -f '%m' "$1" >/dev/null 2>&1; then stat -f '%m' "$1"; else stat -c '%Y' "$1"; fi; }

best_arc=""; best_arc_time=0; best_any=""; best_any_time=0
for ck in "${CKPTS[@]}"; do
  dir="$(dirname "$ck")"; cfg="$dir/all_config.yaml"
  [[ -f "$cfg" ]] || continue
  dp=$(grep -E '^data_path:' "$cfg" | head -n1 | sed 's/data_path:[ ]*//') || true
  mt=$(get_mtime "$ck") || mt=0
  if [[ "$dp" == *arc* || "$dir" == *arc* || "$dir" == *ARC* ]]; then
    if (( mt > best_arc_time )); then best_arc_time=$mt; best_arc="$ck"; fi
  fi
  if (( mt > best_any_time )); then best_any_time=$mt; best_any="$ck"; fi
done

latest_ckpt="$best_arc"; [[ -z "$latest_ckpt" ]] && latest_ckpt="$best_any"
if [[ -z "$latest_ckpt" ]]; then
  echo "[ERROR] Failed to resolve latest ARC (or any) checkpoint." >&2; exit 1; fi

ckpt_dir="$(dirname "$latest_ckpt")"
echo "[INFO] Selected checkpoint: $latest_ckpt" >&2

# Infer data-dir if absent
if [[ -z "$DATA_DIR_ENV" ]]; then
  cfg_file="$ckpt_dir/all_config.yaml"
  if [[ -f "$cfg_file" ]]; then
    DATA_DIR_ENV=$(grep -E '^data_path:' "$cfg_file" | head -n1 | sed 's/data_path:[ ]*//') || true
    if [[ -n "$DATA_DIR_ENV" ]]; then
      echo "[INFO] Inferred data-dir from config: $DATA_DIR_ENV" >&2
    fi
  fi
fi
if [[ -z "$DATA_DIR_ENV" ]]; then
  candidate=$(ls -1d data/arc* 2>/dev/null | head -n1 || true)
  if [[ -n "$candidate" ]]; then DATA_DIR_ENV=$candidate; echo "[INFO] Auto-selected dataset: $DATA_DIR_ENV" >&2; fi
fi
if [[ -z "$DATA_DIR_ENV" || ! -f "$DATA_DIR_ENV/test/dataset.json" ]]; then
  echo "[ERROR] Could not determine ARC dataset directory (provide --data-dir)." >&2; exit 1; fi

exec python scripts/visualize_arc_cli.py --checkpoint-dir "$ckpt_dir" --data-dir "$DATA_DIR_ENV" "${PASSTHRU[@]}"
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
      echo --gif "visualizations/arc_${base}_${ts}.gif"
    fi
  )
