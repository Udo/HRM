#!/usr/bin/env bash
set -euo pipefail
# Auto-pick newest Maze checkpoint and visualize one puzzle.
# Usage: ./scripts/viz_maze_latest.sh [extra visualize_maze_cli.py args]
# Env overrides:
#   CHECKPOINT_ROOT (default checkpoints)
#   DATA_DIR        (if not provided tries to infer from config or data/maze*)
#
# Examples:
#   ./scripts/viz_maze_latest.sh --puzzle-index 2 --auto 0.4
#   DATA_DIR=data/maze-30x30-hard-1k ./scripts/viz_maze_latest.sh

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}
DATA_DIR_ENV=${DATA_DIR:-}
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR_ENV=$2; shift 2;;
    --data-dir=*) DATA_DIR_ENV=${1#*=}; shift;;
    *) PASSTHRU+=("$1"); shift;;
  esac
done

# Collect checkpoints (portable)
CKPTS=()
while IFS= read -r line; do [[ -n "$line" ]] && CKPTS+=("$line"); done < <(python - <<'PY'
import os
root=os.environ.get('CHECKPOINT_ROOT','checkpoints')
for r,_,fs in os.walk(root):
  for f in fs:
    if f.startswith('model_step_') and f.endswith('.pt'):
      print(os.path.join(r,f))
PY
)
if [[ ${#CKPTS[@]} -eq 0 ]]; then echo "[ERROR] No checkpoints under $CHECKPOINT_ROOT" >&2; exit 1; fi

# Select newest ckpt preferring maze in path or config data_path containing 'maze'
get_mtime(){ if stat -f '%m' "$1" >/dev/null 2>&1; then stat -f '%m' "$1"; else stat -c '%Y' "$1"; fi; }
latest_any=""; t_any=0; latest_maze=""; t_maze=0
for ck in "${CKPTS[@]}"; do
  dir=$(dirname "$ck"); cfg="$dir/all_config.yaml"; [[ -f "$cfg" ]] || continue
  dp=$(grep -E '^data_path:' "$cfg" | head -n1 | sed 's/data_path:[ ]*//') || true
  mt=$(get_mtime "$ck") || mt=0
  if [[ "$dp" == *maze* || "$dir" == *maze* || "$dir" == *Maze* ]]; then
    if (( mt > t_maze )); then t_maze=$mt; latest_maze=$ck; fi
  fi
  if (( mt > t_any )); then t_any=$mt; latest_any=$ck; fi
done
sel_ckpt=${latest_maze:-$latest_any}
[[ -z "$sel_ckpt" ]] && { echo "[ERROR] Failed to pick checkpoint" >&2; exit 1; }
ckpt_dir=$(dirname "$sel_ckpt")
echo "[INFO] Selected checkpoint: $sel_ckpt" >&2

# Infer data dir if absent
if [[ -z "$DATA_DIR_ENV" ]]; then
  cfg="$ckpt_dir/all_config.yaml"
  if [[ -f "$cfg" ]]; then
    DATA_DIR_ENV=$(grep -E '^data_path:' "$cfg" | head -n1 | sed 's/data_path:[ ]*//') || true
    [[ -n "$DATA_DIR_ENV" ]] && echo "[INFO] Inferred data-dir from config: $DATA_DIR_ENV" >&2
  fi
fi
if [[ -z "$DATA_DIR_ENV" ]]; then
  candidate=$(ls -1d data/maze* 2>/dev/null | head -n1 || true)
  [[ -n "$candidate" ]] && { DATA_DIR_ENV=$candidate; echo "[INFO] Auto-selected dataset: $DATA_DIR_ENV" >&2; }
fi
[[ -z "$DATA_DIR_ENV" || ! -f "$DATA_DIR_ENV/test/dataset.json" ]] && { echo "[ERROR] Maze dataset directory not found (provide --data-dir)" >&2; exit 1; }

exec python scripts/visualize_maze_cli.py --checkpoint-dir "$ckpt_dir" --data-dir "$DATA_DIR_ENV" "${PASSTHRU[@]}"
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
      echo --gif "visualizations/maze_${base}_${ts}.gif"
    fi
  )
