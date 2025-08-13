#!/usr/bin/env bash
set -euo pipefail

# Evaluate the most recent checkpoint (by mtime) under checkpoints/.
# Usage:
#   ./scripts/eval_latest.sh [--print-path] [extra evaluate.py overrides]
# Options:
#   --print-path   Only print the resolved checkpoint path and exit 0.
# Environment:
#   HRM_DEVICE / HRM_DISABLE_WANDB etc respected via evaluate.py (indirectly if needed).

PRINT_PATH=0
if [ "${1:-}" = "--print-path" ]; then
  PRINT_PATH=1
  shift || true
fi

if [ ! -d checkpoints ]; then
  echo "[ERR] No checkpoints directory." >&2
  exit 2
fi

# Find newest step_* file
# Prefer new naming pattern model_step_*.pt; fallback to legacy step_*
LATEST_CKPT=$(find checkpoints -type f -name 'model_step_*.pt' -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n1 || true)
if [ -z "$LATEST_CKPT" ]; then
  LATEST_CKPT=$(find checkpoints -type f -name 'step_*' -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n1 || true)
fi
if [ -z "$LATEST_CKPT" ]; then
  echo "[ERR] No checkpoint files found." >&2
  exit 3
fi

if [ $PRINT_PATH -eq 1 ]; then
  echo "$LATEST_CKPT"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

echo "[INFO] Evaluating $LATEST_CKPT" >&2
python evaluate.py checkpoint="$LATEST_CKPT" "$@"
