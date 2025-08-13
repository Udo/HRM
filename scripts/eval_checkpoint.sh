#!/usr/bin/env bash
set -euo pipefail

# Evaluate a checkpoint: pass CHECKPOINT=path/to/file or first arg
CHK=${CHECKPOINT:-${1:-}}
if [ -z "$CHK" ]; then
  echo "Usage: CHECKPOINT=path/to/step_x python scripts/eval_checkpoint.sh OR scripts/eval_checkpoint.sh path/to/ckpt" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv
python evaluate.py checkpoint="$CHK"
