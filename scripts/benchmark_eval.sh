#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for Python benchmark script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

if [ $# -lt 1 ]; then
  echo "Usage: $0 <checkpoint_path> [extra hydra overrides]" >&2
  exit 1
fi

# Reconstruct checkpoint path with spaces (until .pt file exists)
if ! ckpt=$(_hrm_reconstruct_ckpt "$@"); then
  echo "[ERR] Could not resolve checkpoint path (remember to quote if it contains spaces)." >&2
  exit 3
fi
# Consume tokens corresponding to checkpoint (split path tokens and shift same count)
IFS=' ' read -r -a _ckpt_tokens <<<"$ckpt"
for ((i=1;i<=${#_ckpt_tokens[@]};i++)); do shift || true; done

python scripts/benchmark_eval.py --checkpoint "$ckpt" "$@"
