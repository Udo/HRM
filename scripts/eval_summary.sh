#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around Python implementation (scripts/eval_summary.py)
# Usage: ./scripts/eval_summary.sh [checkpoint] [-- extra hydra overrides]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

CKPT_ARG=()
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  # Case 1: Entire path provided as a single argument (quoted) and exists.
  if [ -f "$1" ]; then
    CKPT_ARG=(--checkpoint "$1")
    shift || true
  else
    # Case 2: Need to reconstruct joined tokens until a .pt file exists
    if ckpt_path=$(_hrm_reconstruct_ckpt "$@"); then
      # Determine how many tokens consumed by iteratively appending until match
      local_parts=()
      for tok in "$@"; do
        local_parts+=("$tok")
        test_path="${local_parts[*]}"
        if [ -f "$test_path" ] && [[ "$test_path" == *.pt ]]; then
          CKPT_ARG=(--checkpoint "$test_path")
          # Shift consumed tokens
          for ((i=1;i<=${#local_parts[@]};i++)); do shift || true; done
          break
        fi
      done
    fi
  fi
fi

if [ ${#CKPT_ARG[@]} -gt 0 ]; then
  # Use eval-safe quoting by passing as single string after --checkpoint
  python scripts/eval_summary.py "${CKPT_ARG[@]}" "$@"
else
  python scripts/eval_summary.py "$@"
fi
