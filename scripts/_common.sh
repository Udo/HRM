#!/usr/bin/env bash
# Shared helper functions to keep scripts DRY.
# Intended to be sourced by other scripts in this directory.
set -euo pipefail

# Determine repository root-relative script directory (where this file lives)
_hrm_script_dir() { cd "$(dirname "${BASH_SOURCE[0]}")" && pwd; }

# Source lightweight venv path injector
_hrm_ensure_venv() {
  local d shell_dir
  shell_dir=$(_hrm_script_dir)
  # shellcheck disable=SC1091
  source "$shell_dir/_venv.sh"
}

# Reconstruct a possibly spaced checkpoint path from arguments.
# Scans tokens until finding an existing *.pt file; outputs the path on stdout.
_hrm_reconstruct_ckpt() {
  local original=("$@")
  local build="" consumed=0 tok
  for tok in "${original[@]}"; do
    if [ -z "$build" ]; then build="$tok"; else build+=" $tok"; fi
    consumed=$((consumed+1))
    if [[ "$tok" == *.pt ]] && [ -f "$build" ]; then
      printf '%s\n' "$build"
      return 0
    fi
  done
  return 1
}

# Wrapper to call pretrain with common parameters. Accepts key=value pairs.
# Usage: _hrm_run_pretrain data_path=... epochs=... (other hydra overrides...)
_hrm_run_pretrain() {
  _hrm_ensure_venv
  python pretrain.py "$@"
}

export -f _hrm_script_dir _hrm_ensure_venv _hrm_reconstruct_ckpt _hrm_run_pretrain
