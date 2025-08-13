#!/usr/bin/env bash
# Internal helper: ensure .venv exists and prepend its bin to PATH (without full activation).
set -euo pipefail
VENV_DIR=".venv"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[ERR] Missing venv; run scripts/prepare_env.sh" >&2
  exit 2
fi
case ":$PATH:" in
  *":$PWD/$VENV_DIR/bin:"*) ;;
  *) export PATH="$PWD/$VENV_DIR/bin:$PATH";;
 esac
export PYTHONNOUSERSITE=1
