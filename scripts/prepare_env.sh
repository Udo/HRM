#!/usr/bin/env bash
set -euo pipefail

# Basic environment setup for HRM on macOS (MPS) or Linux (CUDA/CPU)
# Creates venv, installs dependencies, optional extras.

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment at $VENV_DIR" >&2
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
"$VENV_DIR"/bin/python -m pip install --upgrade pip
"$VENV_DIR"/bin/pip install -r requirements.txt
# numpy sometimes needed explicitly for tiny synthetic dataset helper (already in requirements but ensure)
"$VENV_DIR"/bin/pip install numpy || true

cat >&2 <<EOF
[INFO] Environment ready.
Scripts now auto-source scripts/_venv.sh (no manual activation needed).
If you want an interactive shell: source $VENV_DIR/bin/activate
EOF
