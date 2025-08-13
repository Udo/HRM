#!/usr/bin/env bash
set -euo pipefail
# Unified helper to ensure all datasets (Sudoku, Maze, ARC, Tiny) are available.
# 1. Builds full Sudoku & Maze from HuggingFace if missing.
# 2. (Optional) Builds a smaller augmented Sudoku subset for faster iteration (disabled by default).
# 3. Attempts ARC raw fetch (git-based if env URLs provided) then ARC build (or tiny fallback if requested).
# 4. Optionally builds tiny synthetic dataset.
# Environment variables:
#   ARC_DATASET_DIR=...              (default data/arc-aug-1000)
#   ALLOW_TINY_ARC=1                 Allow synthetic ARC fallback if raw missing
#   AUTO_FETCH_ARC=1                 Auto-attempt scripts/fetch_arc_raw.sh when ARC dataset absent
#   ARC_AGI_REPO_URL / CONCEPT_ARC_REPO_URL / ARC_AGI_2_REPO_URL  Git URLs for raw ARC sources
#   BUILD_SUDOKU_1K=1                Also build a 1k augmented Sudoku subset (fast dev) (default off)
#   SUDOKU_1K_SUBSAMPLE=1000         Subsample size for the small Sudoku set
#   SUDOKU_1K_NUM_AUG=10             Number of augmentations per puzzle for small set
#   SUDOKU_1K_OUTPUT=data/sudoku-extreme-1k-aug-10  Output path for small set
# Usage:
#   ./scripts/fetch_all_datasets.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ARC_DATASET_DIR:=data/arc-aug-1000}"

cd "$ROOT_DIR"

log() { echo "[fetch-all] $*" >&2; }

# Sudoku (full)
if [ ! -f data/sudoku-extreme-full/train/dataset.json ]; then
  log "Building full Sudoku dataset"
  python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-full || log "Sudoku build failed"
else
  log "Sudoku dataset already present"
fi

# Optional Sudoku small subset (augmented) for quick experiments
if [ "${BUILD_SUDOKU_1K:-0}" = "1" ]; then
  : "${SUDOKU_1K_SUBSAMPLE:=1000}"
  : "${SUDOKU_1K_NUM_AUG:=10}"
  : "${SUDOKU_1K_OUTPUT:=data/sudoku-extreme-1k-aug-10}"
  if [ ! -f "$SUDOKU_1K_OUTPUT/train/dataset.json" ]; then
    log "Building small Sudoku subset -> $SUDOKU_1K_OUTPUT (subsample=$SUDOKU_1K_SUBSAMPLE aug=$SUDOKU_1K_NUM_AUG)"
    python dataset/build_sudoku_dataset.py --output-dir "$SUDOKU_1K_OUTPUT" --subsample-size "$SUDOKU_1K_SUBSAMPLE" --num-aug "$SUDOKU_1K_NUM_AUG" || log "Sudoku 1k build failed"
  else
    log "Sudoku 1k subset already present at $SUDOKU_1K_OUTPUT"
  fi
fi

# Maze
if [ ! -f data/maze-30x30-hard-1k/train/dataset.json ]; then
  log "Building Maze dataset"
  python dataset/build_maze_dataset.py --output-dir data/maze-30x30-hard-1k || log "Maze build failed"
else
  log "Maze dataset already present"
fi

# ARC
if [ ! -f "$ARC_DATASET_DIR/train/dataset.json" ]; then
  if [ "${AUTO_FETCH_ARC:-0}" = "1" ]; then
    log "Attempting ARC raw fetch (AUTO_FETCH_ARC=1)"
    AUTO_FETCH_ARC=1 python dataset/build_arc_dataset.py --output-dir "$ARC_DATASET_DIR" || true
  fi
  if [ ! -f "$ARC_DATASET_DIR/train/dataset.json" ]; then
    if [ "${ALLOW_TINY_ARC:-0}" = "1" ]; then
      log "Building tiny synthetic ARC dataset (fallback)"
      python dataset/build_arc_tiny_smoke.py --output-dir "$ARC_DATASET_DIR"
    else
      log "ARC dataset not built (raw missing and no tiny fallback). Set ALLOW_TINY_ARC=1 to allow synthetic."
    fi
  fi
else
  log "ARC dataset already present"
fi

# Tiny generic (optional)
if [ ! -f data/tiny/train/dataset.json ]; then
  log "Building tiny generic dataset"
  python dataset/build_tiny_dataset.py --output-dir data/tiny || true
else
  log "Tiny dataset already present"
fi

log "Done."
