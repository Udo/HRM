#!/usr/bin/env bash
set -euo pipefail

# cleanup_artifacts.sh
# Safely prune training artifacts (checkpoints, generated GIFs, visualization dirs, and empty run dirs).
# Defaults to DRY RUN (prints what would be removed). Use --apply to actually delete.
#
# Targets (enabled automatically unless filtered by flags):
#   1. Checkpoint files:        checkpoints/**/model_step_*.pt
#   2. Visualization GIFs:      visualizations/*.gif (and root-level *.gif if --include-root-gif)
#   3. Empty run dirs:          checkpoint subdirs left empty after checkpoint removal (if --prune-empty-dirs)
#   4. Temp visualizer dumps:   arc_viz_* maze_viz_* sudoku_viz_* under visualizations/ (already GIFs)
#
# Safety:
#   - Dry run unless --apply specified.
#   - Can scope to only certain artifact classes via flags.
#   - Protect patterns (comma-separated) via PROTECT (env var or --protect) → globs to skip.
#
# Usage examples:
#   ./scripts/cleanup_artifacts.sh                 # Dry run summary
#   ./scripts/cleanup_artifacts.sh --checkpoints   # Dry run only checkpoints
#   ./scripts/cleanup_artifacts.sh --checkpoints --apply
#   PROTECT='*important-run*' ./scripts/cleanup_artifacts.sh --apply
#   ./scripts/cleanup_artifacts.sh --visuals --include-root-gif --apply
#
# Exit codes:
#   0 success (even if nothing to do)
#   2 invalid usage

set +u; HELP_REQ=$([ "$1" = "-h" ] 2>/dev/null || [ "$1" = "--help" ] 2>/dev/null && echo 1 || echo 0); set -u
if [[ $HELP_REQ == 1 ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

APPLY=0
DO_CHECKPOINTS=0
DO_VISUALS=0
DO_EMPTY=0
INCLUDE_ROOT_GIF=0
PROTECT_GLOBS="${PROTECT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift;;
    --checkpoints) DO_CHECKPOINTS=1; shift;;
    --visuals|--gifs) DO_VISUALS=1; shift;;
    --prune-empty-dirs) DO_EMPTY=1; shift;;
    --include-root-gif) INCLUDE_ROOT_GIF=1; shift;;
    --protect) [[ $# -lt 2 ]] && { echo "[ERROR] --protect needs value" >&2; exit 2; }; PROTECT_GLOBS=$2; shift 2;;
    --protect=*) PROTECT_GLOBS=${1#*=}; shift;;
    --all) DO_CHECKPOINTS=1; DO_VISUALS=1; DO_EMPTY=1; shift;;
    *) echo "[ERROR] Unknown arg: $1" >&2; exit 2;;
  esac
done

# If no specific section requested, enable checkpoints + visuals by default.
if (( DO_CHECKPOINTS==0 && DO_VISUALS==0 && DO_EMPTY==0 )); then
  DO_CHECKPOINTS=1; DO_VISUALS=1; DO_EMPTY=1
fi

if [[ -n "$PROTECT_GLOBS" ]]; then
  IFS=',' read -r -a PROTECT_LIST <<< "$PROTECT_GLOBS"
else
  PROTECT_LIST=()
fi

should_protect() {
  local path="$1"
  local plist=("${PROTECT_LIST[@]:-}")
  for pat in "${plist[@]}"; do
    [[ -z "${pat:-}" ]] && continue
    if [[ $path == $pat ]]; then return 0; fi
  done
  return 1
}

RM_LIST=()

if (( DO_CHECKPOINTS )); then
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if should_protect "$f"; then
      echo "[SKIP] Protected checkpoint: $f" >&2
      continue
    fi
    RM_LIST+=("$f")
  done < <(find checkpoints -type f -name 'model_step_*.pt' 2>/dev/null || true)
fi

if (( DO_VISUALS )); then
  # Visualizations directory
  if [[ -d visualizations ]]; then
    while IFS= read -r g; do
      [[ -z "$g" ]] && continue
      if should_protect "$g"; then
        echo "[SKIP] Protected gif: $g" >&2; continue
      fi
      RM_LIST+=("$g")
    done < <(find visualizations -maxdepth 1 -type f -name '*.gif' 2>/dev/null || true)
  fi
  if (( INCLUDE_ROOT_GIF )); then
    while IFS= read -r g; do
      [[ -z "$g" ]] && continue
      if should_protect "$g"; then echo "[SKIP] Protected root gif: $g" >&2; continue; fi
      RM_LIST+=("$g")
    done < <(find . -maxdepth 1 -type f -name '*.gif' 2>/dev/null || true)
  fi
fi

# Deduplicate
COUNT=${#RM_LIST[@]:-0}
if (( COUNT > 0 )); then
  # Deduplicate without mapfile for broader shell portability.
  DEDUP=$(printf '%s\n' "${RM_LIST[@]}" | awk 'NF' | sort -u)
  RM_LIST=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    RM_LIST+=("$line")
  done <<< "$DEDUP"
fi

echo "[INFO] Cleanup plan (dry-run=$((1-APPLY))): ${#RM_LIST[@]:-0} file(s)"
if (( ${#RM_LIST[@]:-0} > 0 )); then
  for f in "${RM_LIST[@]}"; do
    echo "  rm $f"
  done
fi

if (( APPLY )) && (( ${#RM_LIST[@]:-0} > 0 )); then
  for f in "${RM_LIST[@]}"; do
    rm -f -- "$f" || echo "[WARN] Failed to remove $f" >&2
  done
fi

if (( DO_EMPTY )); then
  # After deletions (or prospective), list empty checkpoint run dirs
  EMPTY_DIRS=()
  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    # skip protected
    if should_protect "$d"; then continue; fi
    # Count remaining non-hidden entries
    local_count=$(find "$d" -mindepth 1 -maxdepth 1 2>/dev/null | head -n1 || true)
    if [[ -z "$local_count" ]]; then
      EMPTY_DIRS+=("$d")
    fi
  done < <(find checkpoints -type d -mindepth 1 -maxdepth 4 2>/dev/null || true)

  if (( ${#EMPTY_DIRS[@]} > 0 )); then
    echo "[INFO] Empty run directories: ${#EMPTY_DIRS[@]}"
    for d in "${EMPTY_DIRS[@]}"; do echo "  rmdir $d"; done
    if (( APPLY )); then
      for d in "${EMPTY_DIRS[@]}"; do rmdir "$d" 2>/dev/null || true; done
    fi
  fi
fi

if (( APPLY )); then
  echo "[INFO] Cleanup complete."
else
  echo "[INFO] Dry run only. Re-run with --apply to delete."
fi
