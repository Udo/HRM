#!/usr/bin/env bash
set -euo pipefail
# Fetch ARC raw datasets (ARC-AGI, ConceptARC, ARC-AGI-2) into dataset/raw-data/ structure if empty.
# Tries lightweight public mirrors or instructs user if manual download required.
# This script purposefully avoids large downloads if data already present.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/dataset/raw-data"

arc1_dir="$RAW_DIR/ARC-AGI/data"
concept_dir="$RAW_DIR/ConceptARC/corpus"
arc2_dir="$RAW_DIR/ARC-AGI-2/data"

mkdir -p "$arc1_dir" "$concept_dir" "$arc2_dir"

echo "[INFO] Checking for existing ARC JSON files..."
found_any=0
for d in "$arc1_dir" "$concept_dir" "$arc2_dir"; do
  if find "$d" -type f -name '*.json' -maxdepth 2 | head -n 1 >/dev/null 2>&1; then
    echo "[INFO] Found existing JSON in $d (skipping fetch for this dir)"
    found_any=1
  fi
done

if [ $found_any -eq 1 ]; then
  echo "[INFO] At least one ARC dataset already present; nothing to fetch."
  exit 0
fi

echo "[INFO] No ARC JSON files detected; attempting environment-driven clone/fetch if URLs provided." 

clone_if_set() {
  local var_name="$1"; shift
  local target_dir="$1"; shift
  local url="${!var_name:-}"
  if [ -n "$url" ]; then
    if [ "$(find "$target_dir" -type f -name '*.json' -maxdepth 3 | head -n1)" ]; then
      echo "[INFO] $target_dir already populated; skipping clone for $var_name"
      return 0
    fi
    echo "[INFO] Cloning $var_name -> $url into $target_dir"
    mkdir -p "$target_dir"
    tmp_dir="$(mktemp -d)"
    if git clone --depth 1 "$url" "$tmp_dir"; then
      # Copy any json trees into target
      find "$tmp_dir" -type f -name '*.json' -maxdepth 5 -print0 | while IFS= read -r -d '' f; do
        rel="${f#$tmp_dir/}"
        dest_dir="$target_dir/$(dirname "$rel")"
        mkdir -p "$dest_dir"
        cp -p "$f" "$dest_dir/"
      done
      echo "[INFO] Copied JSON files from $url"
    else
      echo "[WARN] git clone failed for $url"
    fi
    rm -rf "$tmp_dir"
  fi
}

# Environment variables (optional): ARC_AGI_REPO_URL, CONCEPT_ARC_REPO_URL, ARC_AGI_2_REPO_URL
clone_if_set ARC_AGI_REPO_URL "$arc1_dir"
clone_if_set CONCEPT_ARC_REPO_URL "$concept_dir"
clone_if_set ARC_AGI_2_REPO_URL "$arc2_dir"

# Re-scan
found_any=0
for d in "$arc1_dir" "$concept_dir" "$arc2_dir"; do
  if find "$d" -type f -name '*.json' -maxdepth 2 | head -n 1 >/dev/null 2>&1; then
    found_any=1; break
  fi
done

if [ $found_any -eq 1 ]; then
  echo "[INFO] ARC raw data fetched successfully."
  exit 0
fi

cat <<'GUIDE'
[ERROR] Could not fetch ARC raw data automatically.
Provide one of:
  * Populate JSON files manually under dataset/raw-data/ARC-AGI/data and ConceptARC/corpus
  * Set ARC_AGI_REPO_URL / CONCEPT_ARC_REPO_URL / ARC_AGI_2_REPO_URL to git repos containing JSON tasks
  * Supply an archive: unzip arc_raw.zip -d dataset/raw-data/
  * Or run tiny fallback: python dataset/build_arc_tiny_smoke.py --output-dir data/arc-aug-1000
Then re-run build or pipeline.
GUIDE
exit 3
