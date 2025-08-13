#!/usr/bin/env bash
set -euo pipefail

# Multi-seed orchestrator for Maze (mirrors repro_arc_multi_seed.sh) with CSV aggregation.
# Usage:
#   ./scripts/repro_maze_multi_seed.sh [-- hydra.overrides]
# Env Vars:
#   SEEDS (default 0,1,2) PARALLEL (default 1) FAST (0|1) DATASET_DIR (default data/maze-30x30-hard-1k)
#   RESULT_ROOT (default outputs/<date>) EPOCHS LOSS_TYPE SUBSAMPLE AUG SUMMARY_ONLY SKIP_BUILD
#   OUTPUT_CSV (path) if set, aggregate summary appended after aggregation
#   SKIP_EXISTING (1 skip per-seed already done)

HYDRA_OVERRIDES=()
for a in "$@"; do if [ "$a" = "--" ]; then shift; HYDRA_OVERRIDES=("$@" ); break; fi; shift || true; done

: "${SEEDS:=0,1,2}"; : "${PARALLEL:=1}"; : "${DATASET_DIR:=data/maze-30x30-hard-1k}"; : "${FAST:=0}"; : "${LOSS_TYPE:=softmax_cross_entropy}"; : "${SUMMARY_ONLY:=0}"; : "${SKIP_BUILD:=1}"; : "${SKIP_EXISTING:=0}"; : "${OUTPUT_CSV:=}";

ts=$(date -u +%Y-%m-%dT%H-%M-%S); date_dir=$(date -u +%Y-%m-%d)
RESULT_ROOT=${RESULT_ROOT:-outputs/${date_dir}}
base_dir="${RESULT_ROOT}/maze_multi_seed_${ts}${EXTRA_NAME:+_${EXTRA_NAME}}"; mkdir -p "$base_dir"
IFS=',' read -r -a SEED_LIST <<<"$SEEDS"; total=${#SEED_LIST[@]}
echo "[INFO] Multi-seed Maze: seeds=[$SEEDS] total=$total parallel=$PARALLEL fast=$FAST" >&2

active=0; JOB_PIDS=(); JOB_SEEDS=()
launch_seed() {
  local seed="$1"; local out_json="$base_dir/seed_${seed}.json"
  if [ -f "$out_json" ] && [ "$SKIP_EXISTING" = 1 ]; then echo "[INFO] Seed $seed existing -> skip" >&2; return; fi
  (
    set -e
    export FAST SKIP_BUILD DATASET_DIR LOSS_TYPE SUMMARY_ONLY OUTPUT_JSON OUTPUT_CSV_APPEND
    OUTPUT_JSON="$out_json"; OUTPUT_CSV_APPEND=""  # per-seed JSON only
    ./scripts/repro_maze_result.sh ${EPOCHS:+EPOCHS=$EPOCHS} -- seed="$seed" ${HYDRA_OVERRIDES:+"${HYDRA_OVERRIDES[@]}"}
  ) &
  JOB_PIDS+=($!); JOB_SEEDS+=("$seed"); active=$((active+1))
}

poll_jobs() {
  local new_pids=() new_seeds=()
  for i in "${!JOB_PIDS[@]}"; do
    local pid=${JOB_PIDS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid"); new_seeds+=("${JOB_SEEDS[$i]}")
    else
      wait "$pid" || { echo "[ERROR] Seed ${JOB_SEEDS[$i]} failed" >&2; exit 3; }
      active=$((active-1))
    fi
  done
  JOB_PIDS=(${new_pids[@]:-}); JOB_SEEDS=(${new_seeds[@]:-})
}

for s in "${SEED_LIST[@]}"; do
  launch_seed "$s"
  if [ "$PARALLEL" -gt 1 ]; then
    while [ "$active" -ge "$PARALLEL" ]; do poll_jobs; sleep 0.5; done
  else
    wait ${JOB_PIDS[-1]} || { echo "[ERROR] Seed $s failed" >&2; exit 4; }
    active=$((active-1)); JOB_PIDS=(); JOB_SEEDS=()
  fi
done

if [ "$PARALLEL" -gt 1 ]; then while [ "$active" -gt 0 ]; do poll_jobs; sleep 0.5; done; fi

echo "[INFO] Aggregating" >&2
base_dir="$base_dir" python - <<'PY'
import json, os, glob, math
base=os.environ['base_dir']
paths=sorted(glob.glob(os.path.join(base,'seed_*.json')))
recs=[json.load(open(p)) for p in paths]
if not recs:
  print('{}'); raise SystemExit
numeric={k for r in recs for k,v in r.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}

def stat(vals):
  vals=[v for v in vals if isinstance(v,(int,float))];
  if not vals: return None
  import statistics as st
  mean=sum(vals)/len(vals)
  sd=st.pstdev(vals) if len(vals)>1 else 0.0
  return {'mean':mean,'stdev':sd,'n':len(vals)}

agg={'num_seeds':len(recs),'seeds':[r.get('hydra_overrides',[]) for r in recs]}
for k in sorted(numeric): agg[k]=stat([r.get(k) for r in recs])
out=json.dumps(agg,separators=(',',':'))
open(os.path.join(base,'aggregate_summary.json'),'w').write(out+'\n')
print(out)
PY

if [ -n "$OUTPUT_CSV" ]; then
  python - <<PY
import json, csv, os
base_dir=os.environ['base_dir']; csv_path=os.environ['OUTPUT_CSV']
agg=json.load(open(os.path.join(base_dir,'aggregate_summary.json')))
cols=['metric','mean','stdev','n']
rows=[]
for k,v in agg.items():
  if isinstance(v,dict) and {'mean','stdev','n'}<=v.keys():
    rows.append({'metric':k,'mean':v['mean'],'stdev':v['stdev'],'n':v['n']})
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
write_header=not os.path.isfile(csv_path)
with open(csv_path,'a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols)
  if write_header: w.writeheader()
  for r in rows: w.writerow(r)
print(f"[INFO] Aggregate CSV appended -> {csv_path}")
PY
fi

echo "[RESULT_DIR] $base_dir"
