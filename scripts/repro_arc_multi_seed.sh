#!/usr/bin/env bash
set -euo pipefail

# Automated multi-seed ARC reproduction / aggregation.
# Runs repro_arc_result.sh across a list of seeds, collecting each JSON result and producing
# an aggregated summary (mean/std) for key metrics.
#
# Usage:
#   ./scripts/repro_arc_multi_seed.sh [-- hydra.overrides]
#
# Examples:
#   FAST=1 SEEDS=0,1 EPOCHS=2 ./scripts/repro_arc_multi_seed.sh
#   SEEDS=0,1,2,3,4 PARALLEL=2 ./scripts/repro_arc_multi_seed.sh -- arch.loss.loss_type=softmax_cross_entropy
#
# Environment Variables:
#   SEEDS          Comma-separated integer seeds (default: 0,1,2)
#   PARALLEL       Max concurrent runs (default 1)
#   DATASET_DIR    Passed through (default data/arc-aug-1000)
#   RESULT_ROOT    Root for outputs (optional)
#   FAST           Propagate to repro_arc_result.sh (default 0)
#   EPOCHS         Override epochs
#   SKIP_BUILD     Skip dataset build inside child runs (default 1)
#   ALLOW_TINY_ARC Allow synthetic tiny dataset (default 0)
#   LOSS_TYPE      Loss override (default softmax_cross_entropy)
#   EXTRA_NAME     Tag appended to result dir name
#   SUMMARY_ONLY   If 1, only evaluate existing checkpoints for each seed
#   SKIP_EXISTING  If 1, skip seeds with existing JSON
#
# Output: seed_<SEED>.json + aggregate_summary.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HYDRA_OVERRIDES=()
for a in "$@"; do
  if [ "$a" = "--" ]; then
    shift
    HYDRA_OVERRIDES=("$@")
    break
  fi
  shift || true
done

: "${SEEDS:=0,1,2}"
: "${PARALLEL:=1}"
: "${DATASET_DIR:=data/arc-aug-1000}"
: "${FAST:=0}"
: "${SKIP_BUILD:=1}"
: "${ALLOW_TINY_ARC:=0}"
: "${LOSS_TYPE:=softmax_cross_entropy}"
: "${SUMMARY_ONLY:=0}"
: "${SKIP_EXISTING:=0}"

ts=$(date -u +%Y-%m-%dT%H-%M-%S)
date_dir=$(date -u +%Y-%m-%d)
RESULT_ROOT=${RESULT_ROOT:-outputs/${date_dir}}
base_dir="${RESULT_ROOT}/arc_multi_seed_${ts}${EXTRA_NAME:+_${EXTRA_NAME}}"
mkdir -p "$base_dir"

IFS=',' read -r -a SEED_LIST <<<"$SEEDS"
total=${#SEED_LIST[@]}
echo "[INFO] Multi-seed ARC: seeds=[$SEEDS] total=$total parallel=$PARALLEL fast=$FAST" >&2

active_jobs=0
JOB_SEEDS=()
JOB_PIDS=()

launch_seed() {
  local seed="$1"
  local out_json="$base_dir/seed_${seed}.json"
  if [ -f "$out_json" ] && [ "$SKIP_EXISTING" = 1 ]; then
    echo "[INFO] Seed $seed -> skip existing" >&2
    return 0
  fi
  echo "[INFO] Seed $seed start" >&2
  (
    set -e
    export FAST SKIP_BUILD DATASET_DIR ALLOW_TINY_ARC LOSS_TYPE SUMMARY_ONLY OUTPUT_JSON
    OUTPUT_JSON="$out_json"
    if [ ${#HYDRA_OVERRIDES[@]:-0} -gt 0 ]; then
      ./scripts/repro_arc_result.sh ${EPOCHS:+EPOCHS=$EPOCHS} -- seed="$seed" "${HYDRA_OVERRIDES[@]}"
    else
      ./scripts/repro_arc_result.sh ${EPOCHS:+EPOCHS=$EPOCHS} -- seed="$seed"
    fi
  ) &
  JOB_SEEDS+=("$seed")
  JOB_PIDS+=($!)
}

for seed in "${SEED_LIST[@]}"; do
  launch_seed "$seed"
  active_jobs=$((active_jobs+1))
  if [ "$PARALLEL" -gt 1 ]; then
    while [ "$active_jobs" -ge "$PARALLEL" ]; do
      # Poll PIDs
      new_pids=()
      new_seeds=()
      for idx in "${!JOB_PIDS[@]}"; do
        pid=${JOB_PIDS[$idx]}
        if kill -0 "$pid" 2>/dev/null; then
          new_pids+=("$pid")
          new_seeds+=("${JOB_SEEDS[$idx]}")
        else
          wait "$pid" || { echo "[ERROR] Seed ${JOB_SEEDS[$idx]} failed" >&2; exit 3; }
          active_jobs=$((active_jobs-1))
        fi
      done
      JOB_PIDS=(${new_pids[@]:-})
      JOB_SEEDS=(${new_seeds[@]:-})
      sleep 0.5
    done
  else
  # Sequential mode: last launched job must finish OK
  wait ${JOB_PIDS[-1]} || { echo "[ERROR] Seed $seed failed" >&2; exit 4; }
  fi
done

if [ "$PARALLEL" -gt 1 ]; then
  while [ "$active_jobs" -gt 0 ]; do
    new_pids=()
    new_seeds=()
    for idx in "${!JOB_PIDS[@]}"; do
      pid=${JOB_PIDS[$idx]}
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
        new_seeds+=("${JOB_SEEDS[$idx]}")
      else
        wait "$pid" || { echo "[ERROR] Seed ${JOB_SEEDS[$idx]} failed" >&2; exit 5; }
        active_jobs=$((active_jobs-1))
      fi
    done
    JOB_PIDS=(${new_pids[@]:-})
    JOB_SEEDS=(${new_seeds[@]:-})
    sleep 0.5
  done
fi

echo "[INFO] Aggregating" >&2
base_dir="$base_dir" python - <<'PY'
import json, os, glob, math
base_dir=os.environ['base_dir']
paths=sorted(glob.glob(os.path.join(base_dir,'seed_*.json')))
recs=[json.load(open(p)) for p in paths]
if not recs:
    print('{}')
    raise SystemExit
numeric_keys={k for r in recs for k,v in r.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}

def stat(lst):
    vals=[v for v in lst if isinstance(v,(int,float)) and not math.isnan(v)]
    if not vals: return None
    mean=sum(vals)/len(vals)
    sd=(sum((x-mean)**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else 0.0
    return {'mean':mean,'stdev':sd,'n':len(vals)}

agg={'num_seeds': len(recs), 'seeds': [r.get('hydra_overrides',[]) for r in recs]}
for k in sorted(numeric_keys):
    agg[k]=stat([r.get(k) for r in recs])
out=json.dumps(agg,separators=(',',':'))
open(os.path.join(base_dir,'aggregate_summary.json'),'w').write(out+'\n')
print(out)
PY

echo "[RESULT_DIR] $base_dir"
ls -1 "$base_dir"/seed_*.json >&2 || true
echo "[INFO] Aggregate summary saved to $base_dir/aggregate_summary.json" >&2
