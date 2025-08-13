#!/usr/bin/env bash
set -euo pipefail

# Reproducible ARC result script.
# Builds (if needed) the ARC dataset, trains the paper-sized HRM model (unless FAST mode),
# evaluates the latest checkpoint, and emits a structured JSON summary that is easy to
# compare with the paper metrics (token accuracy, exact grid accuracy, avg steps, params, etc.).
#
# Usage:
#   ./scripts/repro_arc_result.sh [-- hydra.overrides]
# Examples:
#   # Full (default arch + long training) – will take many hours; customize epochs first.
#   EPOCHS=100000 EVAL_INTERVAL=10000 ./scripts/repro_arc_result.sh
#
#   # Quick functional smoke (tiny fallback) producing JSON summary
#   FAST=1 ALLOW_TINY_ARC=1 ./scripts/repro_arc_result.sh
#
# Environment Variables:
#   DATASET_DIR       (default: data/arc-aug-1000)
#   SKIP_BUILD        (0|1) skip dataset build if already prepared (default 0)
#   ALLOW_TINY_ARC    (0|1) permit synthetic tiny ARC dataset fallback (default 0 to avoid accidental paper mismatch)
#   FAST              (0|1) if 1: use smaller dev architecture + few epochs, auto enables summary+benchmark (default 0)
#   EPOCHS            Training epochs (default 100000 for paper config; 2 in FAST mode)
#   EVAL_INTERVAL     Interval between evals (default 10000; 1 in FAST mode)
#   GBS               Global batch size (default 768 paper; 64 FAST)
#   HIDDEN, HEADS, H_LAYERS, L_LAYERS, H_CYC, L_CYC  (paper defaults 512,8,4,4,2,2; FAST uses 128,4,1,1,1,1)
#   LR                (default 1e-4)
#   PUZ_LR            (default 1e-2)
#   WD                (default 0.1)
#   PUZ_WD            (default 0.1)
#   WARMUP_STEPS      (default 2000 or 10 FAST)
#   LOSS_TYPE         (default softmax_cross_entropy; paper config file originally stablemax_cross_entropy)
#   OUTPUT_JSON       (path for JSON summary; default outputs/<date>/arc_result_summary.json)
#   SUMMARY_ONLY      (0|1) if 1 skip training & just evaluate + summarize latest ckpt (default 0)
#
# Outputs:
#   - JSON summary printed to stdout
#   - JSON summary written to $OUTPUT_JSON
#
# JSON Schema (keys):
#   domain, git_commit, timestamp_utc, dataset_dir, train_examples, test_examples,
#   seq_len, vocab_size, model_hidden_size, model_heads, model_H_layers, model_L_layers,
#   model_H_cycles, model_L_cycles, model_total_params, loss_type, accuracy, exact_accuracy,
#   lm_loss, q_halt_accuracy, q_halt_loss, q_continue_loss, avg_steps, checkpoint_path,
#   eval_duration_sec, examples_per_sec, tokens_per_sec, training_epochs_requested,
#   hydra_overrides, fast_mode, synthetic_tiny_dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
_hrm_ensure_venv

HYDRA_OVERRIDES=()
for a in "$@"; do
  if [ "$a" = "--" ]; then
    shift
    HYDRA_OVERRIDES=("$@")
    break
  fi
  shift || true
done

# Defaults
: "${DATASET_DIR:=data/arc-aug-1000}"
: "${SKIP_BUILD:=0}"
: "${ALLOW_TINY_ARC:=0}"  # safer default (paper uses real ARC data)
: "${FAST:=0}"

if [ "$FAST" = 1 ]; then
  : "${EPOCHS:=2}"
  : "${EVAL_INTERVAL:=1}"
  : "${GBS:=64}"
  : "${HIDDEN:=128}"
  : "${HEADS:=4}"
  : "${H_LAYERS:=1}"
  : "${L_LAYERS:=1}"
  : "${H_CYC:=1}"
  : "${L_CYC:=1}"
  : "${LR:=1e-4}"
  : "${PUZ_LR:=1e-3}"
  : "${WD:=0.1}"
  : "${PUZ_WD:=0.1}"
  : "${WARMUP_STEPS:=10}"
else
  : "${EPOCHS:=100000}"
  : "${EVAL_INTERVAL:=10000}"
  : "${GBS:=768}"
  : "${HIDDEN:=512}"
  : "${HEADS:=8}"
  : "${H_LAYERS:=4}"
  : "${L_LAYERS:=4}"
  : "${H_CYC:=2}"
  : "${L_CYC:=2}"
  : "${LR:=1e-4}"
  : "${PUZ_LR:=1e-2}"
  : "${WD:=0.1}"
  : "${PUZ_WD:=0.1}"
  : "${WARMUP_STEPS:=2000}"
fi

# Ensure EPOCHS exported so downstream python int() conversion succeeds even SUMMARY_ONLY
export EPOCHS HIDDEN HEADS H_LAYERS L_LAYERS H_CYC L_CYC LOSS_TYPE FAST ALLOW_TINY_ARC DATASET_DIR

: "${LOSS_TYPE:=softmax_cross_entropy}"
: "${OUTPUT_JSON:=outputs/$(date -u +%Y-%m-%d)/arc_result_summary.json}"
: "${SUMMARY_ONLY:=0}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

if [ "$SKIP_BUILD" != 1 ]; then
  if [ -f "$DATASET_DIR/train/dataset.json" ]; then
    echo "[INFO] ARC dataset already exists at $DATASET_DIR" >&2
  else
    if [ "$ALLOW_TINY_ARC" = 1 ]; then
      echo "[WARN] Building synthetic tiny ARC dataset (ALLOW_TINY_ARC=1). NOT comparable to paper numbers." >&2
      python dataset/build_arc_tiny_smoke.py --output-dir "$DATASET_DIR"
    else
      echo "[INFO] Building real ARC dataset (expect official raw data present)" >&2
      python dataset/build_arc_dataset.py --output-dir "$DATASET_DIR" || {
        echo "[ERROR] Failed to build ARC dataset. If you intend a smoke test rerun with ALLOW_TINY_ARC=1" >&2
        exit 2
      }
    fi
  fi
fi

export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

TRAIN_OVERRIDES=(
  data_path="$DATASET_DIR"
  epochs="$EPOCHS"
  eval_interval="$EVAL_INTERVAL"
  global_batch_size="$GBS"
  lr_warmup_steps="$WARMUP_STEPS"
  arch.hidden_size="$HIDDEN"
  arch.num_heads="$HEADS"
  arch.H_layers="$H_LAYERS"
  arch.L_layers="$L_LAYERS"
  arch.H_cycles="$H_CYC"
  arch.L_cycles="$L_CYC"
  lr="$LR"
  puzzle_emb_lr="$PUZ_LR"
  weight_decay="$WD"
  puzzle_emb_weight_decay="$PUZ_WD"
  arch.loss.loss_type="$LOSS_TYPE"
)

if [ "$SUMMARY_ONLY" != 1 ]; then
  echo "[INFO] Starting training (fast=$FAST)" >&2
  _hrm_run_pretrain "${TRAIN_OVERRIDES[@]}" ${HYDRA_OVERRIDES:+"${HYDRA_OVERRIDES[@]}"}
else
  echo "[INFO] SUMMARY_ONLY=1 -> skip training" >&2
fi

# Find latest checkpoint
LATEST_CKPT=$(bash "$SCRIPT_DIR/eval_latest.sh" --print-path || true)
if [ -z "$LATEST_CKPT" ]; then
  echo "[ERROR] No checkpoint found; cannot summarize." >&2
  exit 3
fi
export LATEST_CKPT

echo "[INFO] Using checkpoint: $LATEST_CKPT" >&2

# Evaluate & capture summary lines
EVAL_OUT=$(bash "$SCRIPT_DIR/eval_summary.sh" "$LATEST_CKPT" 2>/dev/null)
echo "$EVAL_OUT" >&2

# Parse key lines
METRICS_JSON=$(echo "$EVAL_OUT" | awk -F= '/^METRICS_JSON=/{print substr($0,index($0,$2))}')
export METRICS_JSON
ACCURACY=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("accuracy",""))' 2>/dev/null || true)
EXACT=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("exact_accuracy",""))' 2>/dev/null || true)
LM_LOSS=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("lm_loss",""))' 2>/dev/null || true)
HALT_ACC=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("q_halt_accuracy",""))' 2>/dev/null || true)
HALT_LOSS=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("q_halt_loss",""))' 2>/dev/null || true)
CONT_LOSS=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("q_continue_loss",""))' 2>/dev/null || true)
AVG_STEPS=$(echo "$METRICS_JSON" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("steps",""))' 2>/dev/null || true)

# Extract perf lines
DURATION=$(echo "$EVAL_OUT" | awk -F= '/^DURATION_SEC=/{print $2}')
EX_PER_S=$(echo "$EVAL_OUT" | awk -F= '/^EXAMPLES_PER_SEC=/{print $2}')
TOK_PER_S=$(echo "$EVAL_OUT" | awk -F= '/^TOKENS_PER_SEC=/{print $2}')
export DURATION EX_PER_S TOK_PER_S

# Dataset stats + param count via Python
PY_JSON=$(python - <<'PY'
import json, os, yaml, numpy as np, pathlib
from datetime import datetime, timezone
ckpt=os.environ.get('LATEST_CKPT')
dataset_dir=os.environ.get('DATASET_DIR')
hidden=int(os.environ.get('HIDDEN'))
heads=int(os.environ.get('HEADS'))
H_layers=int(os.environ.get('H_LAYERS'))
L_layers=int(os.environ.get('L_LAYERS'))
H_cycles=int(os.environ.get('H_CYC'))
L_cycles=int(os.environ.get('L_CYC'))
loss_type=os.environ.get('LOSS_TYPE')
fast=int(os.environ.get('FAST'))
synthetic=int(os.environ.get('ALLOW_TINY_ARC'))

def count_examples(split):
  f=pathlib.Path(dataset_dir)/split/'all__inputs.npy'
  if not f.exists():
    return None
  import numpy as np
  return int(np.load(f, mmap_mode='r').shape[0])

meta=None
for split in ('test','train'):
  mf=pathlib.Path(dataset_dir)/split/'dataset.json'
  if mf.exists():
    meta=json.load(open(mf))
    break

seq_len=meta.get('seq_len') if meta else None
vocab_size=meta.get('vocab_size') if meta else None

# Param count (reuse logic by importing script)
import importlib.util, types
spec=importlib.util.spec_from_file_location('param_count_mod', 'scripts/param_count.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
report=mod.compute_params(dict(hidden_size=hidden,num_heads=heads,H_layers=H_layers,L_layers=L_layers,H_cycles=H_cycles,L_cycles=L_cycles,expansion=4.0,puzzle_emb_ndim=hidden,pos_encodings='rope',halt_max_steps=16,halt_exploration_prob=0.1,vocab_size=vocab_size or 12,seq_len=seq_len or 900,num_puzzle_identifiers=1))

git_commit=''
try:
  import subprocess
  git_commit=subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
except Exception:
  pass

out=dict(
  domain='ARC',
  git_commit=git_commit,
  timestamp_utc=datetime.now(timezone.utc).isoformat(),
  dataset_dir=dataset_dir,
  train_examples=count_examples('train'),
  test_examples=count_examples('test'),
  seq_len=seq_len,
  vocab_size=vocab_size,
  model_hidden_size=hidden,
  model_heads=heads,
  model_H_layers=H_layers,
  model_L_layers=L_layers,
  model_H_cycles=H_cycles,
  model_L_cycles=L_cycles,
  model_total_params=report['total_params'],
  loss_type=loss_type,
  fast_mode=bool(fast),
  synthetic_tiny_dataset=bool(synthetic),
)
print(json.dumps(out,separators=(',',':')))
PY
)
export PY_JSON

# Merge JSON pieces
FINAL_JSON=$(python - <<PY
import json, os, sys
base=json.loads(os.environ['PY_JSON'])
metrics=json.loads(os.environ.get('METRICS_JSON','{}') or '{}')
base.update({
  'accuracy': metrics.get('accuracy'),
  'exact_accuracy': metrics.get('exact_accuracy'),
  'lm_loss': metrics.get('lm_loss'),
  'q_halt_accuracy': metrics.get('q_halt_accuracy'),
  'q_halt_loss': metrics.get('q_halt_loss'),
  'q_continue_loss': metrics.get('q_continue_loss'),
  'avg_steps': metrics.get('steps'),
  'checkpoint_path': os.environ.get('LATEST_CKPT'),
  'eval_duration_sec': float(os.environ.get('DURATION') or 0) if os.environ.get('DURATION') else None,
  'examples_per_sec': float(os.environ.get('EX_PER_S') or 0) if os.environ.get('EX_PER_S') else None,
  'tokens_per_sec': float(os.environ.get('TOK_PER_S') or 0) if os.environ.get('TOK_PER_S') else None,
  'training_epochs_requested': int(os.environ.get('EPOCHS') or 0),
  'hydra_overrides': sys.argv[1:],
})
print(json.dumps(base,separators=(',',':')))
PY
)

echo "$FINAL_JSON" | tee "$OUTPUT_JSON"

echo "[INFO] Summary written to $OUTPUT_JSON" >&2
