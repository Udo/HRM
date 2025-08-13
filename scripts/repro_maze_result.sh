#!/usr/bin/env bash
set -euo pipefail

# Reproducible Maze result script (mirrors repro_arc_result.sh).
# Builds dataset (if needed), trains (unless SUMMARY_ONLY), evaluates latest checkpoint,
# and emits structured JSON + optional CSV row.
#
# Usage:
#   ./scripts/repro_maze_result.sh [-- hydra.overrides]
# Env Vars:
#   DATASET_DIR (default data/maze-30x30-hard-1k)
#   SKIP_BUILD (0|1)
#   SUBSAMPLE  (int) optional subsample size
#   AUG        (0|1) dihedral augmentation
#   FAST       (0|1) small arch + few epochs
#   SUMMARY_ONLY (0|1)
#   EPOCHS/EVAL_INTERVAL/GBS/HIDDEN/HEADS/H_LAYERS/L_LAYERS/H_CYC/L_CYC/LR/PUZ_LR/WD/PUZ_WD/WARMUP_STEPS
#   LOSS_TYPE (default softmax_cross_entropy)
#   OUTPUT_JSON (default outputs/<date>/maze_result_summary.json)
#   OUTPUT_CSV_APPEND (path) if set, append one CSV row (header auto if file empty)
#
# JSON keys similar to ARC: domain='Maze', plus metrics.

HYDRA_OVERRIDES=()
for a in "$@"; do
  if [ "$a" = "--" ]; then
    shift; HYDRA_OVERRIDES=("$@"); break
  fi; shift || true
done

: "${DATASET_DIR:=data/maze-30x30-hard-1k}"
: "${SKIP_BUILD:=0}"
: "${SUBSAMPLE:=}"
: "${AUG:=0}"
: "${FAST:=0}"

if [ "$FAST" = 1 ]; then
  : "${EPOCHS:=2}"; : "${EVAL_INTERVAL:=1}"; : "${GBS:=64}"; : "${HIDDEN:=128}"; : "${HEADS:=4}"; : "${H_LAYERS:=1}"; : "${L_LAYERS:=1}"; : "${H_CYC:=1}"; : "${L_CYC:=1}"; : "${LR:=1e-4}"; : "${PUZ_LR:=1e-3}"; : "${WD:=0.1}"; : "${PUZ_WD:=0.1}"; : "${WARMUP_STEPS:=10}";
else
  : "${EPOCHS:=800}"; : "${EVAL_INTERVAL:=80}"; : "${GBS:=128}"; : "${HIDDEN:=256}"; : "${HEADS:=8}"; : "${H_LAYERS:=2}"; : "${L_LAYERS:=2}"; : "${H_CYC:=2}"; : "${L_CYC:=2}"; : "${LR:=5e-4}"; : "${PUZ_LR:=5e-4}"; : "${WD:=0.1}"; : "${PUZ_WD:=0.1}"; : "${WARMUP_STEPS:=50}";
fi

: "${LOSS_TYPE:=softmax_cross_entropy}"
: "${OUTPUT_JSON:=outputs/$(date -u +%Y-%m-%d)/maze_result_summary.json}"
: "${OUTPUT_CSV_APPEND:=}"; : "${SUMMARY_ONLY:=0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/_common.sh"; _hrm_ensure_venv

export EPOCHS HIDDEN HEADS H_LAYERS L_LAYERS H_CYC L_CYC LOSS_TYPE FAST DATASET_DIR

mkdir -p "$(dirname "$OUTPUT_JSON")"

if [ "$SKIP_BUILD" != 1 ]; then
  if [ -f "$DATASET_DIR/train/dataset.json" ]; then
    echo "[INFO] Maze dataset exists -> skip build" >&2
  else
    echo "[INFO] Building maze dataset -> $DATASET_DIR" >&2
    SUB_ARGS=(); [ -n "$SUBSAMPLE" ] && SUB_ARGS+=(--subsample-size "$SUBSAMPLE"); [ "$AUG" = 1 ] && SUB_ARGS+=(--aug)
    python dataset/build_maze_dataset.py --output-dir "$DATASET_DIR" "${SUB_ARGS[@]}"
  fi
fi

export DISABLE_COMPILE=${DISABLE_COMPILE:-1}
export HRM_DISABLE_WANDB=${HRM_DISABLE_WANDB:-1}
export HRM_DISABLE_SPARSE_EMB_OPTIMIZER=${HRM_DISABLE_SPARSE_EMB_OPTIMIZER:-1}

TRAIN_OVERRIDES=(
  data_path="$DATASET_DIR" epochs="$EPOCHS" eval_interval="$EVAL_INTERVAL" global_batch_size="$GBS" lr_warmup_steps="$WARMUP_STEPS"
  arch.hidden_size="$HIDDEN" arch.num_heads="$HEADS" arch.H_layers="$H_LAYERS" arch.L_layers="$L_LAYERS" arch.H_cycles="$H_CYC" arch.L_cycles="$L_CYC"
  lr="$LR" puzzle_emb_lr="$PUZ_LR" weight_decay="$WD" puzzle_emb_weight_decay="$PUZ_WD" arch.loss.loss_type="$LOSS_TYPE"
)

if [ "$SUMMARY_ONLY" != 1 ]; then
  echo "[INFO] Training maze model" >&2
  _hrm_run_pretrain "${TRAIN_OVERRIDES[@]}" ${HYDRA_OVERRIDES:+"${HYDRA_OVERRIDES[@]}"}
else
  echo "[INFO] SUMMARY_ONLY=1 -> skip training" >&2
fi

LATEST_CKPT=$(bash "$SCRIPT_DIR/eval_latest.sh" --print-path || true)
if [ -z "$LATEST_CKPT" ]; then echo "[ERROR] No checkpoint found" >&2; exit 2; fi
export LATEST_CKPT
echo "[INFO] Using checkpoint: $LATEST_CKPT" >&2

EVAL_OUT=$(bash "$SCRIPT_DIR/eval_summary.sh" "$LATEST_CKPT" 2>/dev/null)
echo "$EVAL_OUT" >&2

METRICS_JSON=$(echo "$EVAL_OUT" | awk -F= '/^METRICS_JSON=/{print substr($0,index($0,$2))}')
export METRICS_JSON
DURATION=$(echo "$EVAL_OUT" | awk -F= '/^DURATION_SEC=/{print $2}')
EX_PER_S=$(echo "$EVAL_OUT" | awk -F= '/^EXAMPLES_PER_SEC=/{print $2}')
TOK_PER_S=$(echo "$EVAL_OUT" | awk -F= '/^TOKENS_PER_SEC=/{print $2}')
export DURATION EX_PER_S TOK_PER_S

PY_JSON=$(python - <<'PY'
import json, os, pathlib
from datetime import datetime, timezone
dataset_dir=os.environ['DATASET_DIR']
hidden=int(os.environ['HIDDEN']); heads=int(os.environ['HEADS'])
H_layers=int(os.environ['H_LAYERS']); L_layers=int(os.environ['L_LAYERS'])
H_cycles=int(os.environ['H_CYC']); L_cycles=int(os.environ['L_CYC'])
loss_type=os.environ['LOSS_TYPE']
fast=int(os.environ['FAST'])

def count_examples(split):
 f=pathlib.Path(dataset_dir)/split/'all__inputs.npy'
 return int(__import__('numpy').load(f, mmap_mode='r').shape[0]) if f.exists() else None

meta=None
for split in ('test','train'):
 mf=pathlib.Path(dataset_dir)/split/'dataset.json'
 if mf.exists(): meta=json.load(open(mf)); break
seq_len=meta.get('seq_len') if meta else None
vocab_size=meta.get('vocab_size') if meta else None

import importlib.util
spec=importlib.util.spec_from_file_location('pc','scripts/param_count.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
report=mod.compute_params(dict(hidden_size=hidden,num_heads=heads,H_layers=H_layers,L_layers=L_layers,H_cycles=H_cycles,L_cycles=L_cycles,expansion=4.0,puzzle_emb_ndim=hidden,pos_encodings='rope',halt_max_steps=16,halt_exploration_prob=0.1,vocab_size=vocab_size or 16,seq_len=seq_len or 900,num_puzzle_identifiers=1))

git_commit=''
try:
 import subprocess; git_commit=subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
except Exception: pass
out=dict(domain='Maze',git_commit=git_commit,timestamp_utc=datetime.now(timezone.utc).isoformat(),dataset_dir=dataset_dir,train_examples=count_examples('train'),test_examples=count_examples('test'),seq_len=seq_len,vocab_size=vocab_size,model_hidden_size=hidden,model_heads=heads,model_H_layers=H_layers,model_L_layers=L_layers,model_H_cycles=H_cycles,model_L_cycles=L_cycles,model_total_params=report['total_params'],loss_type=loss_type,fast_mode=bool(fast))
print(json.dumps(out,separators=(',',':')))
PY
)
export PY_JSON

FINAL_JSON=$(python - <<PY
import json, os, sys
base=json.loads(os.environ['PY_JSON'])
metrics=json.loads(os.environ.get('METRICS_JSON','{}') or '{}')
base.update({k:metrics.get(k) for k in ['accuracy','exact_accuracy','lm_loss','q_halt_accuracy','q_halt_loss','q_continue_loss','steps']})
base['avg_steps']=base.pop('steps', None)
base.update({'checkpoint_path':os.environ.get('LATEST_CKPT'),'eval_duration_sec':float(os.environ.get('DURATION') or 0) if os.environ.get('DURATION') else None,'examples_per_sec':float(os.environ.get('EX_PER_S') or 0) if os.environ.get('EX_PER_S') else None,'tokens_per_sec':float(os.environ.get('TOK_PER_S') or 0) if os.environ.get('TOK_PER_S') else None,'training_epochs_requested':int(os.environ.get('EPOCHS') or 0),'hydra_overrides':sys.argv[1:]})
print(json.dumps(base,separators=(',',':')))
PY
)

echo "$FINAL_JSON" | tee "$OUTPUT_JSON" >&2
echo "[INFO] Summary written to $OUTPUT_JSON" >&2

if [ -n "$OUTPUT_CSV_APPEND" ]; then
  mkdir -p "$(dirname "$OUTPUT_CSV_APPEND")"
  python - <<PY
import json, csv, os, sys
row=json.loads(os.environ['FINAL_JSON'])
csv_path=os.environ['OUTPUT_CSV_APPEND']
cols=["domain","git_commit","timestamp_utc","dataset_dir","train_examples","test_examples","seq_len","vocab_size","model_hidden_size","model_heads","model_H_layers","model_L_layers","model_H_cycles","model_L_cycles","model_total_params","loss_type","fast_mode","accuracy","exact_accuracy","lm_loss","q_halt_accuracy","q_halt_loss","q_continue_loss","avg_steps","eval_duration_sec","examples_per_sec","tokens_per_sec","training_epochs_requested"]
exists=os.path.isfile(csv_path)
with open(csv_path,'a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols)
  if not exists: w.writeheader()
  w.writerow({k:row.get(k) for k in cols})
PY
  echo "[INFO] Appended CSV row -> $OUTPUT_CSV_APPEND" >&2
fi
