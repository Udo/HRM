#!/usr/bin/env python
"""Evaluation summary utility.

Usage:
  python scripts/eval_summary.py [--checkpoint CKPT] [--print-json] [--] [extra hydra overrides for evaluate]

If --checkpoint omitted, the latest model_step_*.pt (or legacy step_*) is auto-detected.
Outputs key=value lines (same format the previous bash script emitted) to stdout.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Local imports (add project root to path if running as module)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # type: ignore
import numpy as np  # type: ignore

# Import training/eval utilities
from pretrain import PretrainConfig, init_train_state, create_dataloader, evaluate, DEFAULT_DEVICE  # type: ignore
from evaluate import EvalConfig  # type: ignore
from viz_common import find_latest_checkpoint_recursive


def find_latest_checkpoint() -> Path | None:  # thin shim for backward compat
    path = find_latest_checkpoint_recursive()
    return Path(path) if path else None


def load_config_for_checkpoint(ckpt: Path) -> PretrainConfig:
    cfg_file = ckpt.parent / 'all_config.yaml'
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing all_config.yaml next to {ckpt}")
    with cfg_file.open('r') as f:
        raw = yaml.safe_load(f)
    return PretrainConfig(**raw)  # type: ignore


def run_eval(ckpt: Path, save_outputs: list[str]) -> Dict[str, Dict[str, float]]:
    config = load_config_for_checkpoint(ckpt)
    config.eval_save_outputs = save_outputs
    config.checkpoint_path = str(ckpt.parent)

    # Build dataloaders (single process)
    train_loader, train_metadata = create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)
    eval_loader, eval_metadata = create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)

    # Train state (model container)
    train_state = init_train_state(config, train_metadata, world_size=1)

    # Load weights
    import torch
    state = torch.load(str(ckpt), map_location=DEFAULT_DEVICE)
    try:
        train_state.model.load_state_dict(state, assign=True)  # type: ignore
    except Exception:
        # Unwrap compile prefix if present
        state2 = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        train_state.model.load_state_dict(state2, assign=True)  # type: ignore

    # Infer step from filename
    name = ckpt.name
    if name.startswith('model_step_') and name.endswith('.pt'):
        train_state.step = int(name[len('model_step_'):-3])
    elif name.startswith('step_'):
        train_state.step = int(name[len('step_'):])

    # Eval
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=0, world_size=1) or {}
    return metrics  # Dict[split -> metrics dict]


def collect_dataset_info(config: PretrainConfig) -> tuple[int | None, int | None]:
    data_path = Path(config.data_path)
    test_dir = data_path / 'test'
    examples = None
    seq_len = None
    try:
        # seq_len from dataset.json (Sudoku/tiny builder format)
        meta_file = test_dir / 'dataset.json'
        if meta_file.exists():
            with meta_file.open('r') as f:
                meta = json.load(f)
            seq_len = meta.get('seq_len')
        inputs_file = test_dir / 'all__inputs.npy'
        if inputs_file.exists():
            arr = np.load(inputs_file, mmap_mode='r')
            examples = int(arr.shape[0])
            if seq_len is None:
                seq_len = int(arr.shape[1])
    except Exception:
        pass
    return examples, seq_len


def flatten_metrics(split_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    # For now assume single split 'all'
    if not split_metrics:
        return {}
    _, metrics = next(iter(split_metrics.items()))
    flat: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            flat[k] = float(v)
        except Exception:
            continue
    return flat


def format_output(ckpt: Path, duration: float, examples: int | None, seq_len: int | None, flat: Dict[str, float]):
    ex_per_s = tok_per_s = None
    if examples and duration > 0:
        ex_per_s = examples / duration
        if seq_len:
            tok_per_s = examples * seq_len / duration

    # Lines
    print(f"CHECKPOINT={ckpt}")
    print(f"DURATION_SEC={duration}")
    print(f"EXAMPLES={examples or ''}")
    print(f"SEQ_LEN={seq_len or ''}")
    print(f"EXAMPLES_PER_SEC={ex_per_s if ex_per_s is not None else ''}")
    print(f"TOKENS_PER_SEC={tok_per_s if tok_per_s is not None else ''}")
    print(f"METRICS_JSON={json.dumps(flat, separators=(',',':'))}")

    parts = []
    for k in ("accuracy","exact_accuracy","lm_loss","q_halt_accuracy","q_halt_loss","q_continue_loss","steps"):
        if k in flat:
            val = flat[k]
            parts.append(f"{k}={val:.4f}" if k.endswith('loss') or k not in ("steps",) else f"{k}={val:.4f}")
    if ex_per_s is not None:
        parts.append(f"ex/s={ex_per_s:.2f}")
    if tok_per_s is not None:
        parts.append(f"tok/s={tok_per_s:.1f}")
    print(f"SUMMARY={' '.join(parts)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', '-c', help='Path to checkpoint file (.pt). Auto-detect latest if omitted.')
    p.add_argument('--save-outputs', nargs='*', default=["inputs","labels","puzzle_identifiers","logits","q_halt_logits","q_continue_logits"], help='Model outputs to retain during eval (passed to model).')
    p.add_argument('--ignore-missing', action='store_true', help='Exit 0 (instead of error) if no checkpoints present.')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt_path is None:
        msg = "No checkpoint found."
        if args.ignore_missing:
            print(msg, file=sys.stderr)
            return 0
        print(msg, file=sys.stderr)
        return 2

    start = time.time()
    split_metrics = run_eval(ckpt_path, args.save_outputs)
    duration = time.time() - start
    try:
        config = load_config_for_checkpoint(ckpt_path)
    except Exception:
        # Can't derive dataset info
        config = None  # type: ignore
    examples = seq_len = None
    if config is not None:
        examples, seq_len = collect_dataset_info(config)

    flat = flatten_metrics(split_metrics)
    format_output(ckpt_path, duration, examples, seq_len, flat)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
