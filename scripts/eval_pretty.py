#!/usr/bin/env python
"""Pretty terminal evaluation summary.

Usage:
  python scripts/eval_pretty.py [--checkpoint CKPT] [--target-accuracy 0.9] [--no-color] [--json]

If --checkpoint omitted, picks latest model_step_*.pt (or legacy step_*).
Prints a formatted box with:
  - Model / architecture basics
  - Checkpoint path & step
  - Dataset info (seq_len, vocab, examples evaluated)
  - Performance metrics (accuracy, losses, steps, throughput)
  - Success rate vs optional target threshold

Returns exit code 0. If --target-accuracy is set and accuracy < target, exit code 2.
"""
from __future__ import annotations
import argparse, json, os, sys, time, math
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

from pretrain import PretrainConfig, init_train_state, create_dataloader, evaluate, DEFAULT_DEVICE  # type: ignore
from viz_common import find_latest_checkpoint_recursive

BOX_HORIZONTAL = '─'
BOX_VERTICAL = '│'
BOX_TL = '╭'
BOX_TR = '╮'
BOX_BL = '╰'
BOX_BR = '╯'
BOX_TJ = '┬'
BOX_BJ = '┴'

COLOR_RESET = '\x1b[0m'
COLORS = {
    'cyan': '\x1b[36m',
    'magenta': '\x1b[35m',
    'yellow': '\x1b[33m',
    'green': '\x1b[32m',
    'red': '\x1b[31m',
    'blue': '\x1b[34m',
    'bold': '\x1b[1m',
    'dim': '\x1b[2m',
}

def c(text: str, color: str, enable: bool) -> str:
    if not enable or color not in COLORS:
        return text
    return f"{COLORS[color]}{text}{COLOR_RESET}"

def find_latest_checkpoint() -> Path | None:  # shim
    p = find_latest_checkpoint_recursive()
    return Path(p) if p else None

def load_config(ckpt: Path) -> PretrainConfig:
    cfg_file = ckpt.parent / 'all_config.yaml'
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing all_config.yaml next to {ckpt}")
    raw = yaml.safe_load(cfg_file.read_text())
    return PretrainConfig(**raw)  # type: ignore

def build_and_load(ckpt: Path):
    cfg = load_config(ckpt)
    # Signature: create_dataloader(config, split, rank, world_size, **kwargs)
    train_loader, train_meta = create_dataloader(
        cfg,
        'train',
        rank=0,
        world_size=1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )
    eval_loader, eval_meta = create_dataloader(
        cfg,
        'test',
        rank=0,
        world_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )
    state = init_train_state(cfg, train_meta, world_size=1)
    raw = torch.load(str(ckpt), map_location=DEFAULT_DEVICE)
    try:
        state.model.load_state_dict(raw, assign=True)  # type: ignore
    except Exception:
        raw2 = {k.removeprefix('_orig_mod.'): v for k,v in raw.items()}
        state.model.load_state_dict(raw2, assign=True)  # type: ignore
    name = ckpt.name
    step = None
    if name.startswith('model_step_') and name.endswith('.pt'):
        step = int(name[len('model_step_'):-3])
    elif name.startswith('step_'):
        step = int(name[len('step_'):])
    return cfg, state, eval_loader, eval_meta, step

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def collect_dataset_info(cfg: PretrainConfig) -> Tuple[int | None, int | None, int | None]:
    data_path = Path(cfg.data_path)
    test_dir = data_path / 'test'
    examples = None
    seq_len = None
    vocab = None
    try:
        meta_file = test_dir / 'dataset.json'
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            seq_len = meta.get('seq_len')
            vocab = meta.get('vocab_size')
            examples = meta.get('size') or meta.get('num_examples')
        inputs_file = test_dir / 'all__inputs.npy'
        if inputs_file.exists():
            arr = np.load(inputs_file, mmap_mode='r')
            examples = int(arr.shape[0])
            if seq_len is None:
                seq_len = int(arr.shape[1])
    except Exception:
        pass
    return examples, seq_len, vocab

def box(lines: list[str], color: bool) -> str:
    width = max(len(strip_ansi(l)) for l in lines) if lines else 0
    top = f"{BOX_TL}{BOX_HORIZONTAL * (width + 2)}{BOX_TR}"
    bottom = f"{BOX_BL}{BOX_HORIZONTAL * (width + 2)}{BOX_BR}"
    inner = [f"{BOX_VERTICAL} {pad(l, width)} {BOX_VERTICAL}" for l in lines]
    return '\n'.join([top, *inner, bottom])

def strip_ansi(s: str) -> str:
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

def pad(s: str, w: int) -> str:
    return s + ' ' * (w - len(strip_ansi(s)))

def fmt_big(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', '-c')
    ap.add_argument('--target-accuracy', type=float, help='Optional target accuracy to highlight & affect exit code.')
    ap.add_argument('--no-color', action='store_true')
    ap.add_argument('--json', action='store_true', help='Also output machine-readable JSON line.')
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt is None:
        print('No checkpoint found.', file=sys.stderr)
        return 2
    if not ckpt.exists():
        print(f'Checkpoint does not exist: {ckpt}', file=sys.stderr)
        return 2

    color = not args.no_color and sys.stdout.isatty()

    start = time.time()
    cfg, state, eval_loader, eval_meta, step = build_and_load(ckpt)
    with torch.inference_mode():
        metrics = evaluate(cfg, state, eval_loader, eval_meta, rank=0, world_size=1) or {}
    duration = time.time() - start

    flat: Dict[str, float] = {}
    if metrics:
        _, m = next(iter(metrics.items()))
        for k, v in m.items():
            try:
                flat[k] = float(v)
            except Exception:
                pass

    acc = flat.get('accuracy')
    exact = flat.get('exact_accuracy')
    lm_loss = flat.get('lm_loss')
    q_halt_acc = flat.get('q_halt_accuracy')
    steps_metric = flat.get('steps')

    examples, seq_len, vocab = collect_dataset_info(cfg)
    ex_per_s = examples / duration if examples and duration > 0 else None
    tok_per_s = examples * seq_len / duration if examples and seq_len and duration > 0 else None

    total_params = param_count(state.model)
    success_line = ''
    exit_code = 0
    if acc is not None and args.target_accuracy is not None:
        if acc >= args.target_accuracy:
            success_line = c(f"Target accuracy met: {acc:.4f} ≥ {args.target_accuracy}", 'green', color)
        else:
            success_line = c(f"Target accuracy NOT met: {acc:.4f} < {args.target_accuracy}", 'red', color)
            exit_code = 2

    def num_or_dash(v):
        return f"{v:.4f}" if isinstance(v, (float,int)) and v is not None and not math.isnan(v) else '—'

    lines = []
    title = f"HRM Evaluation Summary"
    lines.append(c(title, 'bold', color))
    lines.append(f"Checkpoint: {ckpt}")
    if step is not None:
        lines.append(f"Step: {step}")
    lines.append(f"Params: {fmt_big(total_params)}")
    # arch extras stored in cfg.arch.__pydantic_extra__
    arch_extra = getattr(cfg.arch, '__pydantic_extra__', {}) or {}
    h = arch_extra.get('hidden_size')
    heads = arch_extra.get('num_heads')
    H_layers = arch_extra.get('H_layers')
    L_layers = arch_extra.get('L_layers')
    H_cycles = arch_extra.get('H_cycles')
    L_cycles = arch_extra.get('L_cycles')
    expansion = arch_extra.get('expansion')
    lines.append(f"Arch: H={h} heads={heads} H_layers={H_layers} L_layers={L_layers} H_cycles={H_cycles} L_cycles={L_cycles} expansion={expansion}")
    lines.append(f"Dataset: path={cfg.data_path} seq_len={seq_len or '??'} vocab={vocab or '??'} test_examples={examples or '??'}")
    exps = f"{ex_per_s:.2f}" if ex_per_s else "0.00"
    toks = f"{tok_per_s:.1f}" if tok_per_s else "0.0"
    lines.append(f"Duration: {duration:.2f}s examples/s={exps} tokens/s={toks}")
    lines.append("Metrics:")
    metric_lines = [
        ("accuracy", acc),
        ("exact_accuracy", exact),
        ("lm_loss", lm_loss),
        ("q_halt_accuracy", q_halt_acc),
        ("steps", steps_metric),
    ]
    for k,v in metric_lines:
        if k.endswith('loss') and v is not None:
            color_key = 'yellow' if v > 1.5 else 'green'
        elif k.endswith('accuracy') and v is not None:
            color_key = 'green' if v >= 0.5 else 'red'
        else:
            color_key = 'cyan'
        lines.append(f"  {k}: {c(num_or_dash(v), color_key, color)}")
    if ex_per_s:
        lines.append(f"  examples/s: {c(f'{ex_per_s:.2f}', 'magenta', color)}")
    if tok_per_s:
        lines.append(f"  tokens/s: {c(f'{tok_per_s:.1f}', 'magenta', color)}")
    if success_line:
        lines.append(success_line)

    print(box(lines, color))

    if args.json:
        out = dict(
            checkpoint=str(ckpt), step=step, params=total_params,
            arch={k: arch_extra.get(k) for k in ('hidden_size','num_heads','H_layers','L_layers','H_cycles','L_cycles','expansion')},
            dataset=dict(path=cfg.data_path, seq_len=seq_len, vocab=vocab, test_examples=examples),
            duration_sec=duration, examples_per_sec=ex_per_s, tokens_per_sec=tok_per_s,
            metrics=flat,
            target_accuracy=args.target_accuracy,
            target_met=(acc is not None and args.target_accuracy is not None and acc >= args.target_accuracy)
        )
        print(json.dumps(out, separators=(',',':')))
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
