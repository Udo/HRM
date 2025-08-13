#!/usr/bin/env python
"""Parameter counting helper for HRM.

Usage:
  python scripts/param_count.py [--data-path PATH] [--json] [override ...]

Examples:
  python scripts/param_count.py arch.hidden_size=256 arch.num_heads=8 arch.H_layers=2 arch.L_layers=2
  python scripts/param_count.py --data-path data/tiny arch.hidden_size=384 arch.num_heads=6
  python scripts/param_count.py --json arch.hidden_size=512 arch.H_layers=4 arch.L_layers=4

Recognized override keys (with optional 'arch.' prefix):
  hidden_size, num_heads, H_layers, L_layers, H_cycles, L_cycles,
  expansion, puzzle_emb_ndim, pos_encodings,
  halt_max_steps, halt_exploration_prob,
  vocab_size, seq_len, num_puzzle_identifiers.

If --data-path is provided, attempts to read dataset metadata from <data-path>/test/dataset.json
(falling back to train/ if test/ not present). Otherwise defaults are used.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any

# Defaults (mirroring config/arch/hrm_v1.yaml and tiny dataset assumptions)
DEFAULTS: Dict[str, Any] = dict(
    hidden_size=512,
    num_heads=8,
    H_layers=4,
    L_layers=4,
    H_cycles=2,
    L_cycles=2,
    expansion=4.0,
    puzzle_emb_ndim=512,
    pos_encodings='rope',
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    # Dataset derived
    vocab_size=11,
    seq_len=81,
    num_puzzle_identifiers=1,
)

RECOGNIZED_KEYS = set(DEFAULTS.keys())


def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


def multiple_256(x: int) -> int:
    return ((x + 255) // 256) * 256


def load_dataset_meta(data_path: Path, overrides: Dict[str, Any]):
    if not data_path:
        return
    for split in ('test', 'train'):
        meta_file = data_path / split / 'dataset.json'
        if meta_file.exists():
            try:
                import json as _json
                with meta_file.open('r') as f:
                    meta = _json.load(f)
                # Minimal expected keys
                for k in ('seq_len', 'vocab_size', 'num_puzzle_identifiers'):
                    if k in meta:
                        overrides.setdefault(k, meta[k])
                break
            except Exception:
                pass


def parse_overrides(kvs: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs:
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        if k.startswith('arch.'):
            k = k[5:]
        if k not in RECOGNIZED_KEYS:
            # Permit dataset-level overrides like vocab_size directly
            if k not in ('vocab_size', 'seq_len', 'num_puzzle_identifiers'):
                continue
        # Try int then float
        if v.lower() in ('true','false'):
            out[k] = v.lower() == 'true'
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            try:
                out[k] = float(v)
                continue
            except ValueError:
                out[k] = v
    return out


def compute_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    H = int(cfg['hidden_size'])
    num_heads = int(cfg['num_heads'])
    assert H % num_heads == 0, 'hidden_size must be divisible by num_heads'
    head_dim = H // num_heads

    H_layers = int(cfg['H_layers'])
    L_layers = int(cfg['L_layers'])
    expansion = float(cfg['expansion'])
    puzzle_emb_ndim = int(cfg['puzzle_emb_ndim'])
    vocab_size = int(cfg['vocab_size'])
    num_puzzle_identifiers = int(cfg['num_puzzle_identifiers'])
    seq_len = int(cfg['seq_len'])
    pos_encodings = cfg.get('pos_encodings','rope')

    # SwiGLU intermediate dim (matches layers.SwiGLU logic with multiple of 256)
    inter_raw = round(expansion * H * 2 / 3)
    inter = multiple_256(inter_raw)

    # Per block params
    attn_qkv = (3 * H) * H  # qkv_proj weight
    attn_o = H * H          # o_proj
    attn_total = attn_qkv + attn_o  # 4H^2
    mlp_gate_up = (2 * inter) * H
    mlp_down = inter * H
    mlp_total = mlp_gate_up + mlp_down  # 3 * inter * H
    block_params = attn_total + mlp_total

    total_blocks = (H_layers + L_layers) * block_params

    # Embeddings
    token_emb = vocab_size * H
    lm_head = vocab_size * H  # weight (no bias)
    q_head = (2 * H) + 2      # weight + bias

    puzzle_emb = 0
    puzzle_emb_len = 0
    if puzzle_emb_ndim > 0:
        puzzle_emb = num_puzzle_identifiers * puzzle_emb_ndim
        puzzle_emb_len = ceil_div(puzzle_emb_ndim, H)

    pos_emb = 0
    if pos_encodings == 'learned':
        pos_emb = (seq_len + puzzle_emb_len) * H

    init_states = 2 * H  # H_init + L_init buffers

    total = total_blocks + token_emb + lm_head + q_head + puzzle_emb + pos_emb + init_states

    return dict(
        hidden_size=H,
        num_heads=num_heads,
        H_layers=H_layers,
        L_layers=L_layers,
        expansion=expansion,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        puzzle_emb_ndim=puzzle_emb_ndim,
        puzzle_emb_len=puzzle_emb_len,
        inter_dim=inter,
        per_block_params=block_params,
        total_block_params=total_blocks,
        token_embedding=token_emb,
        lm_head=lm_head,
        puzzle_embedding=puzzle_emb,
        positional_embedding=pos_emb,
        q_head=q_head,
        init_states=init_states,
        total_params=total,
    )


def human_readable(report: Dict[str, Any]):
    total = report['total_params']
    def fmt(n):
        return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}K" if n >= 1e3 else str(n))
    print(f"Total Params: {total} ({fmt(total)})")
    print("Breakdown (approx):")
    print(f"  Blocks ({report['H_layers']}H+{report['L_layers']}L): {report['total_block_params']} ({fmt(report['total_block_params'])})")
    print(f"  Token Embedding: {report['token_embedding']} ({fmt(report['token_embedding'])})")
    print(f"  LM Head: {report['lm_head']} ({fmt(report['lm_head'])})")
    if report['puzzle_embedding']:
        print(f"  Puzzle Embedding: {report['puzzle_embedding']} ({fmt(report['puzzle_embedding'])}) len={report['puzzle_emb_len']}")
    if report['positional_embedding']:
        print(f"  Positional Embedding: {report['positional_embedding']} ({fmt(report['positional_embedding'])})")
    print(f"  Q Head: {report['q_head']}")
    print(f"  Init States: {report['init_states']}")
    print(f"  Inter Dim (SwiGLU): {report['inter_dim']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', help='Dataset root to infer vocab/seq_len/identifiers')
    ap.add_argument('--json', action='store_true', help='Output JSON only')
    ap.add_argument('overrides', nargs='*', help='Key=val overrides (Hydra style).')
    args = ap.parse_args()

    cfg = DEFAULTS.copy()
    ov = parse_overrides(args.overrides)

    # Dataset metadata can supply vocab/seq_len/ids if not overridden
    if args.data_path:
        load_dataset_meta(Path(args.data_path), ov)

    cfg.update(ov)

    report = compute_params(cfg)

    if args.json:
        print(json.dumps(report, separators=(',',':')))
    else:
        human_readable(report)
        # Also print a one-line summary for scripts
        print("SUMMARY total_params=" + str(report['total_params']))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
