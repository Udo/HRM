#!/usr/bin/env python
"""ARC puzzle step-by-step visualization for HRM checkpoints.

Displays 30x30 (ARCMaxGridSize) grid predictions each ACT step. Highlights cell
changes and newly-correct cells relative to the ground truth output grid.

Token mapping (see build_arc_dataset):
    0 = PAD, 1 = <eos> boundary marker, 2..11 = colors 0..9.

Usage examples:
    python scripts/visualize_arc_cli.py --checkpoint-dir checkpoints/arc-run --data-dir data/arc-aug-1000
    python scripts/visualize_arc_cli.py --checkpoint-file ckpt.pt --data-dir data/arc-aug-1000 --puzzle-index 12 --auto 0.3
    python scripts/visualize_arc_cli.py --checkpoint-dir checkpoints/arc-run --data-dir data/arc-aug-1000 --gif arc.gif --gif-delay 0.5
"""
from __future__ import annotations
import argparse, os, sys, glob, yaml, time, math, re, datetime
from typing import Optional, List
import numpy as np
import torch
from viz_common import (
    pick_latest_checkpoint as pick_ckpt,
    load_all_config,
    load_dataset_metadata,
    compute_prob_correct,
    prob_grid_to_ascii,
    append_text_frame,
    save_gif,
    derive_auto_gif_path,
    DEFAULT_GRADIENT,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pretrain import PretrainConfig, create_model  # type: ignore
from puzzle_dataset import PuzzleDatasetMetadata  # type: ignore
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # type: ignore

RESET="\x1b[0m"; GREEN="\x1b[32m"; CYAN="\x1b[36m"; DIM="\x1b[2m"

load_config = load_all_config
load_meta = load_dataset_metadata

def render(prev: Optional[torch.Tensor], curr: torch.Tensor, sol: torch.Tensor, no_color: bool) -> str:
    """Render a colorized ARC grid.

    Newly correct -> green, changed incorrect -> cyan, others dim.
    """
    pieces = []
    prev_list = None if prev is None else prev.view(-1).tolist()
    curr_list = curr.view(-1).tolist()
    sol_list  = sol.view(-1).tolist()
    for i, t in enumerate(curr_list):
        sol_t = sol_list[i]
        prev_t = None if prev_list is None else prev_list[i]
        if t == 0: ch = ' '
        elif t == 1: ch = '.'  # eos marker
        else: ch = str(t - 2)
        correct = (t == sol_t) and (sol_t not in (0,))
        changed = prev_t is not None and prev_t != t
        if no_color:
            pieces.append(ch)
        else:
            if correct and (prev_t != sol_t):
                pieces.append(f"{GREEN}{ch}{RESET}")
            elif changed and not correct:
                pieces.append(f"{CYAN}{ch}{RESET}")
            else:
                pieces.append(f"{DIM}{ch}{RESET}")
    # reshape to grid (assumes 30x30, but compute side for safety)
    lines = []
    side = int(len(pieces) ** 0.5)
    for r in range(side):
        lines.append(''.join(pieces[r*side:(r+1)*side]))
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser(description="ARC reasoning visualization")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint-dir')
    g.add_argument('--checkpoint-file')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--puzzle-index', type=int, default=None)
    ap.add_argument('--auto', type=float, default=None, help='Seconds between steps (auto-advance)')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--no-color', action='store_true')
    ap.add_argument('--gif', type=str, default=None, help='Write animated GIF (auto filename if omitted unless --no-gif)')
    ap.add_argument('--no-gif', action='store_true')
    ap.add_argument('--gif-delay', type=float, default=0.6, help='Seconds per GIF frame (default 0.6)')
    # Probability heatmap default ON; keep enabling flag for back-compat.
    ap.add_argument('--no-prob-heatmap', action='store_false', dest='prob_heatmap', help='Disable probability heatmap (enabled by default)')
    ap.add_argument('--prob-heatmap', action='store_true', dest='prob_heatmap', help='(deprecated – heatmap already enabled)')
    ap.add_argument('--heatmap-gradient', type=str, default=' .:-=+*#%@', help='Gradient chars low->high for heatmap')
    ap.set_defaults(prob_heatmap=True)
    args = ap.parse_args()

    derive_auto_gif_path('arc_viz', args, delay_attr='gif_delay', auto_attr='auto')

    if args.checkpoint_file:
        ckpt_file = args.checkpoint_file; ckpt_dir = os.path.dirname(ckpt_file)
    else:
        ckpt_dir = args.checkpoint_dir; ckpt_file = pick_ckpt(ckpt_dir)
        if ckpt_file is None:
            print('[ERROR] No checkpoint files found in checkpoint-dir', file=sys.stderr)
            sys.exit(1)

    cfg = load_config(ckpt_dir, PretrainConfig)
    meta = load_meta(args.data_dir, PuzzleDatasetMetadata)
    # Domain sanity check: ARC expects seq_len multiple of 900 (30x30); abort if mismatch
    if meta.seq_len != 900:
        print(f"[ABORT] Dataset seq_len={meta.seq_len} not 900 (ARC). Use correct dataset.", file=sys.stderr)
        sys.exit(2)
    model, _o, _l = create_model(cfg, train_metadata=meta, world_size=1)
    state = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    hrm: HierarchicalReasoningModel_ACTV1 = model.model  # type: ignore

    inputs = np.load(os.path.join(args.data_dir, 'test', 'all__inputs.npy'))
    labels = np.load(os.path.join(args.data_dir, 'test', 'all__labels.npy'))
    n = inputs.shape[0]
    if n == 0:
        print('[ERROR] Empty ARC dataset', file=sys.stderr); sys.exit(1)
    idx = np.random.randint(0, n) if args.puzzle_index is None else max(0, min(n-1, args.puzzle_index))
    inp = torch.from_numpy(inputs[idx:idx+1]).to(torch.int32)
    lab = torch.from_numpy(labels[idx:idx+1]).to(torch.int32)
    device = next(model.parameters()).device
    batch = {'inputs': inp.to(device), 'labels': lab.to(device), 'puzzle_identifiers': torch.zeros((1,), dtype=torch.int32, device=device)}
    carry = hrm.initial_carry(batch)
    max_steps = args.max_steps or hrm.config.halt_max_steps
    flat_solution = lab[0].to(torch.int32)
    prev_pred = None
    correct = -1
    total = flat_solution.numel()
    print(f"[INFO] ARC puzzle index={idx}")

    frames: List["Image.Image"] = []  # type: ignore[name-defined]
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')

    # If creating GIF and user did not request auto stepping, still advance automatically
    auto_delay = args.auto if args.auto is not None else (args.gif_delay if args.gif else None)

    # Frame 0: initial input vs solution diff
    if args.gif:
        init_tokens = inp[0]
        init_grid = render(None, init_tokens.cpu(), flat_solution, args.no_color)
        header = f"Initial (index={idx})\n" + init_grid
        append_text_frame(frames, header)

    for step in range(1, max_steps+1):
        with torch.inference_mode():
            carry, outputs = hrm(carry=carry, batch=batch)
            logits = outputs['logits']
            pred = torch.argmax(logits, dim=-1)[0].cpu()
            if args.prob_heatmap:
                prob_correct = compute_prob_correct(logits, flat_solution)
                try:
                    side = int(math.sqrt(prob_correct.shape[0]))
                    grid_prob = prob_correct.view(side, side).cpu().numpy()
                except Exception:
                    grid_prob = None
            else:
                grid_prob = None
        correct = (pred == flat_solution).sum().item()
        changed = 0 if prev_pred is None else (pred != prev_pred).sum().item()
        newly_correct = 0 if prev_pred is None else ((pred == flat_solution) & (prev_pred != flat_solution)).sum().item()
        grid_str = render(prev_pred, pred, flat_solution, args.no_color)
        halt_q = outputs['q_halt_logits'][0].item() if 'q_halt_logits' in outputs else float('nan')
        pct = (correct/total)*100 if total else 0.0
        print(f"\n[Step {step}/{max_steps}] correct={correct}/{total} ({pct:.1f}%) +new={newly_correct} changed={changed} halt_q={halt_q:.2f}")
        print(grid_str)
        if args.prob_heatmap and grid_prob is not None:
            print(prob_grid_to_ascii(grid_prob, args.heatmap_gradient or DEFAULT_GRADIENT))
        if args.gif:
            stats_line = f"Step {step}/{max_steps}  correct={correct}/{total} ({pct:.1f}%) +new={newly_correct} Δ={changed}"
            heat_block = ''
            if args.prob_heatmap and grid_prob is not None:
                heat_block = '\n' + prob_grid_to_ascii(grid_prob, args.heatmap_gradient or DEFAULT_GRADIENT)
            if not append_text_frame(frames, stats_line + '\n' + grid_str + heat_block) and not frames:
                print('[WARN] GIF frame rendering failed (PIL not available)')
        prev_pred = pred.clone()
        if correct == total:
            print('[COMPLETE]'); break
        if step < max_steps and auto_delay is not None:
            time.sleep(auto_delay)

    if correct != total:
        print("\n[END] Reached max visualization steps.")

    # Save GIF
    if args.gif and frames:
        if save_gif(args.gif, frames, args.gif_delay):
            print(f"[INFO] Wrote GIF to {args.gif} ({len(frames)} frames)")
        else:
            print('[WARN] Failed to save GIF (PIL missing or error)')

if __name__ == '__main__':
    main()
