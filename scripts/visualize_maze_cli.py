#!/usr/bin/env python
"""Maze puzzle step visualizer (text) for HRM checkpoints.

Shows model predictions over the maze grid each ACT step, coloring changes and
tracking per-step diff stats. Supports optional animated GIF export.

Color semantics (when color enabled):
    Green = cell became correct this step
    Cyan  = cell changed this step but still incorrect
    Dim   = unchanged (correct or not yet correct)

Usage:
    python scripts/visualize_maze_cli.py --checkpoint-dir <ckpt_dir> --data-dir data/maze-30x30-hard-1k
    python scripts/visualize_maze_cli.py --checkpoint-dir <ckpt_dir> --data-dir data/maze-30x30-hard-1k --puzzle-index 3 --auto 0.4
    python scripts/visualize_maze_cli.py --checkpoint-dir <ckpt_dir> --data-dir data/maze-30x30-hard-1k --gif maze.gif
"""
from __future__ import annotations
import argparse, os, sys, glob, yaml, time, math, datetime, re
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
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # type: ignore
from puzzle_dataset import PuzzleDatasetMetadata  # type: ignore

RESET="\x1b[0m"; GREEN="\x1b[32m"; CYAN="\x1b[36m"; DIM="\x1b[2m"; RED="\x1b[31m"; YELLOW="\x1b[33m"; MAGENTA="\x1b[35m"; GRAY="\x1b[90m"

CHARSET = "# SGo"  # from build script
ID2CH = ["?"] + list(CHARSET)  # 0 pad maps to ? (untrained model may predict 0)

load_config = load_all_config
load_meta = load_dataset_metadata

def render(prev, curr, sol, side, use_color=True, pad_char='.', basic=False):
    """Default diff rendering (token-level) used when not in overlay mode."""
    pieces=[]
    curL=curr.tolist()
    solL=sol.tolist()
    prevL=None if prev is None else prev.tolist()
    for i,t in enumerate(curL):
        sol_t=solL[i]
        prev_t=None if prevL is None else prevL[i]
        ch = ID2CH[t] if 0 <= t < len(ID2CH) else '?'
        if ch=='?': ch=pad_char
        if not use_color or basic:
            pieces.append(ch)
            continue
        correct = (t==sol_t)
        changed = prev_t is not None and prev_t!=t
        if correct and (prev_t!=sol_t):
            ch=f"{GREEN}{ch}{RESET}"
        elif changed and not correct:
            ch=f"{CYAN}{ch}{RESET}"
        else:
            # Walls & structural characters dim gray for context, others faint
            if ch in ['#']:
                ch=f"{GRAY}{ch}{RESET}"
            else:
                ch=f"{DIM}{ch}{RESET}"
        pieces.append(ch)
    lines=[ ''.join(pieces[r*side:(r+1)*side]) for r in range(side) ]
    return '\n'.join(lines)

def render_overlay(curr, sol, side, use_color=True, pad_char='.'):
    """Overlay rendering highlighting path correctness independent of diff vs previous.

    Legend:
      # wall, S start, G goal
      Correct path (pred & label 'o'): GREEN o
      Missing path (label 'o' pred != 'o'): YELLOW *
      Wrong path (pred 'o' label != 'o'): RED x
      Pred pad (0) shown as pad_char (.)
    """
    curL=curr.tolist(); solL=sol.tolist()
    lines=[]
    for r in range(side):
        row=[]
        for c in range(side):
            i=r*side+c
            t=curL[i]; s=solL[i]
            ch_t = ID2CH[t] if 0 <= t < len(ID2CH) else '?'
            ch_s = ID2CH[s] if 0 <= s < len(ID2CH) else '?'
            if ch_t=='?': ch_t=pad_char
            out_char=' '
            color_prefix=''
            color_suffix=RESET if use_color else ''
            if ch_s in ['#','S','G']:
                # structural cells must match input & label; show as gray / magenta
                if ch_s=='#':
                    out_char='#'; color_prefix=GRAY if use_color else ''
                elif ch_s in ['S','G']:
                    out_char=ch_s; color_prefix=MAGENTA if use_color else ''
            else:
                # Path semantics
                if ch_s=='o' and ch_t=='o':
                    out_char='o'; color_prefix=GREEN if use_color else ''
                elif ch_s=='o' and ch_t!='o':
                    out_char='*'; color_prefix=YELLOW if use_color else ''
                elif ch_s!='o' and ch_t=='o':
                    out_char='x'; color_prefix=RED if use_color else ''
                else:
                    out_char='.' if use_color else pad_char
            row.append(f"{color_prefix}{out_char}{color_suffix}" if use_color else out_char)
        lines.append(''.join(row))
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(description="Maze reasoning visualization")
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint-dir')
    g.add_argument('--checkpoint-file')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--puzzle-index', type=int, default=None)
    ap.add_argument('--auto', type=float, default=None)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--no-color', action='store_true')
    ap.add_argument('--gif', type=str, default=None, help='Output animated GIF path (auto name if omitted unless --no-gif)')
    ap.add_argument('--no-gif', action='store_true')
    ap.add_argument('--gif-delay', type=float, default=0.6, help='Seconds per frame for GIF (default 0.6)')
    ap.add_argument('--sample-temp', type=float, default=0.0, help='Temperature >0 to sample tokens instead of argmax')
    ap.add_argument('--overlay', action='store_true', help='Use path correctness overlay (shows o/*/x)')
    ap.add_argument('--legend', action='store_true', help='Force printing legend (auto on when overlay)')
    ap.add_argument('--no-legend', action='store_true', help='Disable legend even if overlay')
    ap.add_argument('--pad-char', type=str, default='.', help='Character to display for PAD / unknown (default .)')
    ap.add_argument('--side-by-side', action='store_true', help='Print input | prediction | label each step')
    # Probability heatmap now ON by default; keep deprecated enabler for backward compatibility.
    ap.add_argument('--no-prob-heatmap', action='store_false', dest='prob_heatmap', help='Disable probability heatmap (enabled by default)')
    ap.add_argument('--prob-heatmap', action='store_true', dest='prob_heatmap', help='(deprecated – heatmap enabled by default)')
    ap.add_argument('--heatmap-gradient', type=str, default=' .:-=+*#%@', help='Gradient characters low->high (default " .:-=+*#%@")')
    ap.set_defaults(prob_heatmap=True)
    args=ap.parse_args()

    if args.sample_temp < 0:
        print('[ERROR] --sample-temp must be >=0', file=sys.stderr); sys.exit(2)
    # Auto GIF name
    derive_auto_gif_path('maze_viz', args, delay_attr='gif_delay', auto_attr='auto')

    if args.checkpoint_file:
        ckpt = args.checkpoint_file
        ckpt_dir = os.path.dirname(ckpt)
    else:
        ckpt_dir = args.checkpoint_dir
        ckpt = pick_ckpt(ckpt_dir)
        if ckpt is None:
            print('[ERROR] No checkpoint files found', file=sys.stderr)
            sys.exit(1)
    cfg=load_config(ckpt_dir, PretrainConfig)
    meta=load_meta(args.data_dir, PuzzleDatasetMetadata)
    model,_o,_l=create_model(cfg, train_metadata=meta, world_size=1)
    state=torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    hrm: HierarchicalReasoningModel_ACTV1 = model.model  # type: ignore

    arr_in=np.load(os.path.join(args.data_dir,'test','all__inputs.npy'))
    arr_lab=np.load(os.path.join(args.data_dir,'test','all__labels.npy'))
    n=arr_in.shape[0]
    idx = np.random.randint(0,n) if args.puzzle_index is None else max(0,min(n-1,args.puzzle_index))
    inp=torch.from_numpy(arr_in[idx:idx+1]).to(torch.int32)
    lab=torch.from_numpy(arr_lab[idx:idx+1]).to(torch.int32)
    device = next(model.parameters()).device
    batch={'inputs':inp.to(device), 'labels':lab.to(device), 'puzzle_identifiers':torch.zeros((1,),dtype=torch.int32, device=device)}

    carry=hrm.initial_carry(batch)
    side=int(math.sqrt(inp.shape[1]))
    max_steps=args.max_steps or hrm.config.halt_max_steps
    prev=None
    print(f"[INFO] Maze puzzle index={idx} side={side}")
    if args.overlay and not args.no_legend:
        args.legend=True
    if args.legend and not args.no_legend:
        print("[LEGEND] # wall  S start  G goal  GREEN o correct path  YELLOW * missing path  RED x wrong path  '.' empty")

    frames: List["Image.Image"] = []  # type: ignore[name-defined]
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')

    def sample_from_logits(lg: torch.Tensor) -> torch.Tensor:
        if args.sample_temp <= 0:
            return torch.argmax(lg, dim=-1)
        scaled = lg / max(1e-6, args.sample_temp)
        probs = torch.softmax(scaled, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()

    # Initial frame (input)
    if args.gif:
        base = render(None, inp[0].cpu(), lab[0].cpu(), side, use_color=not args.no_color, pad_char=args.pad_char)
        append_text_frame(frames, f"Initial (idx={idx})\n" + base)

    prev=None
    correct=-1; total=lab.shape[1]
    for step in range(1, max_steps+1):
        prob_grid = None
        with torch.inference_mode():
            carry, out = hrm(carry=carry, batch=batch)
            logits = out['logits']  # (B, Seq, V)
            step_pred = sample_from_logits(logits[0]).cpu()
            if args.prob_heatmap:
                prob_correct = compute_prob_correct(logits, lab[0])
                try:
                    prob_grid = prob_correct.view(side, side).cpu().numpy()
                except Exception:
                    prob_grid = None

        correct = (step_pred == lab[0]).sum().item()
        changed = 0 if prev is None else (step_pred != prev).sum().item()
        newly_correct = 0 if prev is None else ((step_pred == lab[0]) & (prev != lab[0])).sum().item()

        if args.side_by_side:
            raw_inp = render(None, inp[0].cpu(), lab[0].cpu(), side, use_color=False, pad_char=args.pad_char, basic=True).split('\n')
            pred_panel = (render_overlay(step_pred, lab[0], side, use_color=not args.no_color, pad_char=args.pad_char) if args.overlay
                          else render(prev, step_pred, lab[0], side, use_color=not args.no_color, pad_char=args.pad_char)).split('\n')
            label_panel = render(None, lab[0], lab[0], side, use_color=False, pad_char=args.pad_char, basic=True).split('\n')
            board_lines = [f"{raw_inp[r]} | {pred_panel[r]} | {label_panel[r]}" for r in range(side)]
            board = '\n'.join(board_lines)
        else:
            board = (render_overlay(step_pred, lab[0], side, use_color=not args.no_color, pad_char=args.pad_char) if args.overlay
                     else render(prev, step_pred, lab[0], side, use_color=not args.no_color, pad_char=args.pad_char))

        # Path quality stats (if 'o' token present)
        path_correct = path_missing = path_wrong = -1
        if 'o' in ID2CH:
            path_id = ID2CH.index('o')
            sol_arr = lab[0].cpu().numpy()
            pred_arr = step_pred.numpy()
            sol_mask = (sol_arr == path_id)
            pred_mask = (pred_arr == path_id)
            path_correct = int((sol_mask & pred_mask).sum())
            path_missing = int((sol_mask & (~pred_mask)).sum())
            path_wrong = int(((~sol_mask) & pred_mask).sum())

        pct = (correct / total) * 100 if total else 0.0
        extra = f" path: ok={path_correct} miss={path_missing} wrong={path_wrong}" if path_correct >= 0 else ''
        print(f"\n[Step {step}/{max_steps}] correct={correct}/{total} ({pct:.1f}%) +new={newly_correct} changed={changed} halt_q={out['q_halt_logits'][0].item():.2f}{extra}")
        print(board)
        if args.prob_heatmap and prob_grid is not None:
            print(prob_grid_to_ascii(prob_grid, args.heatmap_gradient or DEFAULT_GRADIENT))

        if args.gif:
            stats = f"Step {step}/{max_steps}  acc={correct}/{total} ({pct:.1f}%) +new={newly_correct} Δ={changed}{extra}"
            heat_block = ''
            if args.prob_heatmap and prob_grid is not None:
                heat_block = '\n' + prob_grid_to_ascii(prob_grid, args.heatmap_gradient or DEFAULT_GRADIENT)
            if not append_text_frame(frames, stats + '\n' + board + heat_block) and not frames:
                print('[WARN] GIF frame generation failed (PIL not available)')

        prev = step_pred.clone()
        if correct == total:
            print('[COMPLETE]')
            break
        if step < max_steps and args.auto:
            time.sleep(args.auto)

    if correct != total:
        print("\n[END] Reached max visualization steps.")

    # Save GIF
    if args.gif and frames:
        if save_gif(args.gif, frames, args.gif_delay):
            print(f"[INFO] Wrote GIF to {args.gif} ({len(frames)} frames)")
        else:
            print('[WARN] Failed to save GIF (PIL missing or error)')

if __name__=='__main__':
    main()
