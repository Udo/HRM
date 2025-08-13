#!/usr/bin/env python
"""Interactive / step-by-step Sudoku solver visualization using a trained HRM checkpoint.

This script replays the model's internal ACT reasoning steps on a single Sudoku
puzzle and prints the evolving board each step so you can *see* it converge.

Usage examples:
  # Auto-pick latest checkpoint within a run directory
  python scripts/visualize_sudoku_cli.py --checkpoint-dir checkpoints/YourProject/YourRun \
      --data-dir data/sudoku-extreme-1k-aug-10 --puzzle-index 0

  # Specify an explicit checkpoint file & random puzzle
  python scripts/visualize_sudoku_cli.py --checkpoint-file checkpoints/.../model_step_123.pt \
      --data-dir data/sudoku-extreme-1k-aug-10 --puzzle-index 42

  # Auto-step with 0.5s delay
  python scripts/visualize_sudoku_cli.py --checkpoint-dir checkpoints/... --auto 0.5

Controls (interactive mode):
  ENTER  -> advance one reasoning step
  q      -> quit

Requires: A checkpoint produced by pretrain.py (so all_config.yaml exists in the
same directory) and a built Sudoku dataset.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time
import yaml
import math
from typing import Optional, List

import numpy as np
import torch

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # repo root
from pretrain import PretrainConfig, create_model  # type: ignore
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # type: ignore
from puzzle_dataset import PuzzleDatasetMetadata  # type: ignore

COLOR_RESET = "\x1b[0m"
COLOR_GREEN = "\x1b[32m"
COLOR_YELLOW = "\x1b[33m"
COLOR_CYAN = "\x1b[36m"
COLOR_DIM = "\x1b[2m"


def pick_latest_checkpoint(dir_path: str) -> Optional[str]:
    cand = sorted(glob.glob(os.path.join(dir_path, 'model_step_*.pt')))
    return cand[-1] if cand else None


def load_config(checkpoint_dir: str) -> PretrainConfig:
    cfg_path = os.path.join(checkpoint_dir, 'all_config.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, 'r') as f:
        raw = yaml.safe_load(f)
    # Backwards compatibility: ensure required keys present
    return PretrainConfig(**raw)


def load_metadata(data_dir: str) -> PuzzleDatasetMetadata:
    meta_path = os.path.join(data_dir, 'test', 'dataset.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}. Did you build the Sudoku dataset?")
    with open(meta_path, 'r') as f:
        md = yaml.safe_load(f)
    return PuzzleDatasetMetadata(**md)


def decode_tokens(tokens: torch.Tensor) -> str:
    # Tokens: 0=PAD, 1='0'(blank), 2..10 digits 1..9
    chars = []
    for t in tokens.tolist():
        if t == 1:
            chars.append('.')
        elif t == 0:
            chars.append(' ')
        else:
            chars.append(str(t - 1))
    # 9x9 formatting
    lines = []
    for r in range(9):
        row = ''.join(chars[r*9:(r+1)*9])
        # Insert subgrid separators visually
        row = f"{row[0:3]} {row[3:6]} {row[6:9]}"
        lines.append(row)
    return '\n'.join(lines)


def colorize_board(prev: Optional[torch.Tensor], curr: torch.Tensor, solution: torch.Tensor) -> str:
    """Return colorized string of current board.

    Green: cell now matches solution & wasn't correct previous step.
    Cyan:  cell value changed (prediction update) but not yet correct.
    Yellow: unchanged & incorrect (dim for blanks).
    """
    pieces = []
    for idx, t in enumerate(curr.tolist()):
        sol = solution[idx].item()
        prev_t = None if prev is None else prev[idx].item()
        ch = '.' if t == 1 else (' ' if t == 0 else str(t-1))
        correct = (t == sol) and (sol != 1)
        was_correct = prev_t is not None and prev_t == sol and sol != 1
        changed = prev_t is not None and prev_t != t
        if correct and not was_correct:
            ch = f"{COLOR_GREEN}{ch}{COLOR_RESET}"
        elif changed and not correct:
            ch = f"{COLOR_CYAN}{ch}{COLOR_RESET}"
        elif (t == 1) or not correct:
            ch = f"{COLOR_DIM}{ch}{COLOR_RESET}"
        pieces.append(ch)
    # Reuse formatting
    lines = []
    for r in range(9):
        row = ''.join(pieces[r*9:(r+1)*9])
        row = f"{row[0:3]} {row[3:6]} {row[6:9]}"
        lines.append(row)
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description="Visual step-by-step Sudoku solving visualization")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint-dir', type=str, help="Directory containing model_step_*.pt and all_config.yaml")
    g.add_argument('--checkpoint-file', type=str, help="Specific checkpoint file (.pt)")
    p.add_argument('--data-dir', type=str, required=True, help="Sudoku dataset directory (e.g. data/sudoku-extreme-1k-aug-10)")
    p.add_argument('--puzzle-index', type=int, default=None, help="Index of puzzle to visualize (default: random)")
    p.add_argument('--max-steps', type=int, default=None, help="Override halt_max_steps for visualization (<= configured value)")
    p.add_argument('--auto', type=float, default=None, help="Auto-step every N seconds instead of waiting for ENTER")
    p.add_argument('--no-color', action='store_true', help="Disable ANSI colors")
    p.add_argument('--gif', type=str, default=None, help="Output path to write animated GIF of steps (auto set if omitted unless --no-gif)")
    p.add_argument('--no-gif', action='store_true', help='Disable automatic GIF creation')
    p.add_argument('--gif-delay', type=float, default=0.6, help="Seconds per frame when writing GIF (default 0.6)")
    p.add_argument('--seed', type=int, default=0)
    # Probability heatmap now enabled by default; retain --prob-heatmap for backward compatibility.
    p.add_argument('--no-prob-heatmap', action='store_false', dest='prob_heatmap', help='Disable probability heatmap (enabled by default)')
    p.add_argument('--prob-heatmap', action='store_true', dest='prob_heatmap', help='(deprecated – heatmap already enabled)')
    p.add_argument('--heatmap-gradient', type=str, default=' .:-=+*#%@', help='Gradient chars low-to-high for heatmap')
    p.set_defaults(prob_heatmap=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Derive default GIF path if not provided and not disabled
    if args.gif is None and not args.no_gif:
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.gif = f"sudoku_viz_{ts}.gif"
        if args.auto is None:
            args.auto = args.gif_delay  # ensure progress for GIF frames

    if args.checkpoint_file:
        ckpt_file = args.checkpoint_file
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_file))
    else:
        ckpt_dir = args.checkpoint_dir
        ckpt_file = pick_latest_checkpoint(ckpt_dir)
        if ckpt_file is None:
            print(f"[ERROR] No checkpoint files found in {ckpt_dir}", file=sys.stderr)
            sys.exit(1)
    print(f"[INFO] Using checkpoint: {ckpt_file}")

    # Load config & metadata
    config = load_config(ckpt_dir)
    metadata = load_metadata(args.data_dir)
    # Abort if seq len not 81 for Sudoku
    if metadata.seq_len != 81:
        print(f"[ABORT] Dataset seq_len={metadata.seq_len} not 81 (Sudoku). Use correct dataset.", file=sys.stderr)
        sys.exit(2)

    # Build model (world_size=1)
    model, _opts, _lrs = create_model(config, train_metadata=metadata, world_size=1)  # wrapped loss head
    state = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    # Underlying ACT model
    # model is ACTLossHead; underlying reasoning model is model.model
    hrm: HierarchicalReasoningModel_ACTV1 = model.model  # type: ignore

    # Load dataset arrays directly
    inputs = np.load(os.path.join(args.data_dir, 'test', 'all__inputs.npy'))
    labels = np.load(os.path.join(args.data_dir, 'test', 'all__labels.npy'))
    n = inputs.shape[0]
    if n == 0:
        print('[ERROR] Empty dataset')
        sys.exit(1)

    if args.puzzle_index is None:
        puzzle_index = int(np.random.randint(0, n))
    else:
        puzzle_index = max(0, min(n-1, args.puzzle_index))

    puzzle_inp = torch.from_numpy(inputs[puzzle_index:puzzle_index+1]).to(torch.int32)
    puzzle_lab = torch.from_numpy(labels[puzzle_index:puzzle_index+1]).to(torch.int32)
    puzzle_ids = torch.zeros((1,), dtype=torch.int32)

    batch = {
        'inputs': puzzle_inp,
        'labels': puzzle_lab,
        'puzzle_identifiers': puzzle_ids,
    }

    # Initialize ACT wrapper carry
    carry = hrm.initial_carry(batch)
    max_steps = args.max_steps or hrm.config.halt_max_steps
    max_steps = min(max_steps, hrm.config.halt_max_steps)

    flat_solution = puzzle_lab[0]

    print('\n=== Sudoku Puzzle (index=%d) ===' % puzzle_index)
    print('Original (blanks=.)')
    print(decode_tokens(puzzle_inp[0]))
    print('\nSolution (for reference)')
    print(decode_tokens(flat_solution))
    print('\n--- Reasoning steps ---')

    prev_pred = None
    correct_cells = -1
    total_cells = -1
    frames: List["Image.Image"] = []  # type: ignore[name-defined]
    for step in range(1, max_steps+1):
        with torch.inference_mode():
            carry, outputs = hrm(carry=carry, batch=batch)
            logits = outputs['logits']  # (B, Seq, V)
            pred = torch.argmax(logits, dim=-1)[0]
            if args.prob_heatmap:
                probs = torch.softmax(logits[0], dim=-1)
                gt = flat_solution
                prob_correct = probs[torch.arange(probs.shape[0]), gt]
                grid_prob = prob_correct.view(9,9).cpu().numpy()
            else:
                grid_prob = None

        correct_cells = (pred == flat_solution).sum().item()
        total_cells = (flat_solution != 0).sum().item()
        changed_cells = 0 if prev_pred is None else (pred != prev_pred).sum().item()
        newly_correct = 0 if prev_pred is None else ((pred == flat_solution) & (prev_pred != flat_solution)).sum().item()
        frac = correct_cells / max(1, total_cells)

        if args.no_color:
            board_str = decode_tokens(pred)
        else:
            board_str = colorize_board(prev_pred, pred, flat_solution)  # type: ignore[arg-type]

        print(f"\n[Step {step}/{max_steps}] correct={correct_cells}/{total_cells} ({frac*100:.1f}%) +new={newly_correct} changed={changed_cells} q_halt_logit={outputs['q_halt_logits'][0].item():.2f} q_continue_logit={outputs['q_continue_logits'][0].item():.2f}")
        print(board_str)
        if args.prob_heatmap and grid_prob is not None:
            grad = args.heatmap_gradient or ' .:-=+*#%@'
            L = len(grad)-1 if len(grad)>1 else 1
            lines=[]
            for r in range(9):
                row=[]
                for c in range(9):
                    p = float(grid_prob[r][c]); p=max(0.0,min(1.0,p))
                    idx=int(round(p*L))
                    row.append(grad[idx])
                lines.append(''.join(row))
            print('[ProbHeatmap]\n'+'\n'.join(lines))

        if args.gif:
            try:
                from PIL import Image, ImageDraw, ImageFont
                # Render text to image (monospace assumption)
                # Attempt to load a default monospace font; fallback to default.
                try:
                    font = ImageFont.load_default()
                except Exception:  # noqa: BLE001
                    font = None  # type: ignore

                lines = [f"Step {step}/{max_steps}", f"Correct: {correct_cells}/{total_cells} ({frac*100:.1f}%)", board_str]
                if args.prob_heatmap and grid_prob is not None:
                    grad = args.heatmap_gradient or ' .:-=+*#%@'
                    L = len(grad)-1 if len(grad)>1 else 1
                    heat_lines=[]
                    for r in range(9):
                        row=[]
                        for c in range(9):
                            p=float(grid_prob[r][c]); p=max(0.0,min(1.0,p)); idx=int(round(p*L)); row.append(grad[idx])
                        heat_lines.append(''.join(row))
                    lines.append('[ProbHeatmap]')
                    lines.extend(heat_lines)
                text = "\n".join(lines)
                # Measure
                dummy = Image.new('RGB', (10,10), 'white')
                draw = ImageDraw.Draw(dummy)
                w = max(draw.textlength(l, font=font) for l in text.split('\n')) + 10
                h = (font.getbbox('Hg')[3]-font.getbbox('Hg')[1] if font else 10) * (len(lines)+1) + 10
                img = Image.new('RGB', (int(w), int(h)), 'white')
                draw = ImageDraw.Draw(img)
                # Strip ANSI codes for GIF text
                import re
                ansi_re = re.compile(r'\x1b\[[0-9;]*m')
                clean_text = ansi_re.sub('', text)
                draw.multiline_text((5,5), clean_text, fill=(0,0,0), font=font, spacing=2)
                frames.append(img)
            except Exception as e:  # noqa: BLE001
                if len(frames) == 0:
                    print(f"[WARN] GIF frame rendering failed: {e}")

        prev_pred = pred.clone()

        # Early completion if fully correct
        if correct_cells == total_cells:
            print(f"\n[COMPLETE] Puzzle solved at step {step}.")
            break

        if step < max_steps:
            if args.auto is not None:
                time.sleep(args.auto if args.auto else 0.0)
            else:
                try:
                    inp = input("Press ENTER for next step (q to quit): ")
                    if inp.strip().lower() == 'q':
                        break
                except KeyboardInterrupt:
                    break
    # Reached loop end without full solution
    if correct_cells != total_cells:
        print("\n[END] Reached max visualization steps.")

    # Write GIF if requested
    if args.gif and frames:
        try:
            duration_ms = max(50, int(args.gif_delay * 1000))
            first, *rest = frames
            first.save(args.gif, save_all=True, append_images=rest, duration=duration_ms, loop=0, optimize=False)
            print(f"[INFO] Wrote GIF to {args.gif} ({len(frames)} frames)")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to save GIF: {e}")


if __name__ == '__main__':
    main()
