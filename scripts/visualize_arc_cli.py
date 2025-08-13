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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pretrain import PretrainConfig, create_model  # type: ignore
from puzzle_dataset import PuzzleDatasetMetadata  # type: ignore
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # type: ignore

RESET="\x1b[0m"; GREEN="\x1b[32m"; CYAN="\x1b[36m"; DIM="\x1b[2m"

def pick_ckpt(d: str) -> Optional[str]:
    c = sorted(glob.glob(os.path.join(d, 'model_step_*.pt')))
    return c[-1] if c else None

def load_config(d: str) -> PretrainConfig:
    with open(os.path.join(d, 'all_config.yaml')) as f:
        return PretrainConfig(**yaml.safe_load(f))

def load_meta(data_dir: str) -> PuzzleDatasetMetadata:
    with open(os.path.join(data_dir, 'test', 'dataset.json')) as f:
        return PuzzleDatasetMetadata(**yaml.safe_load(f))

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

    if args.gif is None and not args.no_gif:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.gif = f"arc_viz_{ts}.gif"
        if args.auto is None:
            args.auto = args.gif_delay

    if args.checkpoint_file:
        ckpt_file = args.checkpoint_file; ckpt_dir = os.path.dirname(ckpt_file)
    else:
        ckpt_dir = args.checkpoint_dir; ckpt_file = pick_ckpt(ckpt_dir)
        if ckpt_file is None:
            print('[ERROR] No checkpoint files found in checkpoint-dir', file=sys.stderr)
            sys.exit(1)

    cfg = load_config(ckpt_dir)
    meta = load_meta(args.data_dir)
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
        try:
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()
            init_tokens = inp[0]
            init_grid = render(None, init_tokens.cpu(), flat_solution, args.no_color)
            header = f"Initial (index={idx})"
            text = header + "\n" + init_grid
            clean = ansi_re.sub('', text)
            lines = clean.split('\n')
            dummy = Image.new('RGB', (10,10), 'white'); draw = ImageDraw.Draw(dummy)
            lh = (font.getbbox('Hg')[3]-font.getbbox('Hg')[1]) if hasattr(font,'getbbox') else 10
            w = max(draw.textlength(l, font=font) for l in lines)+8
            h = lh*len(lines)+8
            img = Image.new('RGB',(int(w),int(h)),'white'); draw = ImageDraw.Draw(img)
            y=4
            for l in lines:
                draw.text((4,y),l,fill=(0,0,0),font=font); y+=lh
            frames.append(img)
        except Exception:
            pass

    for step in range(1, max_steps+1):
        with torch.inference_mode():
            carry, outputs = hrm(carry=carry, batch=batch)
            logits = outputs['logits']
            pred = torch.argmax(logits, dim=-1)[0].cpu()
            if args.prob_heatmap:
                probs = torch.softmax(logits[0], dim=-1)
                gt = flat_solution
                prob_correct = probs[torch.arange(probs.shape[0]), gt]
                side = int(math.sqrt(prob_correct.shape[0]))
                grid_prob = prob_correct.view(side, side).cpu().numpy()
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
            grad = args.heatmap_gradient or ' .:-=+*#%@'
            L = len(grad)-1 if len(grad)>1 else 1
            lines=[]
            side = grid_prob.shape[0]
            for r in range(side):
                row=[]
                for c in range(side):
                    p=float(grid_prob[r][c]); p=max(0.0,min(1.0,p)); idx=int(round(p*L)); row.append(grad[idx])
                lines.append(''.join(row))
            print('[ProbHeatmap]\n'+'\n'.join(lines))
        if args.gif:
            try:
                from PIL import Image, ImageDraw, ImageFont
                stats_line = f"Step {step}/{max_steps}  correct={correct}/{total} ({pct:.1f}%) +new={newly_correct} Δ={changed}"
                if args.prob_heatmap and grid_prob is not None:
                    grad = args.heatmap_gradient or ' .:-=+*#%@'
                    L = len(grad)-1 if len(grad)>1 else 1
                    heat_lines=[]
                    side = grid_prob.shape[0]
                    for r in range(side):
                        row=[]
                        for c in range(side):
                            p=float(grid_prob[r][c]); p=max(0.0,min(1.0,p)); idx=int(round(p*L)); row.append(grad[idx])
                        heat_lines.append(''.join(row))
                    heat_block='[ProbHeatmap]\n'+'\n'.join(heat_lines)
                    text = stats_line + "\n" + grid_str + "\n" + heat_block
                else:
                    text = stats_line + "\n" + grid_str
                clean_text = ansi_re.sub('', text)
                font = ImageFont.load_default()
                lines = clean_text.split('\n')
                dummy = Image.new('RGB', (10,10), 'white')
                draw = ImageDraw.Draw(dummy)
                line_height = (font.getbbox('Hg')[3]-font.getbbox('Hg')[1]) if hasattr(font, 'getbbox') else 10
                width = max(draw.textlength(l, font=font) for l in lines) + 8
                height = line_height * len(lines) + 8
                img = Image.new('RGB', (int(width), int(height)), 'white')
                draw = ImageDraw.Draw(img)
                y = 4
                for l in lines:
                    draw.text((4, y), l, fill=(0,0,0), font=font)
                    y += line_height
                frames.append(img)
            except Exception as e:  # noqa: BLE001
                if not frames:
                    print(f"[WARN] Failed to render GIF frame: {e}")
        prev_pred = pred.clone()
        if correct == total:
            print('[COMPLETE]'); break
        if step < max_steps and auto_delay is not None:
            time.sleep(auto_delay)

    if correct != total:
        print("\n[END] Reached max visualization steps.")

    # Save GIF
    if args.gif and frames:
        try:
            from PIL import Image  # noqa: F401
            dur_ms = max(50, int(args.gif_delay * 1000))
            first, *rest = frames
            first.save(args.gif, save_all=True, append_images=rest, duration=dur_ms, loop=0, optimize=False)
            print(f"[INFO] Wrote GIF to {args.gif} ({len(frames)} frames)")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to save GIF: {e}")

if __name__ == '__main__':
    main()
