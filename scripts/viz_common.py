"""Shared utilities for HRM puzzle visualizer CLI scripts.

Centralizes repeated tasks:
  - Checkpoint discovery & config / metadata loading
  - Probability heatmap computation & ASCII rendering
  - GIF text frame rendering (ANSI strip + monospace layout)
  - Standard argparse additions for common flags

All helpers are intentionally lightweight (no heavy imports at module import
besides stdlib + minimal third-party) to keep individual CLIs snappy.
"""
from __future__ import annotations
import glob, os, re, datetime, yaml
from typing import Optional, Sequence, List
import torch
import numpy as np

try:  # Attempt optional PIL import
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _PIL_AVAILABLE = True
except Exception:  # noqa: BLE001
    Image = ImageDraw = ImageFont = None  # type: ignore
    _PIL_AVAILABLE = False

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
DEFAULT_GRADIENT = ' .:-=+*#%@'

# ---------------------------------------------------------------------------
# Checkpoint / config / metadata helpers
# ---------------------------------------------------------------------------

def pick_latest_checkpoint(dir_path: str) -> Optional[str]:
    ckpts = sorted(glob.glob(os.path.join(dir_path, 'model_step_*.pt')))
    return ckpts[-1] if ckpts else None

def find_latest_checkpoint_recursive(root: str = 'checkpoints') -> Optional[str]:
    if not os.path.isdir(root):
        return None
    # Gather both new and legacy naming
    matches = []
    for pattern in ('model_step_*.pt', 'step_*'):
        matches.extend(glob.glob(os.path.join(root, '**', pattern), recursive=True))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def load_yaml_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_all_config(checkpoint_dir: str, cfg_cls):  # cfg_cls = PretrainConfig
    cfg_path = os.path.join(checkpoint_dir, 'all_config.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_cls(**load_yaml_config(cfg_path))


def load_dataset_metadata(data_dir: str, md_cls):  # md_cls = PuzzleDatasetMetadata
    meta_path = os.path.join(data_dir, 'test', 'dataset.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    return md_cls(**load_yaml_config(meta_path))

# ---------------------------------------------------------------------------
# Probability heatmap helpers
# ---------------------------------------------------------------------------

def compute_prob_correct(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-position probability assigned to the ground-truth label.

    Args:
        logits: (Seq, V) or (B, Seq, V)
        labels: (Seq,) ground-truth tokens
    Returns: (Seq,) probabilities in [0,1]
    """
    if logits.dim() == 3:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)
    ar = torch.arange(probs.shape[0], device=probs.device)
    return probs[ar, labels]


def prob_grid_to_ascii(prob_grid: np.ndarray, gradient: str = DEFAULT_GRADIENT) -> str:
    grad = gradient or DEFAULT_GRADIENT
    L = len(grad) - 1 if len(grad) > 1 else 1
    lines = []
    for r in range(prob_grid.shape[0]):
        row_chars = []
        for c in range(prob_grid.shape[1]):
            p = float(prob_grid[r, c])
            p = 0.0 if p < 0 else (1.0 if p > 1 else p)
            idx = int(round(p * L))
            row_chars.append(grad[idx])
        lines.append(''.join(row_chars))
    return '[ProbHeatmap]\n' + '\n'.join(lines)

# ---------------------------------------------------------------------------
# GIF frame rendering
# ---------------------------------------------------------------------------

def make_text_image_block(text: str):
    """Render multiline text (ANSI stripped) to a PIL Image. Returns None if PIL missing."""
    if not _PIL_AVAILABLE:
        return None
    clean = ANSI_RE.sub('', text)
    font = ImageFont.load_default()  # type: ignore[operator]
    lines = clean.split('\n')
    dummy = Image.new('RGB', (10, 10), 'white'); draw = ImageDraw.Draw(dummy)  # type: ignore[operator]
    line_h = (font.getbbox('Hg')[3]-font.getbbox('Hg')[1]) if hasattr(font, 'getbbox') else 10
    width = max(draw.textlength(l, font=font) for l in lines) + 8
    height = line_h * len(lines) + 8
    img = Image.new('RGB', (int(width), int(height)), 'white'); draw = ImageDraw.Draw(img)  # type: ignore[operator]
    y = 4
    for l in lines:
        draw.text((4, y), l, fill=(0,0,0), font=font)
        y += line_h
    return img

def append_text_frame(frames: list, text: str):
    """Create a text image frame and append to list if possible.

    Returns True if frame appended, else False.
    """
    img = make_text_image_block(text)
    if img is not None:
        frames.append(img)
        return True
    return False

def save_gif(path: str, frames: list, delay_sec: float):
    """Save list of PIL frames as animated GIF.

    delay_sec converted to ms; minimum 50ms for visibility. Silently no-op if PIL missing.
    """
    if not _PIL_AVAILABLE or not frames:
        return False
    try:  # type: ignore[attr-defined]
        from PIL import Image  # noqa: F401
        first, *rest = frames
        duration_ms = max(50, int(delay_sec * 1000))
        first.save(path, save_all=True, append_images=rest, duration=duration_ms, loop=0, optimize=False)
        return True
    except Exception:  # noqa: BLE001
        return False

# ---------------------------------------------------------------------------
# Common argument helpers
# ---------------------------------------------------------------------------

def derive_auto_gif_path(prefix: str, args, delay_attr: str = 'gif_delay', auto_attr: str = 'auto'):
    if getattr(args, 'gif', None) is None and not getattr(args, 'no_gif', False):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        setattr(args, 'gif', f"{prefix}_{ts}.gif")
        if getattr(args, auto_attr) is None:
            setattr(args, auto_attr, getattr(args, delay_attr))

__all__ = [
    'pick_latest_checkpoint', 'load_all_config', 'load_dataset_metadata',
    'compute_prob_correct', 'prob_grid_to_ascii', 'make_text_image_block',
    'append_text_frame', 'save_gif', 'derive_auto_gif_path', 'DEFAULT_GRADIENT',
    'find_latest_checkpoint_recursive'
]
