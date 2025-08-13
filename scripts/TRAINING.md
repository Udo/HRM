## Training & Model Size Guide

This document summarizes how to configure Hierarchical Reasoning Model (HRM) sizes for training.

### Core Architecture Hyperparameters

Overridable via Hydra (e.g. `arch.hidden_size=256 arch.num_heads=8`):

| Name | Meaning | Notes |
|------|---------|-------|
| `hidden_size` | Transformer model width (H) | Must be divisible by `num_heads`. Dominates params & compute. |
| `num_heads` | Attention heads | Only constrained by `hidden_size % num_heads == 0`. Changing this alone (fixed H) does NOT change parameter count. |
| `H_layers`, `L_layers` | High / low level block counts | Add linearly to parameter count. |
| `H_cycles`, `L_cycles` | Iterative reasoning cycles | Increase compute only (no new params). |
| `expansion` | SwiGLU expansion factor (default 4) | Controls MLP width; large impact on params. |
| `puzzle_emb_ndim` | Puzzle identifier embedding dim | Rounded up internally; adds sparse emb params if >0. |
| `halt_max_steps`, `halt_exploration_prob` | ACT controls | Runtime loop length; no parameter impact. |
| `forward_dtype` | Forward precision | `bfloat16` default (lower memory). |

Dataset‑derived: `vocab_size`, `seq_len`, `num_puzzle_identifiers`.

### Parameter Count Approximation

Let H=`hidden_size`, L=`H_layers+L_layers`, V=`vocab_size`, E=`expansion`.
SwiGLU inter dim: `inter ≈ multiple_256(round(E * H * 2/3))`.

Per block params ≈ `(4 + 3 * inter / H) * H^2`.
For `expansion=4`, `inter ≈ (8/3)H` → per block ≈ `12 * H^2`.

Total dominant params ≈ `L * 12 * H^2 + 2 * V * H + puzzle_emb + 2H`.

Cycles do NOT add parameters.

### Preset Examples (≈ values)

Assuming V=11, `num_puzzle_identifiers=1`, `puzzle_emb_ndim=H`.

| Preset | H | Heads | H_layers | L_layers | Cycles (H,L) | Approx Params |
|--------|---|-------|----------|----------|--------------|---------------|
| Tiny Demo | 128 | 4 | 1 | 1 | (1,1) | ~0.40M |
| Small Sudoku | 256 | 8 | 2 | 2 | (2,2) | ~3.2M |
| Base Default | 512 | 8 | 4 | 4 | (2,2) | ~25M |

> Counts are approximate; final numbers vary slightly due to padding of intermediate dims.

### Scaling Guidance
1. Depth (layers) increases params linearly (≈ `12 H^2` per added block). Add balanced H/L pairs for symmetry.
2. Width (hidden_size) increases parameters quadratically; use after exhausting depth if memory allows.
3. Expansion lowers/raises MLP cost; `expansion=3` saves ~25% MLP params vs 4.
4. Cycles let you trade extra compute for potentially better reasoning without increasing parameter count.
5. `num_heads` change alone doesn’t change parameter count; tune for performance characteristics.

### Example Overrides
Small custom (~7–8M params):
```
arch.hidden_size=384 arch.num_heads=6 arch.H_layers=3 arch.L_layers=3
```

Larger (~85M params exploratory):
```
arch.hidden_size=768 arch.num_heads=12 arch.H_layers=6 arch.L_layers=6
```

Compute scaling (same params):
```
arch.H_cycles=4 arch.L_cycles=4
```

### Memory & Performance
| Aspect | Approx Scaling |
|--------|----------------|
| Activation Memory | O(batch * seq_len * H * cycles) |
| Params | O(L * H^2) |
| FLOPs | O((1 + cycles) * L * H^2) |

### Quick Param Estimator
```python
def approx_params(H=512, H_layers=4, L_layers=4, vocab=11, expansion=4, puzzle_emb=True, num_ids=1):
    inter = ((round(expansion * H * 2 / 3) + 255)//256)*256
    per_block = (4 + 3*inter/H) * H * H
    blocks = (H_layers + L_layers) * per_block
    emb = 2 * vocab * H
    puzzle = num_ids * (H if puzzle_emb else 0)
    heads = 2 * H
    return round(blocks + emb + puzzle + heads)
print(approx_params())
```

### Choosing a Configuration
| Goal | Suggestion |
|------|------------|
| Smoke test | Tiny demo preset |
| Laptop / MPS iteration | Small Sudoku preset |
| More capacity (single GPU) | Base config (512 width, 4+4 layers) |
| Cheaper compute | Reduce cycles first |
| More reasoning compute | Increase cycles before width |

### Command Examples
Tiny:
```
./scripts/train_tiny_demo.sh EPOCHS=20 EVAL_INTERVAL=5 arch.hidden_size=128 arch.num_heads=4
```
Medium:
```
./scripts/train_tiny_demo.sh arch.hidden_size=384 arch.num_heads=6 arch.H_layers=2 arch.L_layers=2
```
Base:
```
./scripts/train_sudoku_small.sh arch.hidden_size=512 arch.H_layers=4 arch.L_layers=4
```

### Constraints
* `hidden_size % num_heads == 0`
* Fits available device memory (watch batch_size * seq_len * H)

