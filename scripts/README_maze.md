# Maze Pipeline Scripts

This directory now includes maze dataset pipeline helpers.

## Files
- `build_maze_1k.sh` – Wraps `dataset/build_maze_dataset.py` with optional subsample & dihedral augmentation.
- `train_maze_small.sh` – Baseline training config (≈800 epochs, eval every 80).
- `train_maze_tiny.sh` – Very small fast config for smoke tests / CI.
- `run_maze_pipeline.sh` – End‑to‑end: (optionally) build dataset, train, eval summary, benchmark.

## Usage Examples
Build + train + eval (default full 1k set):
```
./scripts/run_maze_pipeline.sh
```
Subsample 200 puzzles with augmentation and keep checkpoints every eval:
```
SUBSAMPLE=200 AUG=1 CKPT_EVERY=1 ./scripts/run_maze_pipeline.sh
```
Skip rebuild if dataset already present:
```
SKIP_BUILD=1 ./scripts/run_maze_pipeline.sh
```
Override architecture (Hydra style) after `--`:
```
./scripts/run_maze_pipeline.sh -- arch.hidden_size=384 arch.num_heads=12
```
Tiny smoke test training only:
```
./scripts/train_maze_tiny.sh
```

## Environment Variables Summary
| Var | Purpose | Default |
|-----|---------|---------|
| DATASET_DIR | Output dataset directory | data/maze-30x30-hard-1k |
| SUBSAMPLE | Limit training samples | (full) |
| AUG | Apply dihedral augmentation | 0 |
| SKIP_BUILD | Skip dataset build step | 0 |
| EPOCHS | Total epochs | 800 |
| EVAL_INTERVAL | Eval interval epochs | 80 |
| GBS | Global batch size | 128 |
| HIDDEN, HEADS, H/L_LAYERS, H/L_CYC | Architecture sizes | (see scripts) |
| LR / PUZ_LR | Learning rates | 5e-4 |
| WD / PUZ_WD | Weight decay | 0.1 |
| CKPT_EVERY | Save ckpt each eval | 0 |
| SUMMARY | Run eval_summary | 1 |
| BENCHMARK | Run benchmark eval | 1 |

## Notes
- Loss forced to `stablemax_cross_entropy` for numerical stability on maze.
- `DISABLE_COMPILE=1` and sparse emb optimizer disabled by default for fast iteration on Apple Silicon.
- Adjust `HRM_DISABLE_WANDB=0` to enable Weights & Biases if configured.
- For interactive reasoning inspection & GIFs, see `visualize_maze_cli.py` or run `./scripts/viz_maze_latest.sh` (auto GIF). Use `--sample-temp` to introduce stochastic exploration.
- To clean old maze GIFs & checkpoints: `./scripts/cleanup_artifacts.sh --visuals` (dry) then add `--apply`.
- Probability heatmap (per-cell correctness) is on by default; disable with `--no-prob-heatmap`.
