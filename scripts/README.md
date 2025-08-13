# Scripts Quick Reference

Dense overview of helper scripts and key environment toggles for fast iteration.

## Script Index
| Script | Purpose |
|--------|---------|
| `prepare_env.sh` | Create & populate Python venv (`.venv`) from `requirements.txt` (idempotent). |
| `build_tiny_dataset.sh` | Produce a minimal synthetic dataset (tiny smoke tests). |
| `build_sudoku_1k.sh` | Build a ~1k (with augmentation) Sudoku dataset subset. |
| `train_tiny_demo.sh` | End‑to‑end tiny training loop for correctness & speed. |
| `train_sudoku_small.sh` | Example training on Sudoku subset. |
| `eval_checkpoint.sh` | Evaluate a saved checkpoint on a dataset split. |
| `eval_latest.sh` | Run evaluation on the newest checkpoint (auto-detect). |
| `eval_summary.sh` | Thin shell wrapper calling `eval_summary.py` for structured metrics + throughput. |
| `eval_summary.py` | Python implementation producing metrics & summary lines (no inline Python in bash). |
| `benchmark_eval.sh` | Wrapper for `benchmark_eval.py` throughput benchmarking. |
| `benchmark_eval.py` | Python benchmark; measures duration and heuristic throughput. |
| `param_count.py` | Estimate parameter count for arbitrary architecture overrides. |
| `viz_sudoku_latest.sh` | Auto-pick latest Sudoku checkpoint & render steps + GIF. |
| `viz_maze_latest.sh` | Auto-pick latest Maze checkpoint & render steps + GIF (supports sampling). |
| `viz_arc_latest.sh` | Auto-pick latest ARC checkpoint & render steps + GIF. |
| `visualize_sudoku_cli.py` | Core Sudoku step visualizer (color diff + GIF). |
| `visualize_maze_cli.py` | Core Maze step visualizer (diff stats, sampling, GIF). |
| `visualize_arc_cli.py` | Core ARC step visualizer (diff stats + GIF). |
| `cleanup_artifacts.sh` | Dry-run checkpoint / GIF cleanup (with protection & apply). |
| (all visualizers) | Probability heatmap now enabled by default; disable with `--no-prob-heatmap`. |

## Core Environment Variables (Feature Toggles)
| Var | Default | Effect |
|-----|---------|--------|
| `HRM_DEVICE` | auto | Force `cuda`, `mps`, or `cpu` (overrides auto‑detect). |
| `DISABLE_COMPILE` | `1` | Skip `torch.compile` (enable with `0` on CUDA only). |
| `HRM_DISABLE_WANDB` | `1` in demos | Disable Weights & Biases logging. Set `0` to enable if `wandb` creds exist. |
| `HRM_DISABLE_SPARSE_EMB_OPTIMIZER` | `1` in demos | Bypass custom sparse SignSGD optimizer (stability on MPS/CPU). |
| `CKPT_EVERY` (`train_tiny_demo.sh`) | `0` | If `1`, enable `checkpoint_every_eval=true` to save after each eval. |
| `OUTPUT_DIR` | script specific | Destination dataset/checkpoints root for tiny run. |

## Training Overrides (Hydra Args)
All scripts pass overrides directly to `pretrain.py` using Hydra syntax. Common ones:
- `epochs=<int>` total training epochs.
- `eval_interval=<int>` divisor of `epochs` (must evenly divide) to trigger evaluation cycles.
- `global_batch_size=<int>` effective batch size.
- Architecture overrides: `arch.hidden_size=`, `arch.num_heads=`, `arch.H_layers=`, `arch.L_layers=`, `arch.H_cycles=`, `arch.L_cycles=`.
- Optimization: `lr=`, `puzzle_emb_lr=`, `weight_decay=`, `puzzle_emb_weight_decay=`.
- Loss selection: `arch.loss.loss_type=stablemax_cross_entropy|softmax_cross_entropy`.

## Fast Start (Tiny Demo)
```
./scripts/prepare_env.sh
./scripts/train_tiny_demo.sh            # 50 epochs (default) tiny smoke test
EPOCHS=10 EVAL_INTERVAL=5 ./scripts/train_tiny_demo.sh
CKPT_EVERY=1 EPOCHS=10 EVAL_INTERVAL=5 ./scripts/train_tiny_demo.sh  # save every eval
```
Add Hydra overrides at the end:
```
EPOCHS=12 EVAL_INTERVAL=6 ./scripts/train_tiny_demo.sh arch.hidden_size=256 arch.num_heads=8
```

## Checkpoints
Automatically written under (new naming):
```
checkpoints/<ProjectName>/<RunName>/model_step_<train_step>.pt
```
Legacy runs may have: `step_<train_step>` (without extension). Prediction dumps are stored as `step_<train_step>_all_preds.<rank>`.

Enable per‑eval saves with `CKPT_EVERY=1` (sets Hydra flag `checkpoint_every_eval=true`). Final step always saved.

## Evaluation
Direct:
```
./scripts/eval_checkpoint.sh checkpoints/"Tiny ACT-torch"/<RunName>/model_step_10.pt
```
Latest automatically:
```
./scripts/eval_latest.sh
```
Summary (includes throughput if dataset inferable):
```
./scripts/eval_summary.sh                 # auto-picks latest
./scripts/eval_summary.sh checkpoints/.../model_step_10.pt
```

## macOS / MPS Notes
- Float64 is avoided on MPS (stablemax uses float32 internally now).
- FlashAttention gracefully falls back to PyTorch SDPA or manual attention when unavailable.
- Custom sparse embedding optimizer disabled by default on non‑CUDA backends (`HRM_DISABLE_SPARSE_EMB_OPTIMIZER=1`).

## Typical Workflow
1. `./scripts/prepare_env.sh`
2. Build dataset (tiny or sudoku): `./scripts/build_tiny_dataset.sh` or `./scripts/build_sudoku_1k.sh`
3. Train: `./scripts/train_tiny_demo.sh` (tune overrides / env vars)
4. Inspect checkpoints in `checkpoints/...`
5. Evaluate: `./scripts/eval_checkpoint.sh <ckpt> <data_dir>`

## Troubleshooting Quick Hits
| Symptom | Likely Fix |
|---------|------------|
| Device mismatch (CPU vs MPS) | Ensure `HRM_DEVICE` unset or correct; avoid mixing CPU tensors in custom code. |
| Slow first step | Compilation disabled; enable on CUDA: `DISABLE_COMPILE=0`. |
| WandB errors | Set `HRM_DISABLE_WANDB=1` (default in demos). |
| No checkpoints | Use `CKPT_EVERY=1` or ensure final epoch reached (check `epochs` & `eval_interval` divisibility). |
| GIF not produced | Pass explicit `--gif out.gif` or ensure not using `--no-gif` (helpers auto-create by default). |
| Wrong domain picked by viz script | Ensure dataset path contains domain keyword (Sudoku helper filters for 'sudoku'). |
| Want to prune artifacts | Use `./scripts/cleanup_artifacts.sh` (dry run first). |

## Minimal Reference (Tiny Script Variables)
```
OUTPUT_DIR=data/tiny
EPOCHS=50
EVAL_INTERVAL=10
GBS=16
HIDDEN=128 HEADS=4
H_LAYERS=1 L_LAYERS=1
H_CYC=1 L_CYC=1
LR=1e-3 PUZ_LR=1e-3
CKPT_EVERY=0
```
Override any via env export or inline prefix.

