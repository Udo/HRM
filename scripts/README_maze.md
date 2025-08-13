# Maze Scripts (Concise)

Lightweight wrappers for building, training, and visualizing Maze experiments. Common environment variable semantics, stability toggles, visualization behavior, and dataset field definitions live in the root `README.md` & `scripts/README.md`.

Core helpers:
* `build_maze_1k.sh` – Build canonical 1k maze dataset (optional subsample & dihedral aug).
* `run_maze_pipeline.sh` – End‑to‑end build→train→eval/benchmark (skip build with `SKIP_BUILD=1`).
* `train_maze_small.sh` – Baseline long run config (~800 epochs / eval 80).
* `train_maze_tiny.sh` – Fast smoke / CI config.
* `viz_maze_latest.sh` / `visualize_maze_cli.py` – Step visualization + GIF; supports stochastic sampling.
* `repro_maze_result.sh` – Single-run Maze JSON result summary (paper-style comparable fields).
* `repro_maze_multi_seed.sh` – Multi-seed orchestration + aggregate JSON (+ optional CSV).

Examples:
```
# Default full 1k pipeline
./scripts/run_maze_pipeline.sh

# Subsample with augmentation & save ckpt each eval
SUBSAMPLE=200 AUG=1 CKPT_EVERY=1 ./scripts/run_maze_pipeline.sh

# Skip rebuild
SKIP_BUILD=1 ./scripts/run_maze_pipeline.sh

# Architecture overrides
./scripts/run_maze_pipeline.sh -- arch.hidden_size=384 arch.num_heads=12

# Tiny quick test
./scripts/train_maze_tiny.sh
```

Dataset encoding & augmentation details: refer to the Maze section in the root `README.md` (now consolidated). Script/global env variable tables: `scripts/README.md`.

Notes:
* Default loss currently `stablemax_cross_entropy` for Maze (empirically stable); override with `arch.loss.loss_type=softmax_cross_entropy`.
* Probability heatmap enabled by default; disable via `--no-prob-heatmap`.
* Sampling: add `--sample-temp <float>` to explore diverse path hypotheses during visualization.
* Cleanup old artifacts: `./scripts/cleanup_artifacts.sh --visuals` (dry run) then add `--apply`.

Minimal mental model: Flattened grid tokens -> embedding (+optional puzzle id) -> hierarchical cycles refine -> logits per cell -> cross entropy vs target path/overlay tokens (pad ignored) with ACT controlling compute depth.

## Dataset Snapshot
```
Layout: train/ & test/ with all__inputs.npy, all__labels.npy, all__puzzle_indices.npy, all__group_indices.npy, all__puzzle_identifiers.npy, dataset.json
Seq len: 900 (30x30 flatten). Vocab: 0=PAD, 1..K semantic cells (wall, empty, start, goal, path...). Position implicit.
Targets: path or reconstructed grid (pad ignored). Grouping indices retained for consistency with multi-example domains.
Augment (optional): dihedral transforms (+ potential channel/color perm if enabled).
Metrics: accuracy (token), exact_accuracy (entire grid), steps (mean ACT steps).
```
See Maze section in root README for extended explanation.

