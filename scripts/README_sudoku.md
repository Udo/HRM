# Sudoku Scripts (Concise)

This directory provides thin wrappers around building, training, evaluating, and visualizing Sudoku experiments. All generic concepts (architecture overrides, environment toggles, visualization features, dataset field semantics) are centralized in the root `README.md` and `scripts/README.md` to avoid duplication.

Primary helpers:
* `build_sudoku_1k.sh` – Build ~1k augmented extreme subset.
* `run_sudoku_pipeline.sh` – End‑to‑end build→train→eval/benchmark (idempotent with `SKIP_BUILD=1`).
* `train_sudoku_small.sh` – Example longer training profile.
* `viz_sudoku_latest.sh` / `visualize_sudoku_cli.py` – Step reasoning + GIF (prob heatmap on by default).

Typical invocations:
```
# Full pipeline (defaults)
./scripts/run_sudoku_pipeline.sh

# Faster smoke: fewer epochs, subset, checkpoint each eval
EPOCHS=200 EVAL_INTERVAL=20 SUBSAMPLE=400 AUG=2 CKPT_EVERY=1 ./scripts/run_sudoku_pipeline.sh

# Architecture / loss overrides after -- (Hydra syntax)
./scripts/run_sudoku_pipeline.sh -- arch.hidden_size=384 arch.num_heads=6 arch.loss.loss_type=softmax_cross_entropy

# Visualization (auto-detect latest checkpoint & dataset)
./scripts/viz_sudoku_latest.sh --max-steps 6
```

Environment variables & dataset encoding: see root `README.md` (Sudoku section) for token scheme and augmentation description; see `scripts/README.md` for global script/env tables.

Notes:
* Default loss: softmax cross entropy (set `arch.loss.loss_type=stablemax_cross_entropy` to experiment).
* Stability toggles: `HRM_CLIP_GRAD`, `HRM_QK_CLAMP`, NaN guards (in training loop) – centrally documented.
* Probability heatmap enabled unless `--no-prob-heatmap` passed.

Minimal tips:
* Increase `AUG` for more surface diversity; raise `SUBSAMPLE` toward full 1k set.
* Use visualization early to inspect step‑wise fill‑in dynamics; exact board correctness often achieved before final epoch.

## Dataset Snapshot
```
Layout: train/ & test/ each with all__inputs.npy, all__labels.npy, all__puzzle_indices.npy, all__group_indices.npy, all__puzzle_identifiers.npy, dataset.json
Seq len: 81 (flatten 9x9). Vocab: 0=PAD, 1..10 = digits 0..9 (blank '.' -> token 1 after +1 shift).
Inputs: unsolved grid (+1 shift). Labels: solved grid (+1 shift). Identifiers all zero (single puzzle class by default).
Augment (train): digit permutation, band/stack shuffle, row/col within band/stack, optional transpose.
Metrics: accuracy (token), exact_accuracy (whole board), steps (avg ACT halting steps).
```
Full detail: see Sudoku section in root README.
