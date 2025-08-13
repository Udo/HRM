# ARC Scripts (Concise)

Wrappers for building ARC datasets (ARC‑1 + ConceptARC, optionally ARC‑2), training, evaluation, benchmarking, and visualization. Generic environment toggles, stability features, visualization behavior, and full dataset encoding details are centralized (root `README.md` + `scripts/README.md`).

Key helpers:
* `build_arc_dataset.py` – Full dataset w/ color perm, dihedral, translation + EOS frame.
* `build_arc_tiny_smoke.py` – Synthetic tiny ARC‑like random dataset (smoke tests, no structure).
* `run_arc_pipeline.sh` – End‑to‑end build→train→eval/benchmark (falls back tiny when allowed).
* `repro_arc_result.sh` – Single-run reproducible ARC training + JSON metrics summary (paper-style fields).
* `repro_arc_multi_seed.sh` – Multi-seed orchestrator: runs multiple seeds and aggregates mean/std JSON.
* `viz_arc_latest.sh` / `visualize_arc_cli.py` – Step reasoning + GIF (prob heatmap on).

Examples:
```
# Full pipeline (real data present)
./scripts/run_arc_pipeline.sh

# Allow auto tiny fallback if raw missing
ALLOW_TINY_ARC=1 ./scripts/run_arc_pipeline.sh EPOCHS=2 EVAL_INTERVAL=1

# Architecture / loss overrides
./scripts/run_arc_pipeline.sh -- arch.hidden_size=384 arch.num_heads=6 arch.loss.loss_type=softmax_cross_entropy

# Visualization (auto-detect latest checkpoints)
./scripts/viz_arc_latest.sh --max-steps 6
```

Notes:
* Tiny synthetic dataset activates only when raw data absent AND `ALLOW_TINY_ARC=1`; intended for CI/timeboxed smoke.
* Probability heatmap enabled by default; disable with `--no-prob-heatmap`.
* Stability toggles (grad clip, QK clamp, NaN guards) documented centrally.
* EOS boundary tokens are standard classes (not ignored); they help delineate padded grid region.

For dataset encoding (padding scheme, vocab, augmentation pipeline, grouping semantics) see ARC section in root `README.md`.

## Dataset Snapshot
```
Layout: train/ & test/ with all__inputs.npy, all__labels.npy, all__puzzle_indices.npy, all__group_indices.npy, all__puzzle_identifiers.npy, dataset.json, identifiers.json
Seq len: 900 (30x30 pad). Vocab: 0=PAD, 1=EOS boundary, 2..11 = colors 0..9 (shift +2).
Inputs/labels: flattened grids; train-time random translation + L-shaped EOS border writes token 1 on lower/right boundary of content.
Augment: color permutation (excl background 0) + dihedral (hash dedupe) + translation (train) until num_aug reached.
Grouping: original puzzle + its accepted augs form one group (balanced sampling). puzzle_identifiers gives unique id (0 reserved) enabling optional puzzle embedding.
Metrics: accuracy, exact_accuracy (full seq), steps (ACT halting).
Tiny fallback: random tokens (structureless) only for CI/smoke.
```
