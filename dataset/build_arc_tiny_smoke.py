#!/usr/bin/env python
"""Build a tiny synthetic ARC-like dataset for smoke tests.
Creates minimal train/test splits with a handful of random puzzles so that
end-to-end training & evaluation can run without the real ARC datasets.

Usage:
  python dataset/build_arc_tiny_smoke.py --output-dir data/arc-tiny-smoke

The produced structure matches puzzle_dataset.py expectations:
  <output-dir>/train/{all__inputs.npy, all__labels.npy, ... , dataset.json}
  <output-dir>/test/{...}

Notes:
- Sequence length fixed at 30x30=900 (like real ARC builder after padding).
- Vocab size 12 (PAD=0, EOS=1, digits 0..9 mapped to 2..11) consistent with build_arc_dataset.
- ignore_label_id = 0 to mirror real builder so padding & EOS (0) get mapped to IGNORE_LABEL_ID.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import sys, pathlib

# Allow running from repository root without installing as a package.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.common import PuzzleDatasetMetadata

SEQ_LEN = 30 * 30
VOCAB_SIZE = 12  # PAD + EOS + 0..9
PAD_ID = 0
IGNORE_LABEL_ID = 0
BLANK_IDENTIFIER_ID = 0

# Tiny config
TRAIN_PUZZLES = 3
TEST_PUZZLES = 1
EXAMPLES_PER_PUZZLE = 2

rng = np.random.default_rng(1234)

def _make_split(num_puzzles: int):
    # Each puzzle has EXAMPLES_PER_PUZZLE examples.
    total_examples = num_puzzles * EXAMPLES_PER_PUZZLE
    inputs = rng.integers(0, VOCAB_SIZE, size=(total_examples, SEQ_LEN), dtype=np.int32)
    labels = rng.integers(0, VOCAB_SIZE, size=(total_examples, SEQ_LEN), dtype=np.int32)

    # puzzle_indices: starting index for each puzzle, plus final sentinel
    puzzle_indices = [i * EXAMPLES_PER_PUZZLE for i in range(num_puzzles)] + [total_examples]
    puzzle_indices = np.array(puzzle_indices, dtype=np.int32)

    # puzzle_identifiers: 0 is blank, assign 1..num_puzzles
    puzzle_ids = []
    for pid in range(num_puzzles):
        puzzle_ids.extend([pid + 1] * EXAMPLES_PER_PUZZLE)
    puzzle_identifiers = np.array(puzzle_ids, dtype=np.int32)

    # Treat each puzzle as its own group
    group_indices = np.array(list(range(num_puzzles)) + [num_puzzles], dtype=np.int32)

    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }, num_puzzles, total_examples


def build(output_dir: str):
    for split_name, num_puzzles in ("train", TRAIN_PUZZLES), ("test", TEST_PUZZLES):
        data, puzzles, examples = _make_split(num_puzzles)
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for k, v in data.items():
            np.save(os.path.join(split_dir, f"all__{k}.npy"), v)

        metadata = PuzzleDatasetMetadata(
            pad_id=PAD_ID,
            ignore_label_id=IGNORE_LABEL_ID,
            blank_identifier_id=BLANK_IDENTIFIER_ID,
            vocab_size=VOCAB_SIZE,
            seq_len=SEQ_LEN,
            num_puzzle_identifiers=num_puzzles + 1,  # + blank
            total_groups=puzzles,
            mean_puzzle_examples=EXAMPLES_PER_PUZZLE,
            sets=["all"],
        )
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

    print(f"[arc_tiny_smoke] Wrote synthetic dataset to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    build(args.output_dir)

if __name__ == "__main__":
    main()
