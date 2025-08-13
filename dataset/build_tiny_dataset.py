"""Create a minimal synthetic dataset for fast smoke tests on any device (CPU/MPS/CUDA).

This avoids downloading external data and keeps evaluation very small.
"""
from __future__ import annotations
import os
import json
import numpy as np
from dataclasses import asdict, dataclass


@dataclass
class TinyMetadata:
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: int
    sets: list[str]

    def model_dump(self):  # mimic pydantic API used elsewhere
        return asdict(self)


def make_split(root: str, split: str, num_puzzles: int):
    # Following sudoku encoding convention: symbols 0..9 shifted by +1 => 1..10, pad=0
    seq_len = 81
    vocab_size = 11  # 0 pad + 1..10 digits

    rng = np.random.default_rng(42 + (0 if split == "train" else 1))

    inputs = rng.integers(low=1, high=11, size=(num_puzzles, seq_len), dtype=np.int32)
    labels = rng.integers(low=1, high=11, size=(num_puzzles, seq_len), dtype=np.int32)

    # Each puzzle is its own group; single example per puzzle.
    puzzle_indices = np.arange(num_puzzles + 1, dtype=np.int32)
    group_indices = np.arange(num_puzzles + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros((num_puzzles,), dtype=np.int32)

    split_dir = os.path.join(root, split)
    os.makedirs(split_dir, exist_ok=True)

    meta = TinyMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num_puzzles,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(meta.model_dump(), f)

    np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(split_dir, "all__labels.npy"), labels)
    np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)

    with open(os.path.join(root, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/tiny")
    p.add_argument("--train-size", type=int, default=4)
    p.add_argument("--test-size", type=int, default=2)
    args = p.parse_args()

    make_split(args.output_dir, "train", args.train_size)
    make_split(args.output_dir, "test", args.test_size)
    print(f"Tiny dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
