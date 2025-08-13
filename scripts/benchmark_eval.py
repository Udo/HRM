#!/usr/bin/env python
"""Benchmark evaluation throughput for a checkpoint.

Usage:
  python scripts/benchmark_eval.py --checkpoint CKPT [--data-path DATA]

Prints: duration, approximate items processed (heuristic), throughput.
"""
from __future__ import annotations
import argparse
import time
import subprocess
import sys
import re
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', '-c', required=True)
    ap.add_argument('--data-path')
    args, extra = ap.parse_known_args()

    start = time.time()
    proc = run([sys.executable, 'evaluate.py', f'checkpoint={args.checkpoint}', *extra])
    end = time.time()
    if proc.returncode != 0:
        print('[ERR] evaluate.py failed', file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return 3

    duration = end - start
    # Heuristic items: count of "'count':" occurrences (aggregated metrics may show once)
    items = len(re.findall(r"'count':", proc.stdout)) or 1

    print(f'Duration(s): {duration}')
    print(f'Approx items processed: {items}')
    print(f'Throughput ~ {items/duration:.2f} items/s')
    print('\n--- Tail Output ---')
    tail_lines = proc.stdout.strip().splitlines()[-30:]
    print('\n'.join(tail_lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
