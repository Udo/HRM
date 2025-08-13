#!/usr/bin/env python
"""Minimal forward smoke test for HierarchicalReasoningModel_ACTV1.

Creates a tiny config, runs a single forward pass and prints tensor shapes.
Usage:
  python scripts/run_model_smoke.py
or
  ./scripts/run_model_smoke.py (after chmod +x)
"""
from __future__ import annotations
import torch
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config


def main():
    cfg = HierarchicalReasoningModel_ACTV1Config(
        batch_size=2, seq_len=8,
        puzzle_emb_ndim=0, num_puzzle_identifiers=1, vocab_size=32,
        H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
        hidden_size=64, expansion=2.0, num_heads=4,
        pos_encodings='rope', halt_max_steps=1, halt_exploration_prob=0.0,
    )

    model = HierarchicalReasoningModel_ACTV1(cfg.model_dump())
    model.eval()

    with torch.no_grad():
        batch = {
            'inputs': torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len)),
            'labels': torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len)),
            'puzzle_identifiers': torch.zeros(cfg.batch_size, dtype=torch.long),
        }
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        print('Logits shape:', outputs['logits'].shape)
        for k, v in outputs.items():
            if hasattr(v, 'shape'):
                print(f"Output[{k}] shape = {tuple(v.shape)}")
        print('Steps tensor shape:', new_carry.steps.shape)
        print('Halted tensor shape:', new_carry.halted.shape)

    print('Smoke test complete.')

if __name__ == '__main__':
    main()
