import os
import torch

from pretrain import PretrainConfig, ArchConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config


def build_inner_config():
    return HierarchicalReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=8,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=32,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=1,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
    )


def test_forward_smoke():
    inner_cfg = build_inner_config()
    model = HierarchicalReasoningModel_ACTV1(inner_cfg.model_dump())
    batch = {
        "inputs": torch.randint(0, inner_cfg.vocab_size, (inner_cfg.batch_size, inner_cfg.seq_len)),
        "labels": torch.randint(0, inner_cfg.vocab_size, (inner_cfg.batch_size, inner_cfg.seq_len)),
        "puzzle_identifiers": torch.zeros(inner_cfg.batch_size, dtype=torch.long),
    }
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    assert "logits" in outputs
    assert outputs["logits"].shape == (inner_cfg.batch_size, inner_cfg.seq_len, inner_cfg.vocab_size)
