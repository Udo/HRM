from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader, DEFAULT_DEVICE


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    cfg_dir = os.path.dirname(eval_cfg.checkpoint)
    cfg_file = os.path.join(cfg_dir, "all_config.yaml")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            config = PretrainConfig(**yaml.safe_load(f))
    else:
        # Fallback: construct minimal config (best-effort) from checkpoint tensor shapes
        state = torch.load(eval_cfg.checkpoint, map_location=DEFAULT_DEVICE)
        # Try unwrapping compile prefix
        if not any(k.startswith('inner.') for k in state.keys()):
            state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        # Infer vocab size from lm_head weight
        vocab_size = None
        for k,v in state.items():
            if k.endswith('lm_head.weight'):
                vocab_size = v.shape[0]
                hidden = v.shape[1]
                break
        if vocab_size is None:
            raise FileNotFoundError("all_config.yaml missing and could not infer vocab size")
        # Build a minimal arch & training config
        from pretrain import ArchConfig, LossConfig
        arch = ArchConfig(name='hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1', loss=LossConfig(name='losses@ACTLossHead'))
        config = PretrainConfig(
            arch=arch,
            data_path='data/tiny',
            global_batch_size=2,
            epochs=1,
            lr=1e-3, lr_min_ratio=0.1, lr_warmup_steps=1,
            weight_decay=0.0, beta1=0.9, beta2=0.95,
            puzzle_emb_lr=1e-3, puzzle_emb_weight_decay=0.0,
            eval_interval=1,
        )
    config.eval_save_outputs = eval_cfg.save_outputs
    config.checkpoint_path = cfg_dir

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    map_loc = DEFAULT_DEVICE
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location=map_loc), assign=True)
    except Exception:  # noqa: BLE001
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location=map_loc).items()}, assign=True)
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("model_step_") and ckpt_filename.endswith('.pt'):
        train_state.step = int(ckpt_filename.removeprefix("model_step_").removesuffix('.pt'))
    elif ckpt_filename.startswith("step_"):
        # legacy naming
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print (metrics)


if __name__ == "__main__":
    launch()
