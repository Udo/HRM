from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
try:
    from adam_atan2 import AdamATan2  # type: ignore
except Exception:  # noqa: BLE001
    AdamATan2 = None  # type: ignore

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    use_pin = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=use_pin,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def get_default_device():
    """Select an appropriate default device.

    Priority: CUDA -> MPS (Apple Silicon) -> CPU. Allow override via HRM_DEVICE.
    """
    override = os.environ.get("HRM_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


DEFAULT_DEVICE = get_default_device()


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Instantiate on meta then move to device (safer for large models on CPU first) not strictly needed here
    model: nn.Module = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    model.to(DEFAULT_DEVICE)
    if "DISABLE_COMPILE" not in os.environ and torch.cuda.is_available():  # torch.compile only on torch>=2 and CUDA currently stable
        try:
            model = torch.compile(model, dynamic=False)  # type: ignore
        except Exception:
            pass

    # Broadcast parameters from rank 0
    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = []

    # Sparse embedding optimizer (optional)
    use_sparse_emb_opt = os.environ.get("HRM_DISABLE_SPARSE_EMB_OPTIMIZER") != "1" and hasattr(model.model, "puzzle_emb")  # type: ignore
    if use_sparse_emb_opt:
        try:
            optimizers.append(
                CastedSparseEmbeddingSignSGD_Distributed(
                    model.model.puzzle_emb.buffers(),  # type: ignore
                    lr=0,
                    weight_decay=config.puzzle_emb_weight_decay,
                    world_size=world_size,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Sparse embedding optimizer disabled (fallback to standard) due to: {e}")
            use_sparse_emb_opt = False

    # Main optimizer (all params including puzzle embeddings if sparse disabled)
    main_params = model.parameters() if not use_sparse_emb_opt else [p for n, p in model.named_parameters() if "puzzle_emb" not in n]
    if AdamATan2 is not None:
        optimizers.append(
            AdamATan2(
                main_params,
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        )
    else:
        from torch.optim import AdamW
        optimizers.append(
            AdamW(
                main_params,
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        )
    optimizer_lrs = []
    if use_sparse_emb_opt:
        optimizer_lrs.append(config.puzzle_emb_lr)
    optimizer_lrs.append(config.lr)

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    """Persist model state dict with an unambiguous filename.

    Uses model_step_<N>.pt to distinguish from prediction dump files (which keep step_<N>_all_preds.*).
    """
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    ckpt_file = os.path.join(config.checkpoint_path, f"model_step_{train_state.step}.pt")
    torch.save(train_state.model.state_dict(), ckpt_file)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    # Step & early exit
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    # Move batch to device
    batch = {k: v.to(DEFAULT_DEVICE) for k, v in batch.items()}

    # Initialize carry
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    scaled_loss = loss / global_batch_size
    if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
        # Skip this batch – return metrics with NaN flags for logging
        if rank == 0:
            print(f"[WARN] Skipping batch: invalid scaled loss {scaled_loss}")
        return {"lm_loss": torch.tensor(float('nan')), "count": torch.tensor(global_batch_size)}
    scaled_loss.backward()

    # Optional grad clipping (env toggle)
    try:
        clip_val = float(os.environ.get("HRM_CLIP_GRAD", "0"))
    except ValueError:
        clip_val = 0.0
    if clip_val > 0:
        torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), clip_val)

    # Detect any NaN/Inf gradients – if present, zero them and skip optimizer step
    bad_grad = False
    for p in train_state.model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                bad_grad = True
                break
    if bad_grad:
        if rank == 0:
            print("[WARN] Detected NaN/Inf gradients – zeroing & skipping optimizer step for this batch")
        for p in train_state.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        return {"lm_loss": torch.tensor(float('nan')), "count": torch.tensor(global_batch_size)}

    # Gradient all-reduce if distributed
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Optimizer steps + LR schedule
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    # Metrics reduction
    if metrics:
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        values = torch.stack([metrics[k].to(DEFAULT_DEVICE) for k in metric_keys])
        if world_size > 1:
            dist.reduce(values, dst=0)
        if rank == 0:
            arr = values.detach().cpu().numpy()
            reduced = {k: arr[i] for i, k in enumerate(metric_keys)}
            count = max(reduced.get("count", 1), 1)
            reduced = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced.items()
            }
            reduced["train/lr"] = lr_this_step
            return reduced


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            # To device
            batch = {k: v.to(DEFAULT_DEVICE) for k, v in batch.items()}
            carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                
                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device=DEFAULT_DEVICE)
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    """Always persist current code + config alongside checkpoints.

    WandB presence only affects code logging, not local persistence.
    """
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy model + loss source files (best-effort)
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None and os.path.exists(code_file):
            code_name = os.path.basename(code_file)
            try:
                shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to copy {code_file}: {e}")

    # Dump config as yaml (always)
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    try:
        with open(config_file, "wt") as f:
            yaml.dump(config.model_dump(), f)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to write config file: {e}")

    # Optional wandb code logging
    if wandb.run is not None:
        try:
            wandb.run.log_code(config.checkpoint_path)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] WandB log_code failed: {e}")


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, choose backend based on available device
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    if train_state.total_steps < 1:
        # Clamp to at least one step so progress bar & scheduling behave sensibly.
        if RANK == 0:
            print(f"[WARN] Computed total_steps={train_state.total_steps} (<1) from epochs * dataset_size / global_batch_size; clamping to 1. Consider increasing epochs or reducing global_batch_size.")
        train_state.total_steps = 1

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb_disabled = os.environ.get("HRM_DISABLE_WANDB") == "1"
        if not wandb_disabled:
            try:
                wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
                wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] WandB init failed: {e}. Continuing without logging.")
                os.environ["HRM_DISABLE_WANDB"] = "1"
        # Always persist code + config locally regardless of wandb
        save_code_and_config(config)

    # Console logging controls / config
    console_metrics_enabled = os.environ.get("HRM_CONSOLE_METRICS", "1") != "0" and RANK == 0
    abort_on_nan = os.environ.get("HRM_ABORT_ON_NAN", "0") == "1"
    jsonl_logging = os.environ.get("HRM_JSONL_LOG", "0") == "1" and RANK == 0
    ema_decay = float(os.environ.get("HRM_EMA_DECAY", "0.9"))
    try:
        log_every_env = int(os.environ.get("HRM_LOG_EVERY", "0"))
    except ValueError:
        log_every_env = 0
    try:
        log_max_wait = float(os.environ.get("HRM_LOG_MAX_WAIT", "30"))
    except ValueError:
        log_max_wait = 30.0
    if log_every_env > 0:
        log_every = max(1, log_every_env)
    else:
        log_every = max(1, train_state.total_steps // 120)  # target ~120 lines

    import time as _time
    py_start = _time.time()
    last_log_time = py_start
    last_step_logged = 0
    ema_metrics: dict[str, float] = {}
    speed_ema = None
    jsonl_file = None
    if jsonl_logging:
        log_path = os.path.join(config.checkpoint_path or '.', 'training_log.jsonl')
        try:
            jsonl_file = open(log_path, 'a', buffering=1)
            if os.path.getsize(log_path) == 0:
                jsonl_file.write('{"event":"meta","total_steps":%d,"config_run_name":"%s"}\n' % (train_state.total_steps, config.run_name))
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to open JSONL log file: {e}")
            jsonl_file = None

    def log_jsonl(obj):
        if jsonl_file is not None:
            import json as _json
            try:
                jsonl_file.write(_json.dumps(obj, separators=(',',':')) + '\n')
            except Exception:
                pass

    # Initial summary (RANK 0)
    if RANK == 0:
        num_params = sum(p.numel() for p in train_state.model.parameters())
        token_per_step = train_state.total_steps and (config.global_batch_size * train_metadata.seq_len)
        device = str(DEFAULT_DEVICE)
        dtype = next(train_state.model.parameters()).dtype if any(True for _ in train_state.model.parameters()) else 'unknown'
        arch_extra = getattr(config.arch, "__pydantic_extra__", {}) or {}
        print(f"[INIT] params={num_params:,} hidden={arch_extra.get('hidden_size','?')} H_layers={arch_extra.get('H_layers','?')} L_layers={arch_extra.get('L_layers','?')} "
              f"steps={train_state.total_steps} batch={config.global_batch_size} seq_len={train_metadata.seq_len} device={device} dtype={dtype}")
        print(f"[INIT] train_groups={train_metadata.total_groups} mean_examples={train_metadata.mean_puzzle_examples} vocab={train_metadata.vocab_size} tokens/step~{token_per_step}")
        log_jsonl({"event":"init","params":num_params,"steps":train_state.total_steps,"batch":config.global_batch_size,"seq_len":train_metadata.seq_len})

    # Training Loop
    last_eval_metrics = None  # store last evaluation metrics dict
    for iter_idx in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {iter_idx * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:  # noqa: F841
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                # NaN / Inf detection on primary loss if present
                primary_loss_key = next((k for k in ("train/lm_loss","train/q_halt_loss","train/q_continue_loss") if k in metrics), None)
                if primary_loss_key is not None:
                    val = metrics[primary_loss_key]
                    try:
                        fval = float(val)
                        if (fval != fval) or fval == float('inf') or fval == float('-inf'):
                            msg = f"[WARN] Detected invalid value ({fval}) in {primary_loss_key} at step {train_state.step}."
                            print(msg)
                            log_jsonl({"event":"nan","step":train_state.step,"metric":primary_loss_key})
                            if abort_on_nan:
                                print("[ABORT] HRM_ABORT_ON_NAN=1 set. Stopping training.")
                                return
                    except Exception:
                        pass

                # EMA update
                for k, v in metrics.items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    prev = ema_metrics.get(k, fv)
                    ema_metrics[k] = prev * ema_decay + (1 - ema_decay) * fv

                # Progress bar
                if progress_bar is not None:
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                # WandB
                if os.environ.get("HRM_DISABLE_WANDB") != "1" and wandb.run is not None:
                    wandb.log(metrics, step=train_state.step)

                # Decide whether to console log
                now = _time.time()
                time_due = (now - last_log_time) >= log_max_wait
                step_due = (train_state.step % log_every == 0) or train_state.step in (1, train_state.total_steps)
                if console_metrics_enabled and (step_due or time_due):
                    elapsed = now - py_start
                    steps_done = train_state.step
                    steps_per_sec = steps_done / max(1e-6, elapsed)
                    speed_ema = steps_per_sec if speed_ema is None else speed_ema * 0.9 + 0.1 * steps_per_sec
                    remaining = max(0, train_state.total_steps - steps_done)
                    eta_sec = remaining / max(1e-9, speed_ema)
                    display_keys = [k for k in ("train/lm_loss","train/q_halt_loss","train/q_continue_loss","train/accuracy") if k in metrics]
                    if not display_keys:
                        display_keys = list(metrics.keys())[:3]
                    pieces = []
                    for k in display_keys:
                        try:
                            raw = float(metrics[k])
                            ema_v = ema_metrics.get(k, raw)
                            if k.endswith('loss'):
                                pieces.append(f"{k.split('/')[-1]}={raw:.4f}(ema={ema_v:.4f})")
                            else:
                                pieces.append(f"{k.split('/')[-1]}={raw:.4f}")
                        except Exception:
                            pass
                    if 'train/lr' in metrics:
                        pieces.append(f"lr={float(metrics['train/lr']):.2e}")
                    pieces.append(f"{speed_ema:.1f} step/s")
                    pieces.append(f"ETA={eta_sec/60:.1f}m")
                    if torch.cuda.is_available():
                        mem = torch.cuda.memory_allocated() / (1024**2)
                        pieces.append(f"mem={mem:.0f}MB")
                    print(f"[train epoch={iter_idx} step {train_state.step}/{train_state.total_steps}] " + ' '.join(pieces))
                    log_jsonl({"event":"train","step":train_state.step,**{k:float(metrics[k]) for k in display_keys if k in metrics}})
                    last_log_time = now
                    last_step_logged = train_state.step

        ############ Evaluation
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
        if metrics is not None:
            last_eval_metrics = metrics
        if RANK == 0 and metrics is not None:
            if os.environ.get("HRM_DISABLE_WANDB") != "1" and wandb.run is not None:
                wandb.log(metrics, step=train_state.step)
            elif console_metrics_enabled:
                # Print evaluation metrics summary
                for split_name, split_metrics in metrics.items():
                    # Sort keys for determinism
                    keys = sorted(split_metrics.keys())
                    # Focus on primary metrics first
                    priority = [k for k in ("accuracy","exact_accuracy","lm_loss","q_halt_accuracy","q_halt_loss","steps") if k in split_metrics]
                    tail = [k for k in keys if k not in priority]
                    show = priority + tail
                    parts = []
                    for k in show[:8]:  # cap display length
                        try:
                            parts.append(f"{k}={float(split_metrics[k]):.4f}")
                        except Exception:
                            pass
                    print(f"[eval epoch={iter_idx} step {train_state.step}] {split_name}: " + ' '.join(parts))
                    log_jsonl({"event":"eval","step":train_state.step,"split":split_name,**{k:float(split_metrics[k]) for k in show if k in split_metrics}})
            
        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (iter_idx == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if os.environ.get("HRM_DISABLE_WANDB") != "1":
        wandb.finish()

    # Final recap (rank 0)
    if RANK == 0:
        if last_eval_metrics is not None:
            # Prefer 'all' split if present
            split_name = 'all' if 'all' in last_eval_metrics else next(iter(last_eval_metrics.keys()))
            split_metrics = last_eval_metrics[split_name]
            # Key metrics ordering
            keys_order = [k for k in ("accuracy","exact_accuracy","lm_loss","q_halt_accuracy","q_halt_loss","steps") if k in split_metrics]
            # Append a few extras if room
            for k in sorted(split_metrics.keys()):
                if k not in keys_order and len(keys_order) < 8:
                    keys_order.append(k)
            parts = []
            for k in keys_order:
                try:
                    parts.append(f"{k}={float(split_metrics[k]):.4f}")
                except Exception:
                    pass
            print(f"[FINAL] steps={train_state.step}/{train_state.total_steps} split={split_name} " + ' '.join(parts))
        else:
            print(f"[FINAL] steps={train_state.step}/{train_state.total_steps} (no eval metrics captured)")


if __name__ == "__main__":
    launch()
