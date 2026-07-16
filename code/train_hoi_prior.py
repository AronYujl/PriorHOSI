"""Phase 1B scene-free HOIPrior DDP training, validation, checkpoint and resume."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import get_smpl_parents
from priors.data import PriorWindowDataset
from priors.diffusion import GaussianDiffusion, normalize_progress
from priors.losses import hoi_training_losses
from priors.models import build_expert
from priors.representation import REPRESENTATION
from priors.window_codec import BPS_SHA256


LOSS_KEYS = (
    "total", "reconstruction", "joint_position", "joint_rotation", "object_translation",
    "object_rotation", "contact", "fk", "object_surface", "velocity", "object_goal", "contact_accuracy",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as value:
        value.bind(("", 0))
        return int(value.getsockname()[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_weight_initialization(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> Dict[str, object]:
    """Load only the hash-locked online model weights allowed by D2-M0."""
    path_value = cfg.weight_init_checkpoint
    if path_value in (None, "", False):
        return {
            "mode": "random",
            "source_checkpoint": None,
            "source_checkpoint_sha256": None,
            "source_model_state_sha256": None,
            "initial_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "restored_components": [],
            "old_optimizer_states_loaded": 0,
            "old_ema_models_loaded": 0,
            "old_scheduler_states_loaded": 0,
            "old_scaler_states_loaded": 0,
            "old_rng_states_loaded": 0,
        }
    from priors.optimizer_reset import (
        CANDIDATES,
        SOURCE_CHECKPOINT_SHA256,
        SOURCE_RUN_ID,
    )

    if str(cfg.d2m_candidate) not in CANDIDATES:
        raise ValueError("weight-only initialization is restricted to a registered D2-M candidate")
    if str(cfg.weight_init_variant) != "online":
        raise ValueError("D2-M weight-only initialization accepts online model weights only")
    if str(cfg.weight_init_sha256) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("D2-M source checkpoint configured SHA-256 mismatch")
    path = Path(str(path_value)).resolve()
    actual_sha256 = _sha256(path)
    if actual_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise ValueError(f"D2-M source checkpoint file hash mismatch: {actual_sha256}")
    checkpoint = torch.load(path, map_location="cpu")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_type") != "hoi_prior_phase1b"
        or checkpoint.get("expert") != "hoi"
        or checkpoint.get("initialization") != "random"
    ):
        raise ValueError(
            "D2-M weight initialization requires a self-trained random-origin Phase 1B checkpoint; "
            "released checkpoint initialization is forbidden"
        )
    if checkpoint.get("run_id") != SOURCE_RUN_ID:
        raise ValueError(f"D2-M source run mismatch: {checkpoint.get('run_id')}")
    if checkpoint.get("model_config") != _model_config(cfg):
        raise ValueError("D2-M source model configuration mismatch")
    if checkpoint.get("data_contract_sha256") != str(cfg.data_contract_sha256):
        raise ValueError("D2-M source data contract mismatch")
    split_sha256 = _sha256(Path(str(cfg.split_manifest)).resolve())
    if checkpoint.get("split_sha256") != split_sha256:
        raise ValueError("D2-M source split mismatch")
    source_model = checkpoint.get("model")
    if not isinstance(source_model, dict):
        raise ValueError("D2-M source checkpoint is missing online model weights")
    source_model_sha256 = _state_dict_sha256(source_model)
    model.load_state_dict(source_model, strict=True)
    initial_model_sha256 = _state_dict_sha256(model.state_dict())
    if initial_model_sha256 != source_model_sha256:
        raise ValueError("D2-M source online model did not load exactly")
    return {
        "mode": "phase1b_online_weight_only",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": actual_sha256,
        "source_run_id": checkpoint.get("run_id"),
        "source_git_commit": checkpoint.get("git_commit"),
        "source_processed_windows": checkpoint.get("processed_windows"),
        "source_optimizer_updates": checkpoint.get("optimizer_updates"),
        "source_model_state_sha256": source_model_sha256,
        "initial_model_state_sha256": initial_model_sha256,
        "restored_components": ["model"],
        "old_optimizer_states_loaded": 0,
        "old_ema_models_loaded": 0,
        "old_scheduler_states_loaded": 0,
        "old_scaler_states_loaded": 0,
        "old_rng_states_loaded": 0,
    }


def _atomic_json(path: Path, value: Dict[str, object], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _model_config(cfg: DictConfig) -> Dict[str, int]:
    return {
        "dim_model": int(cfg.dim_model),
        "num_heads": int(cfg.num_heads),
        "num_layers": int(cfg.num_layers),
    }


def _resume_contract(cfg: DictConfig) -> Dict[str, object]:
    """Critical immutable fields for an exact same-run resume."""
    split = Path(str(cfg.split_manifest)).resolve()
    return {
        "model_config": _model_config(cfg),
        "batch_size": int(cfg.batch_size),
        "num_gpus": int(cfg.num_gpus),
        "effective_batch_size": int(cfg.effective_batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "max_processed_windows": int(cfg.max_processed_windows),
        "validation_windows": int(cfg.validation_windows),
        "validation_interval_windows": int(cfg.validation_interval_windows),
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows),
        "learning_rate": float(cfg.learning_rate),
        "warmup_windows": int(cfg.warmup_windows),
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio),
        "weight_decay": float(cfg.weight_decay),
        "betas": [float(cfg.beta1), float(cfg.beta2)],
        "gradient_clip_norm": float(cfg.gradient_clip_norm),
        "ema_decays": [float(value) for value in cfg.ema_decays],
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows),
        "fk_weight": float(cfg.fk_weight),
        "object_surface_weight": float(cfg.object_surface_weight),
        "velocity_weight": float(cfg.velocity_weight),
        "goal_weight": float(cfg.goal_weight),
        "weight_init_sha256": (
            None if cfg.weight_init_sha256 in (None, "", False) else str(cfg.weight_init_sha256)
        ),
        "weight_init_variant": (
            None if cfg.weight_init_variant in (None, "", False) else str(cfg.weight_init_variant)
        ),
        "d2m_candidate": (
            None if cfg.d2m_candidate in (None, "", False) else str(cfg.d2m_candidate)
        ),
        "d2m_rng_audit": bool(cfg.d2m_rng_audit),
        "amp": bool(cfg.amp),
        "data_contract_sha256": str(cfg.data_contract_sha256),
        "split_sha256": _sha256(split),
    }


def _lr_lambda(update: int, total_updates: int, warmup_updates: int, minimum_ratio: float) -> float:
    if warmup_updates and update < warmup_updates:
        return max((update + 1) / warmup_updates, 1.0 / warmup_updates)
    remaining = max(total_updates - warmup_updates, 1)
    progress = min(max((update - warmup_updates) / remaining, 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in (
            "x", "text_embedding", "object_bps", "goals", "progress", "rest_human_offsets",
            "terminal_window", "rest_object_points", "world_to_local_rotation",
            "object_rotation_reference",
        )
    }


def _forward_losses(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Dict[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg: DictConfig,
    *,
    generator: Optional[torch.Generator] = None,
    audit_digest=None,
) -> Dict[str, torch.Tensor]:
    clean = batch["x"]
    timesteps = torch.randint(
        0, REPRESENTATION.diffusion_steps, (clean.shape[0],), device=clean.device, generator=generator,
    )
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    if audit_digest is not None:
        audit_digest.update(
            clean[:, (0, -1), :8].detach().contiguous().cpu().numpy().tobytes()
        )
        audit_digest.update(timesteps.detach().contiguous().cpu().numpy().tobytes())
        audit_digest.update(
            noise[:, (0, -1), :8].detach().contiguous().cpu().numpy().tobytes()
        )
    noisy = diffusion.q_sample(clean, timesteps, noise)
    prediction = model(
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
    )
    return hoi_training_losses(
        prediction,
        clean,
        batch["goals"],
        batch["rest_human_offsets"],
        parents,
        minimum,
        maximum,
        object_minimum,
        object_maximum,
        batch["terminal_window"],
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
        fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight),
        goal_weight=float(cfg.goal_weight),
    )


@torch.no_grad()
def _ema_update(ema: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    for target_parameter, source_parameter in zip(ema.parameters(), source.parameters()):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(ema.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer)


@torch.no_grad()
def _validate(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg: DictConfig,
    processed_windows: int,
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    local_limit = int(cfg.validation_windows) // world_size
    if local_limit * world_size != int(cfg.validation_windows):
        raise ValueError("validation_windows must be divisible by world size")
    totals = torch.zeros(len(LOSS_KEYS) + 1, dtype=torch.float64, device=minimum.device)
    generator = torch.Generator(device=minimum.device)
    generator.manual_seed(int(cfg.seed) * 1000003 + processed_windows + rank)
    seen = 0
    while seen < local_limit:
        previous_seen = seen
        for raw_batch in loader:
            if seen >= local_limit:
                break
            batch = _move_batch(raw_batch, minimum.device)
            remaining = local_limit - seen
            if batch["x"].shape[0] > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                losses = _forward_losses(
                    model, diffusion, batch, parents, minimum, maximum,
                    object_minimum, object_maximum, cfg, generator=generator,
                )
            count = batch["x"].shape[0]
            for index, key in enumerate(LOSS_KEYS):
                totals[index] += losses[key].detach().double() * count
            totals[-1] += count
            seen += count
        if seen == previous_seen:
            raise RuntimeError("internal validation loader produced no batches")
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    if int(totals[-1].item()) != int(cfg.validation_windows):
        raise RuntimeError(f"validated {int(totals[-1])} windows, expected {cfg.validation_windows}")
    result = {
        key: float((totals[index] / totals[-1]).item()) for index, key in enumerate(LOSS_KEYS)
    }
    result.update({
        "processed_windows": processed_windows,
        "validation_windows": int(totals[-1].item()),
        "finite": all(math.isfinite(value) for value in result.values()),
    })
    model.train(was_training)
    return result


def _rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng(value: Dict[str, object]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state(value["cuda"])


def _save_checkpoint(
    rank: int,
    world_size: int,
    cfg: DictConfig,
    model: DistributedDataParallel,
    ema_models: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    *,
    processed_windows: int,
    optimizer_updates: int,
    amp_overflow_skips: int,
    epoch: int,
    batches_consumed_in_epoch: int,
    checkpoint_hashes: List[Dict[str, object]],
    weight_initialization: Mapping[str, object],
) -> Path:
    checkpoint_dir = Path(str(cfg.checkpoint_dir)).resolve()
    checkpoint_path = checkpoint_dir / f"{cfg.run_id}_windows{processed_windows:09d}.pth"
    rng_path = checkpoint_dir / f"{checkpoint_path.stem}.rank{rank}.rng.pth"
    _atomic_torch_save(rng_path, _rng_state())
    torch.distributed.barrier()
    if rank == 0:
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {checkpoint_path}")
        value = {
            "schema_version": 2,
            "checkpoint_type": "hoi_prior_phase1b",
            "window_state_codec": "state-compositional-v1",
            "expert": "hoi",
            "initialization": "random",
            "run_id": str(cfg.run_id),
            "seed": int(cfg.seed),
            "git_commit": _git_commit(Path(str(cfg.repo_root)).resolve()),
            "processed_windows": processed_windows,
            "processed_frames": processed_windows * REPRESENTATION.window_frames,
            "optimizer_updates": optimizer_updates,
            "amp_overflow_skips": amp_overflow_skips,
            "epoch": epoch,
            "batches_consumed_in_epoch": batches_consumed_in_epoch,
            "world_size": world_size,
            "effective_batch_size": int(cfg.effective_batch_size),
            "model_config": _model_config(cfg),
            "resume_contract": _resume_contract(cfg),
            "data_contract_sha256": str(cfg.data_contract_sha256),
            "split_sha256": _sha256(Path(str(cfg.split_manifest)).resolve()),
            "model": model.module.state_dict(),
            "ema_models": {key: value.state_dict() for key, value in ema_models.items()},
            # Retain the legacy name for the official evaluator until D2 locks a terminal variant.
            "ema_model": ema_models["0.9999"].state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_pattern": f"{checkpoint_path.stem}.rank{{rank}}.rng.pth",
            "weight_initialization": dict(weight_initialization),
        }
        _atomic_torch_save(checkpoint_path, value)
        checkpoint_hashes.append({
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "processed_windows": processed_windows,
        })
    torch.distributed.barrier()
    return checkpoint_path


def _load_resume(
    rank: int,
    cfg: DictConfig,
    model: DistributedDataParallel,
    ema_models: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, int]:
    path = Path(str(cfg.resume_checkpoint)).resolve()
    checkpoint = torch.load(path, map_location=f"cuda:{rank}")
    if checkpoint.get("checkpoint_type") != "hoi_prior_phase1b":
        raise ValueError("resume checkpoint is not a Phase 1B HOIPrior checkpoint")
    current_commit = _git_commit(Path(str(cfg.repo_root)).resolve())
    if checkpoint.get("git_commit") != current_commit:
        raise ValueError(
            f"resume checkpoint Git commit mismatch: {checkpoint.get('git_commit')} != {current_commit}"
        )
    if checkpoint.get("resume_contract") != _resume_contract(cfg):
        raise ValueError("resume checkpoint training contract mismatch")
    for key, expected in (
        ("run_id", str(cfg.run_id)),
        ("seed", int(cfg.seed)),
        ("world_size", int(cfg.num_gpus)),
        ("effective_batch_size", int(cfg.effective_batch_size)),
        ("model_config", _model_config(cfg)),
        ("data_contract_sha256", str(cfg.data_contract_sha256)),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(f"resume checkpoint {key} mismatch: {checkpoint.get(key)!r} != {expected!r}")
    model.module.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_emas = checkpoint.get("ema_models")
    if not isinstance(checkpoint_emas, dict) or set(checkpoint_emas) != set(ema_models):
        raise ValueError("resume checkpoint EMA variants mismatch")
    for key, ema in ema_models.items():
        ema.load_state_dict(checkpoint_emas[key], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng_path = path.parent / checkpoint["rng_pattern"].format(rank=rank)
    _restore_rng(torch.load(rng_path, map_location="cpu"))
    return {
        "processed_windows": int(checkpoint["processed_windows"]),
        "optimizer_updates": int(checkpoint["optimizer_updates"]),
        "amp_overflow_skips": int(checkpoint.get("amp_overflow_skips", 0)),
        "epoch": int(checkpoint["epoch"]),
        "batches_consumed_in_epoch": int(checkpoint["batches_consumed_in_epoch"]),
    }


def _worker(rank: int, cfg: DictConfig) -> None:
    world_size = int(cfg.num_gpus)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    effective = int(cfg.batch_size) * world_size * int(cfg.gradient_accumulation_steps)
    if effective != int(cfg.effective_batch_size):
        raise ValueError(
            f"effective batch mismatch: {cfg.batch_size} x {world_size} x "
            f"{cfg.gradient_accumulation_steps} = {effective}, configured {cfg.effective_batch_size}"
        )
    if int(cfg.max_processed_windows) % effective:
        raise ValueError("max_processed_windows must be divisible by effective_batch_size")
    if int(cfg.warmup_windows) % effective:
        raise ValueError("warmup_windows must be divisible by effective_batch_size")
    if int(cfg.window_frames) != REPRESENTATION.window_frames:
        raise ValueError("HOIPrior window_frames contract mismatch")
    if int(cfg.history_frames) != REPRESENTATION.history_frames:
        raise ValueError("HOIPrior history_frames contract mismatch")
    if int(cfg.diffusion_steps) != REPRESENTATION.diffusion_steps:
        raise ValueError("HOIPrior diffusion_steps contract mismatch")
    if int(cfg.max_consecutive_amp_overflows) < 0:
        raise ValueError("max_consecutive_amp_overflows must be non-negative")
    if _model_config(cfg) != {"dim_model": 512, "num_heads": 16, "num_layers": 8}:
        raise ValueError("HOIPrior architecture must remain 512-wide, 16-head, 8-layer")
    if cfg.init_checkpoint not in (None, "", False):
        raise ValueError("HOIPrior training initialization must be random; init_checkpoint is forbidden")
    d2m_candidate = None if cfg.d2m_candidate in (None, "", False) else str(cfg.d2m_candidate)
    configured_weights = {
        "fk": float(cfg.fk_weight),
        "object_surface": float(cfg.object_surface_weight),
        "velocity": float(cfg.velocity_weight),
        "terminal_goal": float(cfg.goal_weight),
    }
    if d2m_candidate is None:
        if configured_weights != {
            "fk": 50.0,
            "object_surface": 50.0,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        }:
            raise ValueError(
                "Phase 1B remediation loss weights are locked at FK/surface/velocity/goal=50/50/0.1/1"
            )
        if cfg.weight_init_checkpoint not in (None, "", False):
            raise ValueError("weight-only initialization requires a registered D2-M candidate")
    else:
        from priors.optimizer_reset import (
            CANDIDATES,
            EFFECTIVE_BATCH_SIZE,
            OPTIMIZER_UPDATES,
            PROCESSED_WINDOWS,
            SOURCE_OPTIMIZER_LR,
            WEIGHTS,
        )

        if d2m_candidate not in CANDIDATES or configured_weights != WEIGHTS[d2m_candidate]:
            raise ValueError("D2-M candidate loss weights do not match the locked contract")
        if (
            world_size != 4
            or int(cfg.batch_size) != 768
            or int(cfg.gradient_accumulation_steps) != 1
            or int(cfg.effective_batch_size) != EFFECTIVE_BATCH_SIZE
            or int(cfg.max_processed_windows) != PROCESSED_WINDOWS
            or int(cfg.max_processed_windows) // int(cfg.effective_batch_size) != OPTIMIZER_UPDATES
            or float(cfg.learning_rate) != SOURCE_OPTIMIZER_LR
            or int(cfg.warmup_windows) != 0
            or float(cfg.minimum_lr_ratio) != 1.0
            or not bool(cfg.d2m_rng_audit)
        ):
            raise ValueError("D2-M optimizer-reset smoke budget/LR/RNG contract mismatch")
    if world_size not in {1, 4}:
        raise ValueError("Phase 1B supports one-GPU functional smoke or four-GPU worker execution")

    split_manifest = str(Path(str(cfg.split_manifest)).resolve())
    train_dataset = PriorWindowDataset(
        str(cfg.repo_root), "hoi", partition="train", limit=int(cfg.dataset_limit),
        split_manifest=split_manifest,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(cfg.seed), drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        sampler=train_sampler,
        drop_last=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.num_workers) > 0,
    )
    validation_loader = None
    if int(cfg.validation_windows):
        validation_dataset = PriorWindowDataset(
            str(cfg.repo_root), "hoi", partition="internal_validation",
            split_manifest=split_manifest,
        )
        validation_sampler = DistributedSampler(
            validation_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(cfg.validation_batch_size),
            sampler=validation_sampler,
            drop_last=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            persistent_workers=int(cfg.num_workers) > 0,
        )

    model = build_expert(
        "hoi", init_checkpoint=cfg.init_checkpoint, dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads), num_layers=int(cfg.num_layers),
    ).to(device)
    weight_initialization = _load_weight_initialization(cfg, model)
    model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
    ema_decays = [float(value) for value in cfg.ema_decays]
    if ema_decays != [0.999, 0.9999]:
        raise ValueError("Phase 1B remediation requires EMA decays 0.999 and 0.9999")
    ema_models = {
        str(decay): copy.deepcopy(model.module).requires_grad_(False).eval()
        for decay in ema_decays
    }
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay),
        betas=(float(cfg.beta1), float(cfg.beta2)),
    )
    total_updates = int(cfg.max_processed_windows) // effective
    warmup_updates = int(cfg.warmup_windows) // effective
    scheduler = LambdaLR(
        optimizer,
        lambda update: _lr_lambda(update, total_updates, warmup_updates, float(cfg.minimum_lr_ratio)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp))
    initial_optimizer_state_count = len(optimizer.state)
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True), device=device, dtype=torch.long)
    norm = np.load(Path(str(cfg.repo_root)) / "data/train/norm.npy")
    minimum = torch.as_tensor(norm[0], device=device, dtype=torch.float32)
    maximum = torch.as_tensor(norm[1], device=device, dtype=torch.float32)
    object_minimum = torch.as_tensor(norm[2], device=device, dtype=torch.float32)
    object_maximum = torch.as_tensor(norm[3], device=device, dtype=torch.float32)

    state = {
        "processed_windows": 0,
        "optimizer_updates": 0,
        "amp_overflow_skips": 0,
        "epoch": 0,
        "batches_consumed_in_epoch": 0,
    }
    resumed_from = None
    if cfg.resume_checkpoint not in (None, "", False):
        state = _load_resume(rank, cfg, model, ema_models, optimizer, scheduler, scaler)
        resumed_from = str(Path(str(cfg.resume_checkpoint)).resolve())
    processed_windows = state["processed_windows"]
    optimizer_updates = state["optimizer_updates"]
    amp_overflow_skips = state["amp_overflow_skips"]
    start_epoch = state["epoch"]
    resume_batches = state["batches_consumed_in_epoch"]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    compute_update_seconds: List[float] = []
    loss_sums = {key: 0.0 for key in LOSS_KEYS}
    loss_observations = 0
    pending_loss_sums = {key: 0.0 for key in LOSS_KEYS}
    pending_loss_observations = 0
    consecutive_amp_overflows = 0
    initial_grad_scale = float(scaler.get_scale())
    validation_records: List[Dict[str, object]] = []
    checkpoint_hashes: List[Dict[str, object]] = []
    training_rng_digest = hashlib.sha256() if bool(cfg.d2m_rng_audit) else None
    optimizer.zero_grad(set_to_none=True)
    paused = False
    last_checkpoint_windows = -1
    epoch = start_epoch
    batches_consumed = resume_batches

    while processed_windows < int(cfg.max_processed_windows):
        train_sampler.set_epoch(epoch)
        micro_in_accumulation = 0
        group_start = None
        for batch_index, raw_batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batches:
                continue
            if micro_in_accumulation == 0 and bool(cfg.profile_every_update):
                torch.cuda.synchronize(device)
                group_start = time.perf_counter()
            batch = _move_batch(raw_batch, device)
            boundary = micro_in_accumulation + 1 == int(cfg.gradient_accumulation_steps)
            sync_context = contextlib.nullcontext() if boundary else model.no_sync()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                    losses = _forward_losses(
                        model, diffusion, batch, parents, minimum, maximum,
                        object_minimum, object_maximum, cfg,
                        audit_digest=training_rng_digest,
                    )
                    loss = losses["total"] / int(cfg.gradient_accumulation_steps)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite HOIPrior loss at update {optimizer_updates}")
                scaler.scale(loss).backward()
            for key in LOSS_KEYS:
                pending_loss_sums[key] += float(losses[key].detach())
            pending_loss_observations += 1
            micro_in_accumulation += 1
            batches_consumed = batch_index + 1
            if not boundary:
                continue

            scaler.unscale_(optimizer)
            key_gradient = model.module.network.motion_input.weight.grad
            if key_gradient is None:
                raise FloatingPointError("missing key HOIPrior gradient")
            local_nonfinite = torch.tensor(
                int(any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )),
                dtype=torch.int32,
                device=device,
            )
            torch.distributed.all_reduce(local_nonfinite, op=torch.distributed.ReduceOp.MAX)
            if int(local_nonfinite.item()):
                if not bool(cfg.amp):
                    raise FloatingPointError("non-finite HOIPrior gradient with AMP disabled")
                amp_overflow_skips += 1
                consecutive_amp_overflows += 1
                scaler.update(new_scale=float(scaler.get_scale()) * 0.5)
                optimizer.zero_grad(set_to_none=True)
                pending_loss_sums = {key: 0.0 for key in LOSS_KEYS}
                pending_loss_observations = 0
                micro_in_accumulation = 0
                group_start = None
                if consecutive_amp_overflows > int(cfg.max_consecutive_amp_overflows):
                    raise FloatingPointError(
                        f"AMP gradient overflow persisted for {consecutive_amp_overflows} groups"
                    )
                continue
            if not torch.any(key_gradient != 0):
                raise FloatingPointError("zero key HOIPrior gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip_norm))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite HOIPrior gradient norm")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            for decay in ema_decays:
                _ema_update(ema_models[str(decay)], model.module, decay)
            optimizer_updates += 1
            processed_windows += effective
            for key in LOSS_KEYS:
                loss_sums[key] += pending_loss_sums[key]
                pending_loss_sums[key] = 0.0
            loss_observations += pending_loss_observations
            pending_loss_observations = 0
            consecutive_amp_overflows = 0
            micro_in_accumulation = 0
            if group_start is not None:
                torch.cuda.synchronize(device)
                compute_update_seconds.append(time.perf_counter() - group_start)

            if (
                validation_loader is not None
                and int(cfg.validation_interval_windows)
                and processed_windows % int(cfg.validation_interval_windows) == 0
            ):
                validation_records.append(_validate(
                    rank, world_size, ema_models["0.9999"], diffusion, validation_loader, parents,
                    minimum, maximum, object_minimum, object_maximum, cfg, processed_windows,
                ))
            if (
                int(cfg.checkpoint_interval_windows)
                and processed_windows % int(cfg.checkpoint_interval_windows) == 0
            ):
                _save_checkpoint(
                    rank, world_size, cfg, model, ema_models, optimizer, scheduler, scaler,
                    processed_windows=processed_windows,
                    optimizer_updates=optimizer_updates,
                    amp_overflow_skips=amp_overflow_skips,
                    epoch=epoch,
                    batches_consumed_in_epoch=batches_consumed,
                    checkpoint_hashes=checkpoint_hashes,
                    weight_initialization=weight_initialization,
                )
                last_checkpoint_windows = processed_windows
            pause_at = int(cfg.pause_after_windows) if cfg.pause_after_windows is not None else 0
            if pause_at and processed_windows >= pause_at and processed_windows < int(cfg.max_processed_windows):
                if last_checkpoint_windows != processed_windows:
                    _save_checkpoint(
                        rank, world_size, cfg, model, ema_models, optimizer, scheduler, scaler,
                        processed_windows=processed_windows,
                        optimizer_updates=optimizer_updates,
                        amp_overflow_skips=amp_overflow_skips,
                        epoch=epoch,
                        batches_consumed_in_epoch=batches_consumed,
                        checkpoint_hashes=checkpoint_hashes,
                        weight_initialization=weight_initialization,
                    )
                    last_checkpoint_windows = processed_windows
                paused = True
                break
            if processed_windows >= int(cfg.max_processed_windows):
                break
        if paused or processed_windows >= int(cfg.max_processed_windows):
            break
        epoch += 1
        resume_batches = 0

    if not paused and validation_loader is not None:
        if not validation_records or validation_records[-1]["processed_windows"] != processed_windows:
            validation_records.append(_validate(
                rank, world_size, ema_models["0.9999"], diffusion, validation_loader, parents,
                minimum, maximum, object_minimum, object_maximum, cfg, processed_windows,
            ))
    if last_checkpoint_windows != processed_windows:
        terminal_checkpoint = _save_checkpoint(
            rank, world_size, cfg, model, ema_models, optimizer, scheduler, scaler,
            processed_windows=processed_windows,
            optimizer_updates=optimizer_updates,
            amp_overflow_skips=amp_overflow_skips,
            epoch=epoch,
            batches_consumed_in_epoch=batches_consumed,
            checkpoint_hashes=checkpoint_hashes,
            weight_initialization=weight_initialization,
        )
    else:
        terminal_checkpoint = Path(str(cfg.checkpoint_dir)).resolve() / (
            f"{cfg.run_id}_windows{processed_windows:09d}.pth"
        )

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    local = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
            wall_seconds,
            sum(compute_update_seconds),
            len(compute_update_seconds),
            amp_overflow_skips,
            initial_grad_scale,
            float(scaler.get_scale()),
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    loss_vector = torch.tensor(
        [loss_sums[key] for key in LOSS_KEYS] + [loss_observations],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(loss_vector, op=torch.distributed.ReduceOp.SUM)
    audit_hashes = None
    if training_rng_digest is not None:
        audit_bytes = bytes.fromhex(training_rng_digest.hexdigest())
        audit_tensor = torch.tensor(list(audit_bytes), dtype=torch.uint8, device=device)
        gathered_audits = [torch.zeros_like(audit_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_audits, audit_tensor)
        audit_hashes = [
            bytes(int(value) for value in item.cpu().tolist()).hex()
            for item in gathered_audits
        ]
    if rank == 0:
        values = [item.cpu().tolist() for item in gathered]
        device_total = torch.cuda.get_device_properties(device).total_memory
        status = "paused" if paused else "completed"
        state_record = {
            "schema_version": 1,
            "status": status,
            "run_id": str(cfg.run_id),
            "seed": int(cfg.seed),
            "processed_windows": processed_windows,
            "processed_frames": processed_windows * REPRESENTATION.window_frames,
            "optimizer_updates": optimizer_updates,
            "amp_overflow_skips": amp_overflow_skips,
            "terminal_checkpoint": str(terminal_checkpoint),
            "terminal_checkpoint_sha256": _sha256(terminal_checkpoint),
            "resume_checkpoint": resumed_from,
        }
        _atomic_json(Path(str(cfg.state_path)).resolve(), state_record, overwrite=True)
        if not paused:
            optimizer_steps = [
                int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
                for value in optimizer.state.values()
                if "step" in value
            ]
            averaged_losses = {
                key: float(loss_vector[index].item() / max(loss_vector[-1].item(), 1.0))
                for index, key in enumerate(LOSS_KEYS)
            }
            metrics = {
                **state_record,
                "status": "stable",
                "expert": "hoi",
                "initialization": "random",
                "training_start": weight_initialization["mode"],
                "released_checkpoint_used": False,
                "git_commit": _git_commit(Path(str(cfg.repo_root)).resolve()),
                "world_size": world_size,
                "gpu_name": torch.cuda.get_device_name(device),
                "micro_batch_per_gpu": int(cfg.batch_size),
                "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                "effective_batch_size": effective,
                "learning_rate": float(cfg.learning_rate),
                "warmup_windows": int(cfg.warmup_windows),
                "warmup_updates": warmup_updates,
                "epochs_equivalent": processed_windows / len(train_dataset),
                "parameter_count": sum(parameter.numel() for parameter in model.module.parameters()),
                "key_gradient_present": True,
                "amp_overflow_skips_by_rank": [int(value[5]) for value in values],
                "initial_grad_scale_by_rank": [value[6] for value in values],
                "final_grad_scale_by_rank": [value[7] for value in values],
                "loss_finite": all(math.isfinite(value) for value in averaged_losses.values()),
                "mean_training_losses": averaged_losses,
                "validation": validation_records,
                "peak_memory_allocated_bytes_by_rank": [int(value[0]) for value in values],
                "peak_memory_reserved_bytes_by_rank": [int(value[1]) for value in values],
                "memory_headroom_bytes_by_rank": [device_total - int(value[1]) for value in values],
                "device_total_memory_bytes": device_total,
                "wall_seconds_by_rank": [value[2] for value in values],
                "wall_seconds": max(value[2] for value in values),
                "throughput_windows_per_second": processed_windows / max(value[2] for value in values),
                "throughput_frames_per_second": processed_windows * REPRESENTATION.window_frames / max(value[2] for value in values),
                "mean_profiled_update_seconds_by_rank": [
                    value[3] / value[4] if value[4] else None for value in values
                ],
                "timing_cuda_synchronized": True,
                "checkpoint_hashes": checkpoint_hashes,
                "weight_initialization": weight_initialization,
                "initial_optimizer_state_count": initial_optimizer_state_count,
                "terminal_optimizer_state_count": len(optimizer.state),
                "terminal_optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
                "terminal_optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
                "training_rng_audit_schema": (
                    "per-rank SHA256(clean[:,{0,-1},:8], timesteps, q_noise[:,{0,-1},:8])"
                    if audit_hashes is not None else None
                ),
                "training_rng_sha256_by_rank": audit_hashes,
                "d2m_candidate": d2m_candidate,
                "model_config": _model_config(cfg),
                "ema_decays": ema_decays,
                "loss_weights": {
                    "fk": float(cfg.fk_weight),
                    "object_surface": float(cfg.object_surface_weight),
                    "velocity": float(cfg.velocity_weight),
                    "terminal_object_goal": float(cfg.goal_weight),
                },
                "window_state_codec": "state-compositional-v1",
                "bps_sha256": BPS_SHA256,
                "representation": REPRESENTATION.as_dict(),
                "data_contract_sha256": str(cfg.data_contract_sha256),
                "split_manifest": split_manifest,
                "split_sha256": _sha256(Path(split_manifest)),
            }
            _atomic_json(Path(str(cfg.metrics_path)).resolve(), metrics)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="config_train_hoi_prior")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < int(cfg.num_gpus):
        raise RuntimeError(f"requires {cfg.num_gpus} visible CUDA devices")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    torch.multiprocessing.spawn(_worker, args=(cfg,), nprocs=int(cfg.num_gpus), join=True)


if __name__ == "__main__":
    main()
