"""Train a scene-independent HOI prior or an object-free HSI prior."""

import contextlib
import datetime
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from prior_utils import (
    append_jsonl,
    balanced_subset_indices,
    build_motion_state,
    dataset_contract,
    DeterministicSubset,
    format_duration,
    load_training_checkpoint,
    make_prefix_mask,
    move_training_batch,
    save_checkpoint,
    seed_everything,
    split_dataset_indices,
)
from utils import find_free_port

os.environ.setdefault("ROOT_DIR", str(Path(__file__).resolve().parent.parent))


class RunRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train.log"
        self.metrics_path = self.output_dir / "metrics.jsonl"

    def log(self, message):
        timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def metric(self, record):
        append_jsonl(self.metrics_path, record)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_manifest(path, cfg, data_contract, world_size, train_batches,
                    updates_per_epoch, start_epoch, start_global_step):
    manifest = {
        "schema_version": 1,
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": [sys.executable] + sys.argv,
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "world_size": int(world_size),
        "per_device_batch_size": int(cfg.per_device_batch_size),
        "global_micro_batch_size": int(cfg.per_device_batch_size * world_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "effective_global_batch_size": int(
            cfg.per_device_batch_size * world_size * cfg.gradient_accumulation_steps
        ),
        "train_batches_per_epoch": int(train_batches),
        "optimizer_updates_per_epoch": int(updates_per_epoch),
        "planned_optimizer_updates": int((cfg.epochs - start_epoch) * updates_per_epoch),
        "start_epoch": int(start_epoch),
        "start_global_step": int(start_global_step),
        "data_contract": data_contract,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)


def _single_entry(config_group):
    if "_target_" in config_group:
        return config_group
    values = list(config_group.values())
    if len(values) != 1:
        raise ValueError(f"Expected one configured component, got {len(values)}")
    return values[0]


def _distributed_mean(total, count, device):
    stats = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1].clamp_min(1)).item()


def _distributed_max(value, device):
    result = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def _loss_from_batch(trainer, batch, cfg, device):
    b = move_training_batch(batch, device)
    state = build_motion_state(b, cfg.prior_type)
    mask = make_prefix_mask(state, cfg.auto_regre_num)
    timesteps = torch.randint(0, trainer.timesteps, (state.shape[0],), device=device)

    loss_dict = trainer.p_losses(
        state, b["joints"], b["mat"], b["scene_flag"], mask, timesteps,
        b["text_clip_embedding"], b["pelvis_goal"], b["scene_goal"],
        b["object_goal"], b["need_scene"], b["need_pelvis_dir"], b["pi"],
        b["end_pi"], b["seg_len"], b["need_pi"], b["is_loco"],
        b["is_object"], b["obj_bps_data"], b["obj_rot_mat_ref"],
        b["rest_pose_obj_nn_pts"], b["transformed_obj_verts"],
        b["rest_human_offsets"], b["object_points"],
    )

    loss = loss_dict["loss"]
    if loss_dict["loss_object"] is not None:
        loss = loss + cfg.loss_weights.object_points * loss_dict["loss_object"]
    if loss_dict["loss_fk"] is not None:
        loss = loss + cfg.loss_weights.forward_kinematics * loss_dict["loss_fk"]
    return loss, loss_dict


@torch.no_grad()
def _validate(trainer, dataloader, cfg, device):
    trainer.student_model.eval()
    total = 0.0
    count = 0
    component_totals = {}
    component_counts = {}

    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.validation.noise_seed)
        for batch_index, batch in enumerate(dataloader):
            if cfg.validation.max_batches > 0 and batch_index >= cfg.validation.max_batches:
                break
            loss, loss_dict = _loss_from_batch(trainer, batch, cfg, device)
            batch_size = batch["joints"].shape[0]
            total += loss.item() * batch_size
            count += batch_size
            for name, value in loss_dict.items():
                if value is None or not torch.is_tensor(value):
                    continue
                component_totals[name] = component_totals.get(name, 0.0) + value.item() * batch_size
                component_counts[name] = component_counts.get(name, 0) + batch_size

    # Keep the optimized objective separate from loss_dict["loss"], which is
    # the unweighted state reconstruction term. Otherwise the component loop
    # silently overwrites the weighted objective used for HOI model selection.
    metrics = {"validation/total_loss": _distributed_mean(total, count, device)}
    for name in sorted(component_totals):
        metrics[f"validation/{name}"] = _distributed_mean(
            component_totals[name], component_counts[name], device
        )
    trainer.student_model.train()
    return metrics


def _make_loader(dataset, indices, cfg, rank, world_size, train):
    subset = (
        Subset(dataset, indices)
        if train else DeterministicSubset(dataset, indices, cfg.validation.noise_seed)
    )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            seed=cfg.seed,
            drop_last=train,
        )
    return DataLoader(
        subset,
        batch_size=cfg.per_device_batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _worker(rank, world_size, cfg):
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    cfg.device = str(device)
    seed_everything(cfg.seed, rank=rank, deterministic=cfg.deterministic)

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_indices, val_indices = split_dataset_indices(
        dataset,
        val_fraction=cfg.validation.fraction,
        split_unit=cfg.validation.split_unit,
        seed=cfg.validation.split_seed,
    )
    full_validation_count = len(val_indices)
    if cfg.validation.max_batches > 0:
        val_indices = balanced_subset_indices(
            dataset,
            val_indices,
            split_unit=cfg.validation.split_unit,
            max_items=(
                cfg.validation.max_batches
                * cfg.per_device_batch_size
                * world_size
            ),
            seed=cfg.validation.noise_seed,
        )
    train_loader = _make_loader(dataset, train_indices, cfg, rank, world_size, train=True)
    val_loader = _make_loader(dataset, val_indices, cfg, rank, world_size, train=False)

    model = hydra.utils.instantiate(_single_entry(cfg.model)).to(device)
    if model.prior_spec.name != cfg.prior_type:
        raise ValueError(
            f"Configured model is {model.prior_spec.name}, requested {cfg.prior_type}"
        )
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            # InfBaGel retains a legacy embedding_output projection that is not
            # used by the x0 forward path.
            find_unused_parameters=True,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    trainer = hydra.utils.instantiate(_single_entry(cfg.sampler))
    trainer.set_dataset_and_model(dataset, model)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    data_contract = dataset_contract(
        dataset, cfg.prior_type, cfg.validation.split_unit,
        cfg.validation.fraction, cfg.validation.split_seed,
    )
    data_contract["num_train_windows"] = len(train_indices)
    data_contract["num_validation_windows"] = full_validation_count
    data_contract["num_validation_windows_per_evaluation"] = len(val_indices)

    start_epoch = 0
    global_step = 0
    if cfg.resume_from:
        checkpoint = load_training_checkpoint(
            cfg.resume_from, model, optimizer, scaler,
            expected_prior=cfg.prior_type, map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    writer = None
    recorder = None
    train_batches_per_epoch = len(train_loader)
    if cfg.max_steps_per_epoch > 0:
        train_batches_per_epoch = min(train_batches_per_epoch, cfg.max_steps_per_epoch)
    updates_per_epoch = math.ceil(
        train_batches_per_epoch / cfg.gradient_accumulation_steps
    )
    initial_global_step = global_step
    planned_run_updates = (cfg.epochs - start_epoch) * updates_per_epoch
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        recorder = RunRecorder(output_dir)
        _write_manifest(
            output_dir / "run_manifest.json", cfg, data_contract, world_size,
            train_batches_per_epoch, updates_per_epoch, start_epoch, global_step,
        )
        if cfg.tensorboard:
            writer = SummaryWriter(str(output_dir / "tensorboard"))
        recorder.log(
            f"{cfg.prior_type.upper()} prior: {len(train_indices)} train / "
            f"{full_validation_count} validation windows "
            f"({len(val_indices)} selected per evaluation) on "
            f"{world_size} process(es); global_batch="
            f"{cfg.per_device_batch_size * world_size * cfg.gradient_accumulation_steps}; "
            f"updates_per_epoch={updates_per_epoch}; planned_updates={planned_run_updates}"
        )

    best_validation = float("inf")
    optimizer.zero_grad(set_to_none=True)
    run_started_at = time.monotonic()
    processed_samples = 0
    for epoch in range(start_epoch, cfg.epochs):
        epoch_started_at = time.monotonic()
        epoch_samples = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_total = 0.0
        epoch_count = 0

        for batch_index, batch in enumerate(train_loader):
            if cfg.max_steps_per_epoch > 0 and batch_index >= cfg.max_steps_per_epoch:
                break
            sync_step = (
                (batch_index + 1) % cfg.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
                or (
                    cfg.max_steps_per_epoch > 0
                    and batch_index + 1 == cfg.max_steps_per_epoch
                )
            )
            sync_context = contextlib.nullcontext()
            if distributed and not sync_step:
                sync_context = model.no_sync()

            with sync_context:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss, loss_dict = _loss_from_batch(trainer, batch, cfg, device)
                    scaled_loss = loss / cfg.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

            if sync_step:
                if cfg.optimizer.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            batch_size = batch["joints"].shape[0]
            epoch_total += loss.item() * batch_size
            epoch_count += batch_size
            global_batch_samples = batch_size * world_size
            epoch_samples += global_batch_samples
            processed_samples += global_batch_samples

            if rank == 0 and global_step > 0 and global_step % cfg.log_every_steps == 0 and sync_step:
                elapsed = time.monotonic() - run_started_at
                completed_updates = global_step - initial_global_step
                remaining_updates = max(0, planned_run_updates - completed_updates)
                eta_seconds = (
                    elapsed / completed_updates * remaining_updates
                    if completed_updates > 0 else float("nan")
                )
                samples_per_second = processed_samples / max(elapsed, 1e-9)
                recorder.log(
                    f"epoch={epoch:03d} step={global_step:07d} "
                    f"run_step={completed_updates:07d}/{planned_run_updates:07d} "
                    f"loss={loss.item():.6f} samples/s={samples_per_second:.2f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)}"
                )
                recorder.metric({
                    "event": "step", "epoch": int(epoch),
                    "global_step": int(global_step),
                    "run_step": int(completed_updates),
                    "loss": float(loss.item()),
                    "samples_per_second": float(samples_per_second),
                    "elapsed_seconds": float(elapsed),
                    "eta_seconds": float(eta_seconds),
                    "processed_samples": int(processed_samples),
                })
                if writer:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                    writer.add_scalar("runtime/samples_per_second", samples_per_second, global_step)
                    writer.add_scalar("runtime/eta_hours", eta_seconds / 3600, global_step)
                    for name, value in loss_dict.items():
                        if value is not None and torch.is_tensor(value):
                            writer.add_scalar(f"train/{name}_step", value.item(), global_step)

        train_seconds = time.monotonic() - epoch_started_at
        train_loss = _distributed_mean(epoch_total, epoch_count, device)
        metrics = {
            "train/loss": train_loss,
            "runtime/train_seconds": train_seconds,
            "runtime/train_samples_per_second": epoch_samples / max(train_seconds, 1e-9),
        }
        validation_started_at = time.monotonic()
        if epoch % cfg.validation.every_epochs == 0 or epoch == cfg.epochs - 1:
            metrics.update(_validate(trainer, val_loader, cfg, device))
        validation_seconds = time.monotonic() - validation_started_at
        elapsed_seconds = time.monotonic() - run_started_at
        completed_epochs = epoch - start_epoch + 1
        remaining_epochs = cfg.epochs - epoch - 1
        metrics["runtime/validation_seconds"] = validation_seconds
        metrics["runtime/epoch_seconds"] = time.monotonic() - epoch_started_at
        metrics["runtime/elapsed_seconds"] = elapsed_seconds
        metrics["runtime/eta_seconds"] = (
            elapsed_seconds / completed_epochs * remaining_epochs
        )
        if device.type == "cuda":
            metrics["runtime/peak_memory_allocated_gib"] = _distributed_max(
                torch.cuda.max_memory_allocated(device) / 2**30, device
            )
            metrics["runtime/peak_memory_reserved_gib"] = _distributed_max(
                torch.cuda.max_memory_reserved(device) / 2**30, device
            )

        if rank == 0:
            summary = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
            recorder.log(
                f"epoch={epoch:03d} {summary} "
                f"eta={format_duration(metrics['runtime/eta_seconds'])}"
            )
            recorder.metric({
                "event": "epoch", "epoch": int(epoch),
                "global_step": int(global_step), **metrics,
            })
            if writer:
                for name, value in metrics.items():
                    writer.add_scalar(name, value, epoch)

            validation_loss = metrics.get("validation/total_loss")
            if validation_loss is not None and validation_loss < best_validation:
                best_validation = validation_loss
                save_checkpoint(
                    checkpoint_dir / "best.pth", model, optimizer, scaler,
                    epoch, global_step, resolved_cfg, cfg.prior_type,
                    data_contract, metrics,
                )

            if (epoch + 1) % cfg.checkpoint_every_epochs == 0 or epoch == cfg.epochs - 1:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pth", model, optimizer,
                    scaler, epoch, global_step, resolved_cfg, cfg.prior_type,
                    data_contract, metrics,
                )
                save_checkpoint(
                    checkpoint_dir / "last.pth", model, optimizer, scaler,
                    epoch, global_step, resolved_cfg, cfg.prior_type,
                    data_contract, metrics,
                )

        if distributed:
            dist.barrier()

    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="config_train_hoi_prior")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    requested_world_size = int(cfg.num_gpus)
    available_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() and requested_world_size > available_gpus:
        raise RuntimeError(
            f"Requested {requested_world_size} GPUs, only {available_gpus} available"
        )
    if not torch.cuda.is_available() and requested_world_size != 1:
        raise RuntimeError("CPU training requires num_gpus=1")

    world_size = requested_world_size
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", find_free_port())
        mp.spawn(_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        _worker(0, 1, cfg)


if __name__ == "__main__":
    main()
