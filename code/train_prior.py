"""Train a scene-independent HOI prior or an object-free HSI prior."""

import contextlib
import datetime
import os
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
    balanced_subset_indices,
    build_motion_state,
    dataset_contract,
    load_training_checkpoint,
    make_prefix_mask,
    move_training_batch,
    save_checkpoint,
    seed_everything,
    split_dataset_indices,
)
from utils import find_free_port

os.environ.setdefault("ROOT_DIR", str(Path(__file__).resolve().parent.parent))


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

    metrics = {"validation/loss": _distributed_mean(total, count, device)}
    for name in sorted(component_totals):
        metrics[f"validation/{name}"] = _distributed_mean(
            component_totals[name], component_counts[name], device
        )
    trainer.student_model.train()
    return metrics


def _make_loader(dataset, indices, cfg, rank, world_size, train):
    subset = Subset(dataset, indices)
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
            find_unused_parameters=False,
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
    data_contract["num_validation_windows"] = len(val_indices)

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
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        if cfg.tensorboard:
            writer = SummaryWriter(str(output_dir / "tensorboard"))
        print(
            f"{cfg.prior_type.upper()} prior: {len(train_indices)} train / "
            f"{len(val_indices)} validation windows on {world_size} process(es)",
            flush=True,
        )

    best_validation = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, cfg.epochs):
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

            if rank == 0 and global_step > 0 and global_step % cfg.log_every_steps == 0 and sync_step:
                print(
                    f"epoch={epoch:03d} step={global_step:07d} "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )
                if writer:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                    for name, value in loss_dict.items():
                        if value is not None and torch.is_tensor(value):
                            writer.add_scalar(f"train/{name}_step", value.item(), global_step)

        train_loss = _distributed_mean(epoch_total, epoch_count, device)
        metrics = {"train/loss": train_loss}
        if epoch % cfg.validation.every_epochs == 0 or epoch == cfg.epochs - 1:
            metrics.update(_validate(trainer, val_loader, cfg, device))

        if rank == 0:
            summary = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
            print(f"epoch={epoch:03d} {summary}", flush=True)
            if writer:
                for name, value in metrics.items():
                    writer.add_scalar(name, value, epoch)

            validation_loss = metrics.get("validation/loss")
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
