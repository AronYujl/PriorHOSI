"""Phase 1A real-data single-/multi-GPU expert optimizer-update smoke."""

import json
import os
import random
import socket
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from priors.hoi.data import PriorWindowDataset
from priors.core.diffusion_schedule import canonical_diffusion_schedule
from priors.core.expert_api import build_expert
from priors.core.representation import REPRESENTATION, masked_reconstruction_loss


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as value:
        value.bind(("", 0))
        return int(value.getsockname()[1])


def _diffuse(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    alpha_bar = canonical_diffusion_schedule()["alpha_bar"].to(
        device=clean.device,
    )[timesteps]
    noisy = alpha_bar.sqrt()[:, None, None] * clean + (1.0 - alpha_bar).sqrt()[:, None, None] * noise
    noisy[:, :REPRESENTATION.history_frames] = clean[:, :REPRESENTATION.history_frames]
    return noisy


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
    expected = int(cfg.batch_size) * world_size * int(cfg.gradient_accumulation_steps)
    if expected != int(cfg.effective_batch_size):
        raise ValueError(
            f"effective batch mismatch: {cfg.batch_size} x {world_size} x "
            f"{cfg.gradient_accumulation_steps} = {expected}, configured {cfg.effective_batch_size}"
        )
    if int(cfg.window_frames) != REPRESENTATION.window_frames or int(cfg.history_frames) != REPRESENTATION.history_frames:
        raise ValueError("window/history contract mismatch")
    if int(cfg.diffusion_steps) != REPRESENTATION.diffusion_steps:
        raise ValueError("diffusion schedule contract mismatch")
    dataset = PriorWindowDataset(
        str(cfg.repo_root), str(cfg.expert), partition=str(cfg.partition), limit=int(cfg.dataset_limit),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(cfg.seed))
    loader = DataLoader(
        dataset, batch_size=int(cfg.batch_size), sampler=sampler, drop_last=True,
        num_workers=int(cfg.num_workers), pin_memory=True,
    )
    model = build_expert(
        str(cfg.expert), init_checkpoint=cfg.init_checkpoint, dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads), num_layers=int(cfg.num_layers),
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
    optimizer = Adam(model.parameters(), lr=float(cfg.learning_rate))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    masked_gradient_max = 0.0
    updates = 0
    micro_steps = 0
    for batch in loader:
        clean = batch["x"].to(device, non_blocking=True)
        text = batch["text_embedding"].to(device, non_blocking=True)
        goals = batch["goals"].to(device, non_blocking=True)
        raw_progress = batch["progress"].to(device, non_blocking=True)
        denominator = raw_progress[:, 2:].clamp_min(1.0)
        progress = torch.cat((raw_progress[:, :2] / denominator, torch.log1p(denominator) / 10.0), dim=-1)
        timesteps = torch.randint(0, REPRESENTATION.diffusion_steps, (clean.shape[0],), device=device)
        noisy = _diffuse(clean, timesteps, torch.randn_like(clean))
        if str(cfg.expert) == "hoi":
            prediction = model(
                noisy, timesteps, text, batch["object_bps"].to(device, non_blocking=True), goals, progress,
            )
        else:
            prediction = model(
                noisy, timesteps, text, batch["scene_condition"].to(device, non_blocking=True), goals, progress,
            )
        prediction.retain_grad()
        loss = masked_reconstruction_loss(prediction, clean, str(cfg.expert))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {cfg.expert} loss")
        (loss / int(cfg.gradient_accumulation_steps)).backward()
        losses.append(float(loss.detach()))
        micro_steps += 1
        if str(cfg.expert) == "hsi":
            masked_gradient_max = max(masked_gradient_max, float(prediction.grad[:, :, 216:].abs().max()))
        if micro_steps % int(cfg.gradient_accumulation_steps) == 0:
            key_gradient = (
                model.module.network.motion_input.weight.grad
                if str(cfg.expert) == "hoi"
                else model.module.network.input.weight.grad
            )
            if key_gradient is None or not torch.isfinite(key_gradient).all() or not torch.any(key_gradient != 0):
                raise FloatingPointError("missing or invalid key gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates >= int(cfg.optimizer_updates):
                break
    torch.cuda.synchronize(device)
    if updates != int(cfg.optimizer_updates):
        raise RuntimeError(f"completed {updates} optimizer updates, expected {cfg.optimizer_updates}")
    local = torch.tensor(
        [losses[-1], masked_gradient_max, torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.float64, device=device,
    )
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    if rank == 0:
        metrics_path = Path(str(cfg.metrics_path))
        if metrics_path.exists():
            raise FileExistsError(f"refusing to overwrite {metrics_path}")
        values = [entry.cpu().tolist() for entry in gathered]
        metrics = {
            "schema_version": 1, "status": "stable", "expert": str(cfg.expert), "seed": int(cfg.seed),
            "world_size": world_size, "micro_batch_per_gpu": int(cfg.batch_size),
            "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps), "effective_batch_size": expected,
            "optimizer_updates": updates, "last_loss_by_rank": [value[0] for value in values],
            "loss_finite": all(np.isfinite(value[0]) for value in values),
            "masked_object_contact_output_gradient_max_by_rank": [value[1] for value in values],
            "key_gradient_present": True,
            "parameter_count": sum(parameter.numel() for parameter in model.module.parameters()),
            "peak_memory_allocated_bytes_by_rank": [int(value[2]) for value in values],
            "peak_memory_reserved_bytes_by_rank": [int(value[3]) for value in values],
            "max_peak_memory_allocated_bytes": int(max(value[2] for value in values)),
            "max_peak_memory_reserved_bytes": int(max(value[3] for value in values)),
            "representation": REPRESENTATION.as_dict(),
            "initialization": "random",
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, metrics_path)
    torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="config_prior_smoke")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < int(cfg.num_gpus):
        raise RuntimeError(f"requires {cfg.num_gpus} visible CUDA devices")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    torch.multiprocessing.spawn(_worker, args=(cfg,), nprocs=int(cfg.num_gpus), join=True)


if __name__ == "__main__":
    main()
