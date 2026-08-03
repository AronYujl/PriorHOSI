#!/usr/bin/env python3
"""Measure single-arm vs concurrent-arm training throughput for D2-AI/D2-AJ.

The preregistration fixes a contention rule: each arm is compared against
sealed D2-X's 3243.04 windows/s, and anything below 2757 (-15%) must be
recorded as contention.  Two concurrent four-GPU arms change the compute and
communication profile, so the sealed execution profile may not simply be
reused.

Runs a bounded number of real optimizer-free forward/backward steps at the
registered micro-batch on one GPU per arm, first alone and then concurrently,
and reports the ratio.  Creates no optimizer state and writes no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import build_expert  # noqa: E402
from train_hoi_prior import _move_batch  # noqa: E402

MICRO_BATCH = 512
WARMUP_STEPS = 3
MEASURED_STEPS = 10
SEALED_D2X_WINDOWS_PER_SECOND = 3243.036186840841
CONTENTION_FLOOR = 2757.0


def _seed(value: int = 42) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _config(arm: str):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml"),
        OmegaConf.load(ROOT / f"code/config/config_train_hoi_prior_{arm}.yaml"),
    )
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(
        ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    OmegaConf.resolve(cfg)
    return cfg


def measure(arm: str, device_index: int, output: Path) -> None:
    """Run in a dedicated process so two arms can be timed concurrently."""
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.init()
    torch.cuda.set_device(device)
    cfg = _config(arm)
    _seed()

    dataset = PriorWindowDataset(
        str(ROOT), "hoi", partition="train",
        split_manifest=str(cfg.split_manifest),
    )
    loader = DataLoader(
        dataset, batch_size=MICRO_BATCH, shuffle=True, num_workers=int(cfg.num_workers),
        drop_last=True, pin_memory=True, persistent_workers=True,
    )
    model = build_expert(
        "hoi", dim_model=int(cfg.dim_model), num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        architecture_variant=str(cfg.hoi_architecture_variant),
        bps_path=str(ROOT / "code/bps.pt"),
    ).to(device).train()
    diffusion = GaussianDiffusion(timesteps=int(cfg.diffusion_steps)).to(device)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    bounds = {
        "position_minimum": torch.as_tensor(dataset.minimum, device=device, dtype=torch.float32),
        "position_maximum": torch.as_tensor(dataset.maximum, device=device, dtype=torch.float32),
        "object_minimum": torch.as_tensor(dataset.object_minimum, device=device, dtype=torch.float32),
        "object_maximum": torch.as_tensor(dataset.object_maximum, device=device, dtype=torch.float32),
    }

    iterator = iter(loader)
    durations = []
    for step in range(WARMUP_STEPS + MEASURED_STEPS):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batch = _move_batch(raw, device)
        batch.update(bounds)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        clean = batch["x"]
        timesteps = torch.randint(0, int(cfg.diffusion_steps), (clean.shape[0],), device=device)
        noisy = diffusion.q_sample(clean, timesteps, torch.randn_like(clean))
        predicted = model.network(
            noisy, timesteps, batch["text_embedding"], batch["object_bps"],
            batch["goals"], batch["progress"],
        )
        losses = hoi_training_losses(
            predicted, clean, batch["goals"], batch["rest_human_offsets"], parents,
            batch["position_minimum"], batch["position_maximum"],
            batch["object_minimum"], batch["object_maximum"],
            batch["terminal_window"], batch["rest_object_points"],
            batch["world_to_local_rotation"], batch["object_rotation_reference"],
            fk_weight=float(cfg.fk_weight),
            object_surface_weight=float(cfg.object_surface_weight),
            velocity_weight=float(cfg.velocity_weight),
            goal_weight=float(cfg.goal_weight),
            fk_foot_temporal_routing=bool(cfg.fk_foot_temporal_routing),
        )
        losses["total"].backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        if step >= WARMUP_STEPS:
            durations.append(time.perf_counter() - started)

    seconds = float(np.mean(durations))
    payload = {
        "arm": arm,
        "device_index": device_index,
        "micro_batch": MICRO_BATCH,
        "measured_steps": MEASURED_STEPS,
        "mean_step_seconds": seconds,
        "single_gpu_windows_per_second": MICRO_BATCH / seconds,
        # Four ranks process four micro-batches per optimizer step in parallel,
        # so the four-GPU rate is the per-GPU rate times four.
        "projected_four_gpu_windows_per_second": 4.0 * MICRO_BATCH / seconds,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm")
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.arm:
        measure(args.arm, args.device_index, Path(args.output))
        return 0
    raise SystemExit("run via tools/run_d2ai_d2aj_contention.sh")


if __name__ == "__main__":
    raise SystemExit(main())
