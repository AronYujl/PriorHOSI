#!/usr/bin/env python3
"""One-GPU, no-update real-data functional smoke for the D2-AI/D2-AJ arms.

Preregistered in docs/EXPECTED_PLAN.md,
"2026-08-03 Phase 1B D2-AI 全预算与 D2-AJ 目标条件通路（双臂，用户批准）".

Exercises the real training data path for both arms with a finite forward and
backward pass, records gradient presence, memory and the output API, and proves
the two arms differ only in the goal conditioning pathway.  Creates no
optimizer, performs no update and writes no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AJ,
    build_expert,
)
from train_hoi_prior import (  # noqa: E402
    LOSS_KEYS,
    _locked_loss_weights,
    _model_config,
    _move_batch,
    _state_dict_sha256,
    _validate_fk_foot_temporal_routing_mode,
)

BATCH_SIZE = 8
TIMESTEPS = (0, 249, 499, 0, 249, 499, 0, 499)
ARMS = {
    "d2ai": ("config_train_hoi_prior_d2ai.yaml", HOI_ARCHITECTURE_BASE),
    "d2aj": ("config_train_hoi_prior_d2aj.yaml", HOI_ARCHITECTURE_D2AJ),
}


def _seed(value: int = 42) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _config(arm: str):
    name, _ = ARMS[arm]
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml"),
        OmegaConf.load(ROOT / "code/config" / name),
    )
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(
        ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    OmegaConf.resolve(cfg)
    return cfg


def _run_arm(arm: str, device: torch.device) -> dict:
    cfg = _config(arm)
    _validate_fk_foot_temporal_routing_mode(cfg)
    expected_variant = ARMS[arm][1]
    assert str(cfg.hoi_architecture_variant) == expected_variant, arm

    _seed()
    dataset = PriorWindowDataset(
        str(ROOT),
        "hoi",
        partition="train",
        split_manifest=str(cfg.split_manifest),
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        drop_last=True, pin_memory=True,
    )
    batch = _move_batch(next(iter(loader)), device)
    batch.update({
        "position_minimum": torch.as_tensor(
            dataset.minimum, device=device, dtype=torch.float32,
        ),
        "position_maximum": torch.as_tensor(
            dataset.maximum, device=device, dtype=torch.float32,
        ),
        "object_minimum": torch.as_tensor(
            dataset.object_minimum, device=device, dtype=torch.float32,
        ),
        "object_maximum": torch.as_tensor(
            dataset.object_maximum, device=device, dtype=torch.float32,
        ),
    })
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )

    _seed()
    model = build_expert(
        "hoi",
        dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        architecture_variant=expected_variant,
        bps_path=str(ROOT / "code/bps.pt"),
    ).to(device)
    model.train()

    initial_sha = _state_dict_sha256(model.state_dict())
    parameters = sum(p.numel() for p in model.parameters())
    network = model.network

    diffusion = GaussianDiffusion(timesteps=int(cfg.diffusion_steps)).to(device)
    timesteps = torch.tensor(TIMESTEPS[:BATCH_SIZE], device=device, dtype=torch.long)
    clean = batch["x"]
    noise = torch.randn_like(clean)
    noisy = diffusion.q_sample(clean, timesteps, noise)

    predicted = network(
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        batch["progress"],
    )

    weights = _locked_loss_weights(cfg)
    losses = hoi_training_losses(
        predicted,
        clean,
        batch["goals"],
        batch["rest_human_offsets"],
        parents,
        batch["position_minimum"],
        batch["position_maximum"],
        batch["object_minimum"],
        batch["object_maximum"],
        batch["terminal_window"],
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
        fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight),
        goal_weight=float(cfg.goal_weight),
        fk_foot_temporal_routing=bool(cfg.fk_foot_temporal_routing),
        routed_foot_residual_multiplier=float(
            cfg.get("routed_foot_residual_multiplier", 1.0)
        ),
    )
    losses["total"].backward()

    grads = {
        name: float(p.grad.detach().abs().sum().item())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    goal_modules = sorted(
        name for name in grads
        if any(key in name for key in
               ("goal_progress", "pelvis_goal", "object_goal", "progress"))
    )

    return {
        "arm": arm,
        "architecture_variant": expected_variant,
        "parameters": parameters,
        "initial_state_sha256": initial_sha,
        "condition_tokens": int(network.condition_tokens),
        "position_tokens": int(network.position.shape[1]),
        "model_config": _model_config(cfg),
        "max_processed_windows": int(cfg.max_processed_windows),
        "optimizer_updates_planned": int(cfg.max_processed_windows)
        // int(cfg.effective_batch_size),
        "loss_weights": weights,
        "losses": {key: float(losses[key].detach().item()) for key in LOSS_KEYS
                   if key in losses},
        "total_loss": float(losses["total"].detach().item()),
        "all_losses_finite": all(
            bool(torch.isfinite(value).all()) for value in losses.values()
        ),
        "output_shape": list(predicted.shape),
        "output_finite": bool(torch.isfinite(predicted).all()),
        "parameters_with_grad": len(grads),
        "parameters_total": sum(1 for _ in model.parameters()),
        "all_grads_finite": all(np.isfinite(v) for v in grads.values()),
        "zero_grad_parameters": sorted(k for k, v in grads.items() if v == 0.0),
        "goal_pathway_modules_with_grad": goal_modules,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the functional smoke requires a CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)

    results = {}
    for arm in ("d2ai", "d2aj"):
        torch.cuda.reset_peak_memory_stats(device)
        results[arm] = _run_arm(arm, device)
        print(f"[{arm}] params={results[arm]['parameters']:,} "
              f"tokens={results[arm]['condition_tokens']} "
              f"total_loss={results[arm]['total_loss']:.6f} "
              f"finite={results[arm]['all_losses_finite']} "
              f"grads={results[arm]['parameters_with_grad']}"
              f"/{results[arm]['parameters_total']} "
              f"peak_mem={results[arm]['peak_memory_allocated_bytes']/2**30:.2f}GiB")

    payload = {
        "schema_version": 1,
        "lifecycle": "d2ai_d2aj_single_gpu_functional_smoke",
        "reportable": False,
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "timesteps": list(TIMESTEPS[:BATCH_SIZE]),
        "real_data": True,
        "partition": "train",
        "gpu_model": torch.cuda.get_device_name(device),
        "python": str(Path(sys.executable).resolve()),
        "arms": results,
        "cross_arm": {
            "parameter_delta": results["d2aj"]["parameters"]
            - results["d2ai"]["parameters"],
            "expected_parameter_delta": 525312,
            "condition_token_delta": results["d2aj"]["condition_tokens"]
            - results["d2ai"]["condition_tokens"],
            "same_budget": results["d2ai"]["max_processed_windows"]
            == results["d2aj"]["max_processed_windows"],
            "same_loss_weights": results["d2ai"]["loss_weights"]
            == results["d2aj"]["loss_weights"],
        },
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

    cross = payload["cross_arm"]
    ok = (
        all(results[a]["all_losses_finite"] and results[a]["output_finite"]
            and results[a]["all_grads_finite"] for a in results)
        and cross["parameter_delta"] == cross["expected_parameter_delta"]
        and cross["same_budget"] and cross["same_loss_weights"]
        and cross["condition_token_delta"] == 2
    )
    print(f"\nwrote {out}")
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
