#!/usr/bin/env python3
"""Run the registered no-update D2-AB real-data GPU forward/backward smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.d2ab import (  # noqa: E402
    D2ABPriorWindowDataset,
    d2ab_hoi_training_losses,
    sha256_file,
)
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.models import build_expert  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    LOSS_KEYS,
    _move_batch,
    _state_dict_sha256,
    _validate_author_update_execution_host,
    _validate_d2ab_contract,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID = "p1-hoi-d2ab-gpu-smoke-s42-20260725"
FORMAL_RUN_ID = "p1-hoi-d2ab-predicted-support-no-slip-s42-20260725"
EXPECTED_INITIAL_MODEL_SHA256 = (
    "ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e"
)
EXPECTED_METADATA_SHA256 = (
    "807978580221910ad00260c2dff4f33ddacbb1bf72bad7443bf21ac48f31f079"
)


def _atomic_json(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gradient_record(value: Optional[torch.Tensor]) -> dict:
    if value is None:
        return {"present": False, "finite": False, "nonzero": False, "norm": None}
    finite = bool(torch.isfinite(value).all())
    norm = float(value.detach().float().norm().item())
    return {
        "present": True,
        "finite": finite,
        "nonzero": finite and norm > 0.0,
        "norm": norm,
    }


def _support_record(support: torch.Tensor) -> dict:
    """Summarize predicted support and reject only degenerate distributions."""
    values = support.detach().float().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise FloatingPointError("D2-AB smoke support is empty or non-finite")
    quantiles = torch.quantile(
        values,
        torch.as_tensor([0.05, 0.50, 0.95], device=values.device),
    )
    q05, q50, q95 = (float(item.item()) for item in quantiles)
    minimum = float(values.min().item())
    maximum = float(values.max().item())
    mean = float(values.mean().item())
    fraction_gt_005 = float((values > 0.05).float().mean().item())
    fraction_lt_095 = float((values < 0.95).float().mean().item())
    noncollapsed = (
        0.0 <= minimum <= maximum <= 1.0
        and maximum - minimum > 1e-3
        and fraction_gt_005 >= 0.05
        and fraction_lt_095 >= 0.05
    )
    return {
        "minimum": minimum,
        "q05": q05,
        "median": q50,
        "q95": q95,
        "maximum": maximum,
        "mean": mean,
        "fraction_gt_0.05": fraction_gt_005,
        "fraction_lt_0.95": fraction_lt_095,
        "noncollapsed": bool(noncollapsed),
    }


def _resolved_config(repo: Path):
    base = OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml")
    d2ab = OmegaConf.load(repo / "code/config/config_train_hoi_prior_d2ab.yaml")
    cfg = OmegaConf.merge(base, d2ab)
    cfg.repo_root = str(repo)
    cfg.split_manifest = str(
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    cfg.d2ab_support_metadata_path = str(
        repo / "experiments/metadata/omomo_hoi_d2ab_train_support_seed42.json"
    )
    cfg.output_dir = str(repo / "results/experiments" / FORMAL_RUN_ID)
    cfg.checkpoint_dir = str(Path(cfg.output_dir) / "checkpoints")
    cfg.metrics_path = str(Path(cfg.output_dir) / "metrics.json")
    cfg.state_path = str(Path(cfg.output_dir) / "training_state.json")
    OmegaConf.resolve(cfg)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if args.batch_size != 8:
        raise ValueError("registered D2-AB GPU smoke batch size is exactly 8")
    if socket.gethostname() != "node01":
        raise RuntimeError("D2-AB GPU smoke is restricted to the HOI worker node01")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-AB GPU smoke requires INFBAGEL_WORKER_EXPERT=hoi")
    expected_python = Path("/home/yujinlun/data/envs/infbagel/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(f"unexpected worker Python: {sys.executable}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("D2-AB GPU smoke requires exactly four visible CUDA devices")

    cfg = _resolved_config(repo)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2ab_contract(cfg, 4)
    _validate_author_update_execution_host(cfg)
    metadata_path = Path(str(cfg.d2ab_support_metadata_path)).resolve()
    if (
        str(cfg.run_id) != FORMAL_RUN_ID
        or str(cfg.d2ab_support_metadata_sha256) != EXPECTED_METADATA_SHA256
        or sha256_file(metadata_path) != EXPECTED_METADATA_SHA256
    ):
        raise ValueError("D2-AB formal identity/support metadata mismatch")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    dataset = D2ABPriorWindowDataset(
        str(repo),
        "hoi",
        partition="train",
        split_manifest=str(cfg.split_manifest),
        support_metadata_path=str(cfg.d2ab_support_metadata_path),
        support_metadata_sha256=str(cfg.d2ab_support_metadata_sha256),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    model = build_expert(
        "hoi",
        init_checkpoint=None,
        dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
    )
    initial_model_sha256 = _state_dict_sha256(model.state_dict())
    if initial_model_sha256 != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(
            f"random initialization hash mismatch: {initial_model_sha256}"
        )

    raw_batch = next(iter(loader))
    model = model.to(device)
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    batch = _move_batch(raw_batch, device)
    norm = np.load(repo / "data/train/norm.npy")
    minimum = torch.as_tensor(norm[0], device=device, dtype=torch.float32)
    maximum = torch.as_tensor(norm[1], device=device, dtype=torch.float32)
    object_minimum = torch.as_tensor(norm[2], device=device, dtype=torch.float32)
    object_maximum = torch.as_tensor(norm[3], device=device, dtype=torch.float32)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True),
        device=device,
        dtype=torch.long,
    )

    clean = batch["x"]
    registered_timesteps = torch.tensor(
        [0, 249, 499, 0, 249, 499, 0, 499],
        device=device,
        dtype=torch.long,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    noisy = diffusion.q_sample(clean, registered_timesteps, noise)
    torch.cuda.reset_peak_memory_stats(device)
    prediction = model(
        noisy,
        registered_timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
    )
    prediction.retain_grad()
    losses = d2ab_hoi_training_losses(
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
        batch["d2ab_floor_m"],
        fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight),
        goal_weight=float(cfg.goal_weight),
    )
    loss_values = {key: float(losses[key].detach().item()) for key in LOSS_KEYS}
    if not all(math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError(f"non-finite D2-AB smoke loss: {loss_values}")
    support_record = _support_record(losses["d2ab_support_pair"])
    if not support_record["noncollapsed"]:
        raise FloatingPointError(f"D2-AB smoke support collapsed: {support_record}")
    losses["total"].backward()
    torch.cuda.synchronize(device)

    transformer_parameter = next(model.network.transformer.parameters())
    gradients = {
        "motion_input_weight": _gradient_record(model.network.motion_input.weight.grad),
        "motion_output_weight": _gradient_record(model.network.output.weight.grad),
        "transformer_first_parameter": _gradient_record(transformer_parameter.grad),
        "prediction_root_translation": _gradient_record(prediction.grad[..., :3]),
        "prediction_joint_rotations": _gradient_record(prediction.grad[..., 84:216]),
    }
    if not all(
        record["present"] and record["finite"] and record["nonzero"]
        for record in gradients.values()
    ):
        raise FloatingPointError(f"invalid D2-AB smoke gradients: {gradients}")

    device_total = torch.cuda.get_device_properties(device).total_memory
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True,
    ).strip()
    result = {
        "schema_version": 1,
        "status": "stable",
        "run_id": RUN_ID,
        "subphase": "1B-D2-AB0-gpu-smoke",
        "seed": 42,
        "hostname": socket.gethostname(),
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(device),
        "visible_cuda_devices": torch.cuda.device_count(),
        "git_commit": git_commit,
        "batch_size": args.batch_size,
        "timesteps": registered_timesteps.cpu().tolist(),
        "support_metadata_path": str(metadata_path),
        "support_metadata_sha256": EXPECTED_METADATA_SHA256,
        "floor_min_m": float(batch["d2ab_floor_m"].min().item()),
        "floor_max_m": float(batch["d2ab_floor_m"].max().item()),
        "support": support_record,
        "losses": loss_values,
        "loss_finite": True,
        "gradients": gradients,
        "key_gradient_present": True,
        "initialization": "random",
        "initial_model_state_sha256": initial_model_sha256,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": device_total,
        "memory_headroom_bytes": device_total - peak_reserved,
        "checkpoint_selected": False,
        "consistency_started": False,
    }
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
