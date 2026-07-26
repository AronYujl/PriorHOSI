#!/usr/bin/env python3
"""Registered no-update real-data GPU smoke for D2-AC0."""

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
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.interaction_adapter import (  # noqa: E402
    ADAPTER_PARAMETER_COUNT,
    ASSIGNMENT_SHA256,
    BPS_SHA256,
)
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AC, build_expert  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    LOSS_KEYS,
    _d2ac_gradient_audit,
    _move_batch,
    _state_dict_sha256,
    _validate_author_update_execution_host,
    _validate_d2ac_contract,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID = "p1-hoi-d2ac-gpu-smoke-r1-s42-20260726"
FORMAL_RUN_ID = "p1-hoi-d2ac-interaction-adapter-s42-20260726"
EXPECTED_BATCH_SIZE = 8
REGISTERED_TIMESTEPS = (0, 249, 499, 0, 249, 499, 0, 499)
FORMAL_MICRO_BATCH_PER_GPU = 512


def _atomic_json(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _resolved_config(repo: Path):
    base = OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml")
    d2ac = OmegaConf.load(repo / "code/config/config_train_hoi_prior_d2ac.yaml")
    cfg = OmegaConf.merge(base, d2ac)
    cfg.repo_root = str(repo)
    cfg.split_manifest = str(
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    cfg.run_id = FORMAL_RUN_ID
    cfg.output_dir = str(repo / "results/experiments" / FORMAL_RUN_ID)
    cfg.checkpoint_dir = str(Path(cfg.output_dir) / "checkpoints")
    cfg.metrics_path = str(Path(cfg.output_dir) / "metrics.json")
    cfg.state_path = str(Path(cfg.output_dir) / "training_state.json")
    OmegaConf.resolve(cfg)
    return cfg


def _gpu_contention() -> dict:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(query, text=True)
    gpus = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 6:
            gpus.append({
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
                "pstate": fields[5],
            })
    processes = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    ).stdout.splitlines()
    processes = [line.strip() for line in processes if line.strip()]
    return {"gpus": gpus, "compute_processes": processes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError("registered D2-AC GPU smoke batch size is exactly 8")
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-AC GPU smoke run id must be exactly {RUN_ID}")
    if socket.gethostname() != "node01":
        raise RuntimeError("D2-AC GPU smoke is restricted to infbagel-4gpu/node01")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-AC GPU smoke requires INFBAGEL_WORKER_EXPERT=hoi")
    expected_python = Path("/home/yujinlun/data/envs/infbagel/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(f"unexpected worker Python: {sys.executable}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("D2-AC GPU smoke requires exactly four visible CUDA devices")

    cfg = _resolved_config(repo)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2ac_contract(cfg, 4)
    _validate_author_update_execution_host(cfg)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    contention_before = _gpu_contention()

    dataset = PriorWindowDataset(
        str(repo), "hoi", partition="train", split_manifest=str(cfg.split_manifest)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=0, pin_memory=True,
    )
    raw_batch = next(iter(loader))
    model = build_expert(
        "hoi", init_checkpoint=None, dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads), num_layers=int(cfg.num_layers),
        architecture_variant=HOI_ARCHITECTURE_D2AC,
    )
    initial_model_sha256 = _state_dict_sha256(model.state_dict())
    model = model.to(device).train()
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    batch = _move_batch(raw_batch, device)
    norm = np.load(repo / "data/train/norm.npy")
    minimum = torch.as_tensor(norm[0], device=device, dtype=torch.float32)
    maximum = torch.as_tensor(norm[1], device=device, dtype=torch.float32)
    object_minimum = torch.as_tensor(norm[2], device=device, dtype=torch.float32)
    object_maximum = torch.as_tensor(norm[3], device=device, dtype=torch.float32)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    clean = batch["x"]
    timesteps = torch.tensor(REGISTERED_TIMESTEPS, device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    noisy = diffusion.q_sample(clean, timesteps, noise)
    torch.cuda.reset_peak_memory_stats(device)
    prediction = model(
        noisy, timesteps, batch["text_embedding"], batch["object_bps"],
        batch["goals"], normalize_progress(batch["progress"]),
    )
    prediction.retain_grad()
    losses = hoi_training_losses(
        prediction, clean, batch["goals"], batch["rest_human_offsets"], parents,
        minimum, maximum, object_minimum, object_maximum, batch["terminal_window"],
        batch["rest_object_points"], batch["world_to_local_rotation"],
        batch["object_rotation_reference"], fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight), goal_weight=float(cfg.goal_weight),
        fk_foot_temporal_routing=True, routed_foot_residual_multiplier=1.0,
    )
    loss_values = {key: float(losses[key].detach().item()) for key in LOSS_KEYS}
    if not all(math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError(f"non-finite D2-AC smoke loss: {loss_values}")
    losses["total"].backward()
    torch.cuda.synchronize(device)
    initial_audit = _d2ac_gradient_audit(model, require_adapter_paths=False)
    initial_gradients = {
        "motion_input_weight": _gradient_record(model.network.motion_input.weight.grad),
        "transformer_first_parameter": _gradient_record(
            next(model.network.transformer.parameters()).grad
        ),
        "prediction": _gradient_record(prediction.grad),
    }

    # Test-only nonzero gate probe.  It is never stepped or serialized.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.network.interaction_adapter.alpha.copy_(torch.atanh(
            torch.tensor(0.1, device=device)
        ))
    probe_prediction = model(
        noisy, timesteps, batch["text_embedding"], batch["object_bps"],
        batch["goals"], normalize_progress(batch["progress"]),
    )
    probe_losses = hoi_training_losses(
        probe_prediction, clean, batch["goals"], batch["rest_human_offsets"], parents,
        minimum, maximum, object_minimum, object_maximum, batch["terminal_window"],
        batch["rest_object_points"], batch["world_to_local_rotation"],
        batch["object_rotation_reference"], fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight), goal_weight=float(cfg.goal_weight),
        fk_foot_temporal_routing=True, routed_foot_residual_multiplier=1.0,
    )
    probe_losses["total"].backward()
    torch.cuda.synchronize(device)
    probe_audit = _d2ac_gradient_audit(model, require_adapter_paths=True)
    model.network.interaction_adapter.clear_diagnostic_state()
    device_total = torch.cuda.get_device_properties(device).total_memory
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    contention_after = _gpu_contention()
    result = {
        "schema_version": 1,
        "status": "stable",
        "run_id": args.run_id,
        "formal_run_id": FORMAL_RUN_ID,
        "subphase": "1B-D2-AC0-gpu-smoke",
        "seed": 42,
        "hostname": socket.gethostname(),
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(device),
        "visible_cuda_devices": torch.cuda.device_count(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True,
        ).strip(),
        "batch_size": args.batch_size,
        "timesteps": list(REGISTERED_TIMESTEPS),
        "losses": loss_values,
        "loss_finite": True,
        "initial_gradients": initial_gradients,
        "initial_alpha_gradient": initial_audit,
        "probe_adapter_gradients": probe_audit,
        "initialization": "random",
        "initial_model_state_sha256": initial_model_sha256,
        "adapter_parameter_count": ADAPTER_PARAMETER_COUNT,
        "bps_sha256": BPS_SHA256,
        "assignment_sha256": ASSIGNMENT_SHA256,
        "smoke_cross_attention_score_shape": [
            args.batch_size, 16, 3, 4, 16,
        ],
        "smoke_cross_attention_score_elements": (
            args.batch_size * 16 * 3 * 4 * 16
        ),
        "registered_formal_cross_attention_score_shape_estimate": [
            FORMAL_MICRO_BATCH_PER_GPU, 16, 3, 4, 16,
        ],
        "registered_formal_cross_attention_score_elements_estimate": (
            FORMAL_MICRO_BATCH_PER_GPU * 16 * 3 * 4 * 16
        ),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": device_total,
        "memory_headroom_bytes": device_total - peak_reserved,
        "cuda_timing_synchronized": True,
        "contention_before": contention_before,
        "contention_after": contention_after,
        "checkpoint_selected": False,
        "consistency_started": False,
    }
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
