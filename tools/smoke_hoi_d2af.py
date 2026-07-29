#!/usr/bin/env python3
"""Registered one-GPU, no-update real-data functional smoke for D2-AF0."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
from datetime import datetime
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
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SENTINELS,
    SQRT_ALPHA_BAR_SHA256,
    tensor_sha256,
)
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AF, build_expert  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT,
    build_sparse_relation_geometry,
    diffusion_reliability_contract_metadata,
)
from tools.smoke_hoi_d2ac import (  # noqa: E402
    _atomic_json,
    _gpu_contention,
    _gradient_record,
)
from tools.smoke_hoi_d2ae import (  # noqa: E402
    _atomic_text,
    _model_arguments,
    _sha256_file,
    _tensor_summary,
)
from train_hoi_prior import (  # noqa: E402
    LOSS_KEYS,
    _d2af_formal_source_contract,
    _d2ae_gradient_audit,
    _move_batch,
    _state_dict_sha256,
    _validate_author_update_execution_host,
    _validate_d2af_contract,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-gpu-functional-smoke"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
EXPECTED_BATCH_SIZE = 8
REGISTERED_TIMESTEPS = (0, 249, 499, 0, 249, 499, 0, 499)
DISTINCT_TIMESTEPS = (0, 249, 499)
EXPECTED_INITIAL_MODEL_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
SCALING_MAX_ABS_TOLERANCE = 1.0e-6
FAILURE_CLASSIFICATION = "diffusion-reliability-contract-failure-stop"


def _validate_actual_run_id(run_id: str) -> str:
    match = RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or match.group("date") != actual_date:
        raise ValueError(
            "D2-AF functional smoke run id must use the locked stem and actual date"
        )
    return match.group("date")


def _formal_run_id_for_date(date: str) -> str:
    return f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{date}"


def _resolved_config(repo: Path, formal_run_id: str):
    cfg = OmegaConf.merge(
        OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml"),
        OmegaConf.load(repo / "code/config/config_train_hoi_prior_d2af.yaml"),
    )
    cfg.repo_root = str(repo)
    cfg.split_manifest = str(
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    cfg.run_id = formal_run_id
    cfg.output_dir = str(repo / "results/experiments" / formal_run_id)
    cfg.checkpoint_dir = str(Path(cfg.output_dir) / "checkpoints")
    cfg.metrics_path = str(Path(cfg.output_dir) / "metrics.json")
    cfg.state_path = str(Path(cfg.output_dir) / "training_state.json")
    OmegaConf.resolve(cfg)
    return cfg


def _resolved_workload_config(
    cfg,
    *,
    repo: Path,
    run_id: str,
    expected_commit: str,
    formal_source_contract: dict,
    output: Path,
    resolved_config_output: Path,
) -> str:
    schedule = diffusion_reliability_contract_metadata()["schedule"]
    value = {
        "schema_version": 1,
        "lifecycle": "d2af_single_gpu_functional_smoke",
        "run_id": run_id,
        "formal_run_id": str(cfg.run_id),
        "expected_git_commit": expected_commit,
        "formal_source_contract": formal_source_contract,
        "repo_root": str(repo),
        "python": str(Path(sys.executable).resolve()),
        "output": str(output.resolve()),
        "resolved_config_output": str(resolved_config_output.resolve()),
        "workload": {
            "host": "node01",
            "visible_gpus": 1,
            "device": "cuda:0",
            "gpu_model": "RTX 3090",
            "real_data": True,
            "partition": "train",
            "batch_size": EXPECTED_BATCH_SIZE,
            "timesteps": list(DISTINCT_TIMESTEPS),
            "mixed_batch_timesteps": list(REGISTERED_TIMESTEPS),
            "seed": 42,
            "random_initialization": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "checkpoint_loads": 0,
            "checkpoint_writes": 0,
            "relation_source": "current_noisy_state_only",
            "reliability": "sqrt_alpha_bar[current_timestep]",
        },
        "contracts": {
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
            "expected_initial_model_state_sha256": EXPECTED_INITIAL_MODEL_SHA256,
            "schedule": schedule,
            "mixed_batch_scaling_tolerance": SCALING_MAX_ABS_TOLERANCE,
            "per_timestep_gradient_audit": list(DISTINCT_TIMESTEPS),
        },
        "formal_training_config_reference": OmegaConf.to_container(
            cfg, resolve=True,
        ),
    }
    resolved = OmegaConf.to_yaml(OmegaConf.create(value), resolve=True)
    if "${" in resolved:
        raise RuntimeError("D2-AF functional smoke resolved config is incomplete")
    return resolved


def _verify_worker(repo: Path, expected_commit: str) -> dict:
    if socket.gethostname() != "node01":
        raise RuntimeError("D2-AF functional smoke is restricted to node01")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-AF functional smoke requires INFBAGEL_WORKER_EXPERT=hoi")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if not configured_python or not Path(configured_python).is_absolute():
        raise RuntimeError("D2-AF functional smoke requires an absolute INFBAGEL_PYTHON")
    if Path(sys.executable).resolve() != Path(configured_python).resolve():
        raise RuntimeError(
            f"worker Python mismatch: {sys.executable} != {configured_python}"
        )
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True,
    ).strip()).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
    ).splitlines()
    if root != repo or commit != expected_commit or status:
        raise RuntimeError(
            f"D2-AF worker Git identity mismatch: root={root}, commit={commit}, "
            f"status={status[:20]}"
        )
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "worktree_clean": True,
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _losses(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    batch: dict,
    parents: torch.Tensor,
    cfg,
) -> dict:
    return hoi_training_losses(
        prediction,
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
        fk_foot_temporal_routing=True,
        routed_foot_residual_multiplier=1.0,
    )


def _forward(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    batch: dict,
) -> torch.Tensor:
    return model(
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
        **_model_arguments(batch, noisy),
    )


def _activated_gradient_probe(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
    batch: dict,
    parents: torch.Tensor,
    cfg,
) -> dict:
    model.zero_grad(set_to_none=True)
    timesteps = torch.full(
        (clean.shape[0],),
        timestep,
        device=clean.device,
        dtype=torch.long,
    )
    noisy = diffusion.q_sample(clean, timesteps, noise)
    prediction = _forward(model, noisy, timesteps, batch)
    losses = _losses(prediction, clean, batch, parents, cfg)
    values = {key: float(losses[key].detach().item()) for key in LOSS_KEYS}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(
            f"non-finite D2-AF activated probe loss at timestep {timestep}"
        )
    losses["total"].backward()
    torch.cuda.synchronize(clean.device)
    return {
        "timestep": timestep,
        "losses": values,
        "gradients": _d2ae_gradient_audit(
            model, require_relation_paths=True,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="archive and validate the exact workload config without touching CUDA",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError("registered D2-AF functional smoke batch size is exactly 8")
    run_date = _validate_actual_run_id(args.run_id)
    formal_run_id = _formal_run_id_for_date(run_date)

    identity = _verify_worker(repo, args.expected_commit)
    formal_source_contract = _d2af_formal_source_contract(repo)
    cfg = _resolved_config(repo, formal_run_id)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2af_contract(
        cfg,
        4,
        require_eligibility_gate=False,
        require_performance_gate=False,
    )
    _validate_author_update_execution_host(cfg)
    resolved_yaml = _resolved_workload_config(
        cfg,
        repo=repo,
        run_id=args.run_id,
        expected_commit=args.expected_commit,
        formal_source_contract=formal_source_contract,
        output=args.output,
        resolved_config_output=args.resolved_config_output,
    )
    if args.resolve_only:
        _atomic_text(args.resolved_config_output, resolved_yaml)
    else:
        resolved_path = args.resolved_config_output.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                "D2-AF functional smoke requires a pre-archived resolved config"
            )
        if resolved_path.read_text(encoding="utf-8") != resolved_yaml:
            raise RuntimeError(
                "D2-AF functional smoke workload differs from archived config"
            )
    resolved_sha256 = _sha256_file(args.resolved_config_output)
    if args.resolve_only:
        print(json.dumps({
            "schema_version": 1,
            "status": "resolved-config-archived",
            "run_id": args.run_id,
            "resolved_config_path": str(args.resolved_config_output.resolve()),
            "resolved_config_sha256": resolved_sha256,
            "gpu_workload_started": False,
        }, indent=2, sort_keys=True), flush=True)
        return 0

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "D2-AF functional smoke requires exactly one visible CUDA device"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if "RTX 3090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("D2-AF functional smoke requires an RTX 3090")
    contention_before = _gpu_contention()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = build_expert(
        "hoi",
        init_checkpoint=None,
        dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        architecture_variant=HOI_ARCHITECTURE_D2AF,
    )
    initial_model_sha256 = _state_dict_sha256(model.state_dict())
    if initial_model_sha256 != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(
            "D2-AF seed-42 initial model-state hash differs from D2-AE"
        )
    if sum(parameter.numel() for parameter in model.parameters()) != TOTAL_PARAMETER_COUNT:
        raise RuntimeError("D2-AF total parameter count is not exact")
    model = model.to(device).train()

    dataset = PriorWindowDataset(
        str(repo),
        "hoi",
        partition="train",
        split_manifest=str(cfg.split_manifest),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    raw_batch = next(iter(loader))
    if "local_object_bps" in raw_batch:
        raise RuntimeError("D2-AF functional smoke received CPU dynamic geometry")
    batch = _move_batch(raw_batch, device)
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
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    clean = batch["x"]
    timesteps = torch.tensor(
        REGISTERED_TIMESTEPS, device=device, dtype=torch.long,
    )
    generator = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    noisy = diffusion.q_sample(clean, timesteps, noise)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        geometry = build_sparse_relation_geometry(
            noisy,
            batch["rest_object_points"],
            batch["world_to_local_rotation"],
            batch["object_rotation_reference"],
            batch["position_minimum"],
            batch["position_maximum"],
            batch["object_minimum"],
            batch["object_maximum"],
        )
        field = model.network.sparse_relation_field
        encoded = field.point_encoder(geometry["features"])
        pooled = torch.cat(
            (encoded.mean(dim=-2), encoded.amax(dim=-2)), dim=-1,
        )
        relation = field._relation_vectors(pooled)
        routed = relation.index_select(1, field.routing_slots)
        motion = model.network.motion_input(noisy)
        field.set_gate_override(0.1)
        field.set_capture(True)
        scheduled_motion = field(
            motion,
            noisy,
            **_model_arguments(batch, noisy),
            timesteps=timesteps,
        )
        scheduled_snapshot = field.snapshot()
        field.set_capture(False)
        field.set_rho_override(1.0)
        unit_motion = field(
            motion,
            noisy,
            **_model_arguments(batch, noisy),
            timesteps=timesteps,
        )
        field.set_rho_override(None)
        field.set_gate_override(None)
        raw_writeback = unit_motion - motion
        attenuated_writeback = scheduled_motion - motion
        rho = field.sqrt_alpha_bar.gather(0, timesteps).to(
            device=device,
            dtype=motion.dtype,
        ).reshape(clean.shape[0], 1, 1)
        scaling_max_abs = float(
            (attenuated_writeback - rho * raw_writeback).abs().amax().item()
        )

    relation_summaries = {
        "surface": _tensor_summary(geometry["surface"]),
        "features": _tensor_summary(geometry["features"]),
        "encoded_points": _tensor_summary(encoded),
        "pooled_blocks": _tensor_summary(pooled),
        "relation_vectors": _tensor_summary(relation),
        "routed_relation": _tensor_summary(routed),
        "raw_unit_rho_writeback": _tensor_summary(raw_writeback),
        "attenuated_writeback": _tensor_summary(attenuated_writeback),
    }
    if (
        not all(value["finite"] for value in relation_summaries.values())
        or not all(
            str(value["device"]).startswith("cuda")
            for value in relation_summaries.values()
        )
    ):
        raise FloatingPointError("D2-AF relation values are non-finite or not GPU-native")
    if scaling_max_abs > SCALING_MAX_ABS_TOLERANCE:
        raise FloatingPointError(
            f"D2-AF mixed-batch reliability scaling mismatch: {scaling_max_abs}"
        )
    if scheduled_snapshot is None:
        raise RuntimeError("D2-AF functional smoke did not capture reliability values")

    prediction = _forward(model, noisy, timesteps, batch)
    prediction.retain_grad()
    losses = _losses(prediction, clean, batch, parents, cfg)
    loss_values = {
        key: float(losses[key].detach().item()) for key in LOSS_KEYS
    }
    if not all(math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError(f"non-finite D2-AF smoke loss: {loss_values}")
    losses["total"].backward()
    torch.cuda.synchronize(device)
    initial_audit = _d2ae_gradient_audit(
        model, require_relation_paths=False,
    )
    initial_gradients = {
        "motion_input_weight": _gradient_record(
            model.network.motion_input.weight.grad,
        ),
        "transformer_first_parameter": _gradient_record(
            next(model.network.transformer.parameters()).grad,
        ),
        "prediction": _gradient_record(prediction.grad),
    }

    with torch.no_grad():
        field.alpha.copy_(torch.atanh(torch.tensor(0.1, device=device)))
    per_timestep_activated_gradients = [
        _activated_gradient_probe(
            model,
            diffusion,
            clean,
            noise,
            timestep,
            batch,
            parents,
            cfg,
        )
        for timestep in DISTINCT_TIMESTEPS
    ]
    with torch.no_grad():
        field.alpha.zero_()
    model.zero_grad(set_to_none=True)

    schedule_contract = diffusion_reliability_contract_metadata()["schedule"]
    field_schedule_sha256 = tensor_sha256(field.sqrt_alpha_bar)
    diffusion_schedule_sha256 = tensor_sha256(diffusion.sqrt_alpha_bar)
    if (
        field_schedule_sha256 != SQRT_ALPHA_BAR_SHA256
        or diffusion_schedule_sha256 != SQRT_ALPHA_BAR_SHA256
    ):
        raise RuntimeError("D2-AF GPU schedule hash mismatch")
    registered_rho = {
        str(timestep): float(field.sqrt_alpha_bar[timestep].item())
        for timestep in DISTINCT_TIMESTEPS
    }
    if registered_rho != {
        str(timestep): SQRT_ALPHA_BAR_SENTINELS[timestep]
        for timestep in DISTINCT_TIMESTEPS
    }:
        raise RuntimeError("D2-AF registered rho sentinels differ on GPU")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    contention_after = _gpu_contention()
    result = {
        "schema_version": 1,
        "status": "stable",
        "classification": "functional-smoke-passed",
        "run_id": args.run_id,
        "formal_run_id": formal_run_id,
        "subphase": "1B-D2-AF0-gpu-functional-smoke",
        "seed": 42,
        "identity": identity,
        "formal_source_contract": formal_source_contract,
        "resolved_config_path": str(args.resolved_config_output.resolve()),
        "resolved_config_sha256": resolved_sha256,
        "resolved_config_has_unresolved_interpolation": False,
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(device),
        "visible_cuda_devices": torch.cuda.device_count(),
        "batch_size": args.batch_size,
        "timesteps": list(DISTINCT_TIMESTEPS),
        "mixed_batch_timesteps": list(REGISTERED_TIMESTEPS),
        "relation_source": "current_noisy_state_only",
        "relation_build_device": "cuda:0",
        "relation_gpu_only": True,
        "relation_intermediates": relation_summaries,
        "reliability_scaling": {
            "rho_per_sample": rho.reshape(-1).detach().float().tolist(),
            "raw_writeback_l2_per_sample": (
                raw_writeback.detach().float().flatten(1).norm(dim=1).tolist()
            ),
            "attenuated_writeback_l2_per_sample": (
                attenuated_writeback.detach().float().flatten(1).norm(dim=1).tolist()
            ),
            "max_abs_delta_minus_rho_times_unit": scaling_max_abs,
            "maximum_allowed": SCALING_MAX_ABS_TOLERANCE,
            "passed": True,
        },
        "runtime_snapshot": {
            key: value.tolist() for key, value in scheduled_snapshot.items()
        },
        "schedule": schedule_contract,
        "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
        "field_schedule_sha256": field_schedule_sha256,
        "diffusion_schedule_sha256": diffusion_schedule_sha256,
        "registered_rho": registered_rho,
        "losses": loss_values,
        "loss_finite": True,
        "initial_alpha_gradient": initial_audit,
        "initial_gradients": initial_gradients,
        "test_only_activated_gradients_by_timestep": (
            per_timestep_activated_gradients
        ),
        "test_only_gate": 0.1,
        "test_only_probe_saved": False,
        "initialization": "random",
        "initial_model_state_sha256": initial_model_sha256,
        "expected_initial_model_state_sha256": EXPECTED_INITIAL_MODEL_SHA256,
        "total_parameter_count": TOTAL_PARAMETER_COUNT,
        "sparse_relation_parameter_count": SPARSE_RELATION_PARAMETER_COUNT,
        "sparse_point_mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "sparse_point_manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
        "sparse_point_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": total_memory,
        "memory_headroom_bytes": total_memory - peak_reserved,
        "cuda_timing_synchronized": True,
        "contention_before": contention_before,
        "contention_after": contention_after,
        "checkpoint_selected": False,
        "formal_training_started": False,
        "consistency_started": False,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{FAILURE_CLASSIFICATION}: {error}", file=sys.stderr, flush=True)
        raise
