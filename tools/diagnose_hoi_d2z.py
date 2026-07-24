#!/usr/bin/env python3
"""Run the preregistered D2-X/D2-Y/D2-Z gated-gradient diagnostic."""

from __future__ import annotations

import argparse
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.d2z import D2ZPriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.losses import (  # noqa: E402
    D2X_FOOT_XZ_VELOCITY_SLOTS,
    _velocity_residuals,
    hoi_training_losses,
)
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.near_ground import sha256_file  # noqa: E402
from priors.optimizer_reset import (  # noqa: E402
    NATIVE_SELECTION_SHA256,
    select_native_holdout,
    stable_seed,
)
from priors.representation import REPRESENTATION  # noqa: E402
from tools.diagnose_hoi_d2y import (  # noqa: E402
    CHECKPOINT_WINDOWS,
    D2X_HASHES,
    EXPECTED_DATA_CONTRACT_SHA256,
    OBJECTIVE_WEIGHTS,
    TIMESTEPS,
    _checkpoint_contract as _d2xy_checkpoint_contract,
    _gradient_norm,
    _output_slice_norm,
    _predicted_fk,
    exclusive_json,
    gradient_cosine,
    tensor_sha256,
)


RUN_ID = "p1-hoi-d2z-immutable-gt-near-ground-gating-internal-r1-s42-20260724"
SUBPHASE = "1B-D2-Z0-internal-r1"
EXPECTED_PYTHON = "/home/yujinlun/data/envs/infbagel/bin/python"
D2Y_FINAL_SHA256 = "8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7"
D2Z_TRAINING_RUN_ID = "p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724"


def _checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    for expert in ("d2x", "d2y", "d2z"):
        for stratum in CHECKPOINT_WINDOWS:
            parser.add_argument(
                f"--{expert}-{stratum}-checkpoint", type=Path, required=True,
            )
            parser.add_argument(f"--{expert}-{stratum}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--gate-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    _checkpoint_arguments(parser)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return args


def _d2z_checkpoint_contract(
    path: Path,
    expected_sha256: str,
    *,
    stratum: str,
    gate_audit_sha256: str,
) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_windows = CHECKPOINT_WINDOWS[stratum]
    expected_name = f"{D2Z_TRAINING_RUN_ID}_windows{expected_windows:09d}.pth"
    if actual != expected_sha256:
        raise ValueError(f"D2-Z/{stratum} checkpoint hash mismatch")
    if path.name != expected_name:
        raise ValueError(f"D2-Z/{stratum} checkpoint basename mismatch")
    raw = torch.load(path, map_location="cpu")
    initialization = raw.get("weight_initialization", {})
    resume = raw.get("resume_contract", {})
    checks = {
        "checkpoint_type": raw.get("checkpoint_type") == "hoi_prior_phase1b",
        "run_id": raw.get("run_id") == D2Z_TRAINING_RUN_ID,
        "seed": raw.get("seed") == 42,
        "processed_windows": raw.get("processed_windows") == expected_windows,
        "optimizer_updates": raw.get("optimizer_updates") == expected_windows // 2048,
        "world_size": raw.get("world_size") == 4,
        "effective_batch_size": raw.get("effective_batch_size") == 2048,
        "data_contract": (
            raw.get("data_contract_sha256") == EXPECTED_DATA_CONTRACT_SHA256
        ),
        "random_initialization": (
            initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("restored_components") == []
        ),
        "d2z_mode": resume.get("d2z_immutable_gt_near_ground_gating") is True,
        "fk_routing": resume.get("fk_foot_temporal_routing") is True,
        "multiplier": resume.get("routed_foot_residual_multiplier") == 1024.0,
        "gate_audit": resume.get("d2z_gate_audit_sha256") == gate_audit_sha256,
        "online_model": isinstance(raw.get("model"), dict),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z/{stratum} checkpoint contract mismatch: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
        "processed_windows": expected_windows,
        "optimizer_updates": expected_windows // 2048,
        "git_commit": raw.get("git_commit"),
        "checks": checks,
    }


def _move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    keys = (
        "x",
        "text_embedding",
        "object_bps",
        "goals",
        "progress",
        "rest_human_offsets",
        "terminal_window",
        "rest_object_points",
        "world_to_local_rotation",
        "object_rotation_reference",
        "d2z_near_ground_gate",
    )
    return {key: batch[key].to(device) for key in keys}


def _masked_per_sequence(
    squared_error: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if squared_error.shape != mask.shape or squared_error.shape[:1] != (96,):
        raise ValueError("D2-Z internal masked residual expects 96 identically shaped windows")
    error = squared_error.reshape(32, 3, -1)
    selected = mask.reshape(32, 3, -1)
    counts = selected.sum(dim=(1, 2))
    numerator = (error * selected).sum(dim=(1, 2))
    mse = numerator / counts.clamp_min(1)
    mse = mse.masked_fill(counts == 0, float("nan"))
    return mse, counts


def _per_sequence_mse_json(
    mse: torch.Tensor,
    counts: torch.Tensor,
) -> list:
    if mse.shape != (32,) or counts.shape != (32,):
        raise ValueError("D2-Z internal per-sequence report expects 32 entries")
    values = []
    for value, count in zip(
        mse.detach().double().cpu().tolist(),
        counts.detach().cpu().tolist(),
    ):
        if int(count) == 0:
            values.append(None)
        elif math.isfinite(float(value)):
            values.append(float(value))
        else:
            raise ValueError("D2-Z internal defined residual is non-finite")
    return values


def evaluate_model_timestep(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    *,
    timestep: int,
    seed_label: str,
) -> Dict[str, object]:
    model.train()
    clean = batch["x"]
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(stable_seed(seed_label + ":noise"))
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    timesteps = torch.full(
        (clean.shape[0],), timestep, device=clean.device, dtype=torch.long,
    )
    noisy = diffusion.q_sample(clean, timesteps, noise)
    dropout_seed = stable_seed(seed_label + ":dropout")
    torch.manual_seed(dropout_seed)
    torch.cuda.manual_seed_all(dropout_seed)
    prediction = model(
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
    )
    fk = _predicted_fk(
        prediction, batch["rest_human_offsets"], parents, minimum, maximum,
    )
    predicted_residual, target_residual = _velocity_residuals(
        prediction,
        clean,
        fk,
        minimum,
        maximum,
        fk_foot_temporal_routing=True,
    )
    routed_error = (
        predicted_residual[..., list(D2X_FOOT_XZ_VELOCITY_SLOTS)]
        - target_residual[..., list(D2X_FOOT_XZ_VELOCITY_SLOTS)]
    ).square()
    gate4 = batch["d2z_near_ground_gate"]
    if gate4.shape != (96, 14, 4) or gate4.dtype != torch.bool:
        raise ValueError(f"D2-Z internal gate contract mismatch: {gate4.shape}/{gate4.dtype}")
    gate8 = gate4.repeat_interleave(2, dim=-1)
    if not torch.any(gate8) or not torch.any(~gate8):
        raise ValueError("D2-Z internal aggregate residual strata must both be nonempty")
    active_mse, active_counts = _masked_per_sequence(routed_error, gate8)
    inactive_mse, inactive_counts = _masked_per_sequence(routed_error, ~gate8)

    losses = hoi_training_losses(
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
        fk_weight=0.3569973401779424,
        object_surface_weight=0.4772322188400037,
        velocity_weight=0.1,
        goal_weight=1.0,
        fk_foot_temporal_routing=True,
        routed_foot_residual_multiplier=1.0,
    )
    denominator = predicted_residual.numel()
    gated_weights = torch.where(
        gate8,
        routed_error.new_tensor(1024.0),
        routed_error.new_tensor(1.0),
    )
    gated_contribution = 0.1 * (routed_error * gated_weights).sum() / denominator
    uniform_contribution = 0.1 * 1024.0 * routed_error.sum() / denominator
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    gated_gradients = torch.autograd.grad(
        gated_contribution, parameters, retain_graph=True, allow_unused=True,
    )
    uniform_gradients = torch.autograd.grad(
        uniform_contribution, parameters, retain_graph=True, allow_unused=True,
    )
    named_gated = list(zip((name for name, _ in named_parameters), gated_gradients))
    named_gated_map = dict(named_gated)
    gated_norm = _gradient_norm(named_gated)
    uniform_norm = _gradient_norm(
        list(zip((name for name, _ in named_parameters), uniform_gradients))
    )
    cosines = {}
    objective_norms = {}
    objective_items = list(OBJECTIVE_WEIGHTS.items())
    for index, (name, weight) in enumerate(objective_items):
        gradients = torch.autograd.grad(
            float(weight) * losses[name],
            parameters,
            retain_graph=index < len(objective_items) - 1,
            allow_unused=True,
        )
        cosines[name] = gradient_cosine(gated_gradients, gradients)
        objective_norms[name] = _gradient_norm(
            list(zip((item[0] for item in named_parameters), gradients))
        )
    numeric = torch.cat((
        active_mse[active_counts > 0],
        inactive_mse[inactive_counts > 0],
    )).detach().double().cpu().numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("D2-Z internal routed residual contains non-finite values")
    return {
        "timestep": timestep,
        "noise_sha256": tensor_sha256(noise),
        "dropout_seed": dropout_seed,
        "gate_occupancy": float(gate4.float().mean().item()),
        "active_routed_residual_rms": float(
            routed_error[gate8].mean().sqrt().item()
        ),
        "inactive_routed_residual_rms": float(
            routed_error[~gate8].mean().sqrt().item()
        ),
        "active_routed_residual_mse_by_sequence": _per_sequence_mse_json(
            active_mse, active_counts,
        ),
        "inactive_routed_residual_mse_by_sequence": _per_sequence_mse_json(
            inactive_mse, inactive_counts,
        ),
        "active_entries_by_sequence": active_counts.detach().cpu().tolist(),
        "inactive_entries_by_sequence": inactive_counts.detach().cpu().tolist(),
        "gated_routed_training_contribution": float(gated_contribution.detach().item()),
        "uniform_routed_training_contribution": float(uniform_contribution.detach().item()),
        "gated_gradient_norms": {
            "all_parameters": gated_norm,
            "input_projection": _gradient_norm(
                named_gated, ("network.motion_input.",)
            ),
            "output_projection": _gradient_norm(
                named_gated, ("network.output.",)
            ),
            "transformer": _gradient_norm(
                named_gated, ("network.transformer.",)
            ),
            "root_output": _output_slice_norm(named_gated_map, 0, 3),
            "rotation_output": _output_slice_norm(named_gated_map, 84, 216),
        },
        "objective_gradient_norms": objective_norms,
        "gated_gradient_cosines": cosines,
        "uniform_vs_gated": {
            "uniform_gradient_norm": uniform_norm,
            "gated_gradient_norm": gated_norm,
            "gated_retained_fraction": (
                gated_norm / uniform_norm if uniform_norm > 0.0 else None
            ),
            "gradient_cosine": gradient_cosine(uniform_gradients, gated_gradients),
        },
    }


def _load_and_evaluate(
    path: Path,
    device: torch.device,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    *,
    expert: str,
    stratum: str,
) -> Dict[str, object]:
    model, metadata = load_trained_hoi_prior(
        str(path), device, weight_variant="online",
    )
    if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
        raise ValueError(f"{expert}/{stratum} loaded model data contract mismatch")
    results = {
        str(timestep): evaluate_model_timestep(
            model,
            diffusion,
            batch,
            parents,
            minimum,
            maximum,
            object_minimum,
            object_maximum,
            timestep=timestep,
            seed_label=f"D2Z:paired:t{timestep}",
        )
        for timestep in TIMESTEPS
    }
    del model
    torch.cuda.empty_cache()
    return {"metadata": metadata, "timesteps": results}


def diagnostic_summary(results: Mapping[str, object]) -> Dict[str, object]:
    required = []
    finite = []
    for expert in ("d2x", "d2y", "d2z"):
        for stratum in CHECKPOINT_WINDOWS:
            for timestep in (str(value) for value in TIMESTEPS):
                item = results[expert][stratum]["timesteps"][timestep]
                stratum_contracts = []
                for label in ("active", "inactive"):
                    sequence_mse = item[
                        f"{label}_routed_residual_mse_by_sequence"
                    ]
                    sequence_counts = item[f"{label}_entries_by_sequence"]
                    entries = (
                        len(sequence_mse) == 32
                        and len(sequence_counts) == 32
                        and all(
                            (
                                int(count) == 0 and value is None
                            ) or (
                                int(count) > 0
                                and value is not None
                                and math.isfinite(float(value))
                            )
                            for value, count in zip(sequence_mse, sequence_counts)
                        )
                    )
                    stratum_contracts.append(
                        entries and sum(int(count) for count in sequence_counts) > 0
                    )
                required.append(all(stratum_contracts))
                values = [
                    item["active_routed_residual_rms"],
                    item["inactive_routed_residual_rms"],
                    item["gate_occupancy"],
                    item["uniform_vs_gated"]["gated_gradient_norm"],
                    item["uniform_vs_gated"]["uniform_gradient_norm"],
                ]
                finite.append(all(math.isfinite(float(value)) for value in values))
    return {
        "contract_passed": all(required) and all(finite),
        "all_required_records_present": all(required),
        "all_required_scalars_finite": all(finite),
        "selection_use": False,
        "checkpoint_selected": False,
        "consistency_authorized": False,
        "consistency_started": False,
    }


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-Z internal run id must be {RUN_ID}")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-Z internal diagnostic requires absolute INFBAGEL_PYTHON")
    if args.python.resolve() != Path(EXPECTED_PYTHON).resolve():
        raise ValueError(f"D2-Z internal diagnostic requires {EXPECTED_PYTHON}")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi" or socket.gethostname() != "node01":
        raise RuntimeError("D2-Z internal diagnostic is restricted to the HOI worker")
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        text=True,
    ).strip():
        raise RuntimeError("D2-Z internal diagnostic refuses a dirty worker checkout")
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if sha256_file(args.gate_audit.resolve()) != args.gate_audit_sha256:
        raise ValueError("D2-Z internal gate audit hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-Z internal diagnostic requires worker CUDA")

    checkpoint_contracts = {
        expert: {} for expert in ("d2x", "d2y", "d2z")
    }
    checkpoint_paths = {
        expert: {} for expert in ("d2x", "d2y", "d2z")
    }
    for expert in ("d2x", "d2y", "d2z"):
        for stratum in CHECKPOINT_WINDOWS:
            path = getattr(args, f"{expert}_{stratum}_checkpoint").resolve()
            expected_hash = getattr(args, f"{expert}_{stratum}_sha256")
            if expert == "d2x" and expected_hash != D2X_HASHES[stratum]:
                raise ValueError(f"D2-X {stratum} hash is not the sealed control")
            if expert == "d2y" and stratum == "final" and expected_hash != D2Y_FINAL_SHA256:
                raise ValueError("D2-Y final hash is not the sealed comparator")
            checkpoint_paths[expert][stratum] = path
            if expert == "d2z":
                contract = _d2z_checkpoint_contract(
                    path,
                    expected_hash,
                    stratum=stratum,
                    gate_audit_sha256=args.gate_audit_sha256,
                )
            else:
                contract = _d2xy_checkpoint_contract(
                    path, expected_hash, expert=expert, stratum=stratum,
                )
            checkpoint_contracts[expert][stratum] = contract

    dataset = D2ZPriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        gate_audit_path=str(args.gate_audit.resolve()),
        gate_audit_sha256=args.gate_audit_sha256,
    )
    selection = select_native_holdout(dataset)
    if selection["sha256"] != NATIVE_SELECTION_SHA256 or selection["sequences"] != 32:
        raise ValueError("D2-Z internal selection mismatch")
    items = [
        dataset[position]
        for triple in selection["triples"]
        for position in triple
    ]
    batch = _move_batch(default_collate(items), device)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    minimum = torch.as_tensor(dataset.minimum, device=device)
    maximum = torch.as_tensor(dataset.maximum, device=device)
    object_minimum = torch.as_tensor(dataset.object_minimum, device=device)
    object_maximum = torch.as_tensor(dataset.object_maximum, device=device)
    diffusion = GaussianDiffusion(REPRESENTATION.diffusion_steps).to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    results = {
        expert: {
            stratum: _load_and_evaluate(
                checkpoint_paths[expert][stratum],
                device,
                diffusion,
                batch,
                parents,
                minimum,
                maximum,
                object_minimum,
                object_maximum,
                expert=expert,
                stratum=stratum,
            )
            for stratum in CHECKPOINT_WINDOWS
        }
        for expert in ("d2x", "d2y", "d2z")
    }
    torch.cuda.synchronize(device)
    summary = diagnostic_summary(results)
    if not summary["contract_passed"]:
        raise ValueError("D2-Z internal diagnostic produced an incomplete/non-finite result")
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "status": "completed",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "eligible_sequence_ranks": [128, 159],
            "sequences": 32,
            "windows": 96,
            "sha256": selection["sha256"],
            "global_indices": selection["global_indices"],
            "official_test_sequences": 0,
        },
        "gate_audit": {
            "path": str(args.gate_audit.resolve()),
            "sha256": args.gate_audit_sha256,
        },
        "checkpoints": {
            expert: {
                f"{stratum}_sha256": checkpoint_contracts[expert][stratum]["sha256"]
                for stratum in CHECKPOINT_WINDOWS
            }
            for expert in ("d2x", "d2y", "d2z")
        },
        "checkpoint_contracts": checkpoint_contracts,
        "timesteps": list(TIMESTEPS),
        "pairing": {
            "same_clean_windows": True,
            "same_timestep": True,
            "same_noise": True,
            "same_condition_dropout": True,
        },
        "results": results,
        "diagnostic_summary": summary,
        "runtime_seconds": time.perf_counter() - started,
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selected": False,
        "official_test_used": False,
        "consistency_authorized": False,
        "consistency_started": False,
    }
    exclusive_json(output_path, output)


if __name__ == "__main__":
    main()
