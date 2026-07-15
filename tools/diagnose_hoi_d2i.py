#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-I0 frozen gradient-routing audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
from datasets.utils import get_smpl_parents


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import (  # noqa: E402
    BASE_COMPONENTS,
    BLOCK_SIZE,
    CHECKPOINTS,
    EXPECTED_PRIMARY_SHA256,
    EXPECTED_TERMINAL_SHA256,
    LOSS_COMPONENTS,
    PARAMETER_GROUPS,
    TIMESTEPS,
    gradient_geometry,
    mechanism_gate,
    parameter_group_indices,
    select_fresh_holdouts,
    stable_seed,
    state_dict_sha256,
)
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from tools.diagnose_hoi_d2h import exclusive_json, stack_diagnostic_batch  # noqa: E402
from tools.diagnose_hoi_remediation import seed_everything, sha256_file  # noqa: E402


RUN_ID = "p1-hoi-d2i-gradient-dominance-s42-20260715"
EXPECTED_CHECKPOINTS = {
    "R-1024": "d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23",
    "R-3072": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
}
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_NORMALIZATION_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
EXPECTED_BPS_SHA256 = "fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042"


def _noise_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def prepare_blocks(
    dataset: PriorWindowDataset,
    positions: Sequence[int],
) -> Sequence[Dict[str, torch.Tensor]]:
    if len(positions) % BLOCK_SIZE:
        raise ValueError("D2-I cohort size must be divisible by block size")
    blocks = []
    for offset in range(0, len(positions), BLOCK_SIZE):
        batch, _, _ = stack_diagnostic_batch(
            dataset, positions[offset:offset + BLOCK_SIZE], torch.device("cpu"),
        )
        blocks.append(batch)
    return blocks


def _loss_tensors(losses: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    human = losses["joint_position"] + losses["joint_rotation"]
    objects = losses["object_translation"] + losses["object_rotation"]
    result = {
        "human_reconstruction": human,
        "object_reconstruction": objects,
        "contact": losses["contact"],
        "reconstruction": losses["reconstruction"],
        "weighted_fk": 50.0 * losses["fk"],
        "weighted_object_surface": 50.0 * losses["object_surface"],
        "weighted_velocity": 0.1 * losses["velocity"],
        "terminal_goal": losses["object_goal"],
    }
    result["auxiliary_sum"] = sum(result[name] for name in (
        "weighted_fk", "weighted_object_surface", "weighted_velocity", "terminal_goal",
    ))
    result["total"] = losses["total"]
    return result


def gradient_block(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    batch_cpu: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    groups: Mapping[str, Sequence[int]],
    *,
    cohort: str,
    timestep: int,
    block_index: int,
    device: torch.device,
) -> Dict[str, object]:
    batch = _to_device(batch_cpu, device)
    count = batch["x"].shape[0]
    times = torch.full((count,), timestep, device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(f"D2I:{cohort}:{timestep}:{block_index}"))
    noise = torch.randn(batch["x"].shape, device=device, generator=generator)
    noisy = diffusion.q_sample(batch["x"], times, noise)
    prediction = model(
        noisy, times, batch["text_embedding"], batch["object_bps"], batch["goals"],
        normalize_progress(batch["progress"]),
    )
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    losses = hoi_training_losses(
        prediction, batch["x"], batch["goals"], batch["rest_human_offsets"],
        parents, dataset.codec.position_minimum.to(device), dataset.codec.position_maximum.to(device),
        dataset.codec.object_minimum.to(device), dataset.codec.object_maximum.to(device),
        batch["terminal_window"], batch["rest_object_points"],
        batch["world_to_local_rotation"], batch["object_rotation_reference"],
    )
    components = _loss_tensors(losses)
    base_gradients = {}
    for name in BASE_COMPONENTS:
        base_gradients[name] = torch.autograd.grad(
            components[name], parameters, retain_graph=True, allow_unused=True,
        )
    direct_total = torch.autograd.grad(
        components["total"], parameters, retain_graph=False, allow_unused=True,
    )
    geometry = gradient_geometry(base_gradients, direct_total, groups)
    loss_values = {name: float(components[name].detach()) for name in LOSS_COMPONENTS}
    finite_losses = all(math.isfinite(value) for value in loss_values.values())
    result = {
        "cohort": cohort,
        "timestep": timestep,
        "block_index": block_index,
        "windows": count,
        "q_noise_sha256": _noise_sha256(noise),
        "loss_values": loss_values,
        **geometry,
    }
    result["finite"] = bool(result["finite"] and finite_losses)
    del batch, noisy, prediction, losses, components, base_gradients, direct_total
    return result


def audit_checkpoint(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    prepared: Mapping[str, Sequence[Mapping[str, torch.Tensor]]],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    parameters, groups = parameter_group_indices(model)
    before = state_dict_sha256(model)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("D2-I model unexpectedly has populated .grad buffers")
    cohorts = {}
    finite = True
    maximum_formula_error = 0.0
    for cohort in ("primary", "terminal"):
        timestep_records = {}
        for timestep in TIMESTEPS:
            blocks = [
                gradient_block(
                    model, diffusion, dataset, batch, parameters, groups,
                    cohort=cohort, timestep=timestep, block_index=index, device=device,
                )
                for index, batch in enumerate(prepared[cohort])
            ]
            finite = finite and all(block["finite"] for block in blocks)
            maximum_formula_error = max(
                maximum_formula_error,
                max(block["total_gradient_formula_relative_l2"] for block in blocks),
            )
            timestep_records[str(timestep)] = {
                "timestep": timestep,
                "blocks": blocks,
            }
        cohorts[cohort] = {"timesteps": timestep_records}
    after = state_dict_sha256(model)
    gradients_clear = all(parameter.grad is None for parameter in parameters)
    return {
        "cohorts": cohorts,
        "finite": bool(finite),
        "maximum_total_gradient_formula_relative_l2": maximum_formula_error,
        "state_dict_sha256_before": before,
        "state_dict_sha256_after": after,
        "state_dict_unchanged": before == after,
        "parameter_grad_buffers_clear": gradients_clear,
        "optimizer_created": False,
        "training_updates": 0,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-I0",
        "mode": "frozen-weighted-objective-gradient-routing-audit",
        "repo_root": str(REPO),
        "seed": 42,
        "device": args.device,
        "block_size": args.block_size,
        "timesteps": list(TIMESTEPS),
        "gate_timesteps": [250, 499],
        "primary_windows": 128,
        "primary_selection_sha256": EXPECTED_PRIMARY_SHA256,
        "terminal_windows": 64,
        "terminal_selection_sha256": EXPECTED_TERMINAL_SHA256,
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
        "checkpoints": {
            "R-1024": {"path": str(Path(args.checkpoint_r1024).resolve()), "sha256": args.sha256_r1024},
            "R-3072": {"path": str(Path(args.checkpoint_r3072).resolve()), "sha256": args.sha256_r3072},
        },
        "loss_weights": {"fk": 50.0, "object_surface": 50.0, "velocity": 0.1, "terminal_goal": 1.0},
        "loss_components": list(LOSS_COMPONENTS),
        "parameter_groups": list(PARAMETER_GROUPS),
        "gate": {
            "formula_relative_l2_max": 1e-5,
            "ratio_geometric_mean_min": 20.0,
            "ratio_bootstrap_lower_min": 10.0,
            "cosine_bootstrap_upper_max": 0.25,
        },
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "bps_sha256": EXPECTED_BPS_SHA256,
        "model_eval": True,
        "autograd_api": "torch.autograd.grad",
        "optimizer_created": False,
        "training_updates": 0,
        "production_change": False,
        "released_checkpoint_used": False,
        "ema_used": False,
        "official_test_used": False,
        "chois_used": False,
        "output": str(Path(args.output).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-r1024", required=True)
    parser.add_argument("--checkpoint-r3072", required=True)
    parser.add_argument("--sha256-r1024", default=EXPECTED_CHECKPOINTS["R-1024"])
    parser.add_argument("--sha256-r3072", default=EXPECTED_CHECKPOINTS["R-3072"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-I0 run id must be {RUN_ID}")
    if args.block_size != BLOCK_SIZE:
        raise ValueError(f"D2-I0 block size must be {BLOCK_SIZE}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match archived D2-I0 resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-I0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("D2-I0 refuses a dirty worker checkout")
    if sha256_file(REPO / "data/train/norm.npy") != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("D2-I0 normalization hash mismatch")
    if sha256_file(REPO / "code/bps.pt") != EXPECTED_BPS_SHA256:
        raise ValueError("D2-I0 BPS hash mismatch")
    checkpoint_paths = {
        "R-1024": Path(args.checkpoint_r1024).resolve(),
        "R-3072": Path(args.checkpoint_r3072).resolve(),
    }
    requested_hashes = {"R-1024": args.sha256_r1024, "R-3072": args.sha256_r3072}
    for name in CHECKPOINTS:
        actual = sha256_file(checkpoint_paths[name])
        if requested_hashes[name] != EXPECTED_CHECKPOINTS[name] or actual != requested_hashes[name]:
            raise ValueError(f"{name} checkpoint hash mismatch: {actual}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-I0 is a four-GPU-worker CUDA diagnostic")
    seed_everything(42)
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    selections = select_fresh_holdouts(dataset)
    if selections["primary"]["sha256"] != EXPECTED_PRIMARY_SHA256:
        raise ValueError("D2-I0 primary selection hash mismatch")
    if selections["terminal"]["sha256"] != EXPECTED_TERMINAL_SHA256:
        raise ValueError("D2-I0 terminal selection hash mismatch")
    if selections["primary"]["terminal_windows"] != 0:
        raise ValueError("D2-I0 primary cohort must be nonterminal")
    prepared = {
        cohort: prepare_blocks(dataset, selections[cohort]["positions"])
        for cohort in ("primary", "terminal")
    }
    diffusion = GaussianDiffusion(500).to(device)
    output: Dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": "1B-D2-I0",
        "seed": 42,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "selection": {
            cohort: {
                "partition": "internal_validation",
                "windows": len(selections[cohort]["positions"]),
                "blocks": len(prepared[cohort]),
                "block_size": BLOCK_SIZE,
                "window_indices_sha256": selections[cohort]["sha256"],
                "terminal_windows": selections[cohort]["terminal_windows"],
            }
            for cohort in ("primary", "terminal")
        },
        "selection_disjoint_from_d2h0": True,
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
            "bps_sha256": EXPECTED_BPS_SHA256,
        },
        "candidates": {},
        "checkpoint_count_loaded": 0,
        "training_updates": 0,
        "optimizer_created": False,
        "production_model_change": False,
        "representation_change": False,
        "loss_or_weight_change": False,
        "condition_change": False,
        "sampler_change": False,
        "released_checkpoint_used": False,
        "ema_used": False,
        "official_test_used": False,
        "chois_used": False,
    }
    for name in CHECKPOINTS:
        model, metadata = load_trained_hoi_prior(
            str(checkpoint_paths[name]), device, weight_variant="online",
        )
        if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError(f"{name} data contract mismatch")
        output["candidates"][name] = {
            "checkpoint": metadata,
            **audit_checkpoint(model, diffusion, dataset, prepared, device),
        }
        output["checkpoint_count_loaded"] = len(output["candidates"])
        del model
        torch.cuda.empty_cache()
    output["paired_noise_identity"] = bool(
        all(
            output["candidates"]["R-1024"]["cohorts"][cohort]["timesteps"][str(timestep)]["blocks"][block]["q_noise_sha256"]
            == output["candidates"]["R-3072"]["cohorts"][cohort]["timesteps"][str(timestep)]["blocks"][block]["q_noise_sha256"]
            for cohort in ("primary", "terminal")
            for timestep in TIMESTEPS
            for block in range(len(prepared[cohort]))
        )
    )
    output["decision"] = mechanism_gate(output["candidates"])
    output["runtime_seconds"] = time.time() - started
    output["gpu"] = {
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
