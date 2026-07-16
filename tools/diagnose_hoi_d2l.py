#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-L0 fixed auxiliary-balance audit."""

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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.adamw_routing import (  # noqa: E402
    DIRECTIONS,
    EXPECTED_OPTIMIZER,
    mapped_optimizer_states,
    mapped_state_sha256,
    optimizer_state_sha256,
    validate_optimizer_contract,
)
from priors.auxiliary_balancing import (  # noqa: E402
    BALANCED_WEIGHTS,
    CANDIDATES,
    CURRENT_WEIGHTS,
    DERIVATION_RAW_FK_NORM,
    DERIVATION_RAW_OBJECT_SURFACE_NORM,
    DERIVATION_TARGET_NORM,
    EXPECTED_PRIMARY_SHA256,
    RAW_COMPONENTS,
    WEIGHT_SOURCE_METRICS_SHA256,
    WEIGHT_SOURCE_RUN,
    WEIGHTS,
    mechanism_gate,
    paired_routing_geometry,
    select_fresh_primary,
    verify_weight_source,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_clipping import (  # noqa: E402
    BLOCK_SIZE,
    FIELD_COMPONENTS,
    GATE_TIMESTEPS,
    GRADIENT_CLIP_NORM,
    LOSS_COMPONENTS,
    PRIMARY_WINDOWS,
    TIMESTEPS,
)
from priors.gradient_routing import (  # noqa: E402
    CHECKPOINTS,
    PARAMETER_GROUPS,
    parameter_group_indices,
    stable_seed,
    state_dict_sha256,
)
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from tools.diagnose_hoi_d2h import exclusive_json, stack_diagnostic_batch  # noqa: E402
from tools.diagnose_hoi_remediation import seed_everything, sha256_file  # noqa: E402


RUN_ID = "p1-hoi-d2l-aux-balance-s42-20260716"
EXPECTED_CHECKPOINTS = {
    "R-1024": "d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23",
    "R-3072": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
}
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_NORMALIZATION_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
EXPECTED_BPS_SHA256 = "fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042"


def _noise_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def prepare_blocks(dataset: PriorWindowDataset, positions: Sequence[int]):
    if len(positions) != PRIMARY_WINDOWS or len(positions) % BLOCK_SIZE:
        raise ValueError("D2-L requires exactly 128 windows in fixed 16-window blocks")
    return [
        stack_diagnostic_batch(
            dataset, positions[offset:offset + BLOCK_SIZE], torch.device("cpu"),
        )[0]
        for offset in range(0, len(positions), BLOCK_SIZE)
    ]


def _raw_loss_tensors(losses: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "joint_position": losses["joint_position"],
        "joint_rotation": losses["joint_rotation"],
        "object_translation": losses["object_translation"],
        "object_rotation": losses["object_rotation"],
        "contact": losses["contact"],
        "fk": losses["fk"],
        "object_surface": losses["object_surface"],
        "velocity": losses["velocity"],
        "terminal_goal": losses["object_goal"],
    }


def _candidate_loss_tensors(
    raw: Mapping[str, torch.Tensor], weights: Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    result = {name: raw[name] for name in FIELD_COMPONENTS}
    result["human_reconstruction"] = (
        result["joint_position"] + result["joint_rotation"]
    )
    result["object_reconstruction"] = (
        result["object_translation"] + result["object_rotation"]
    )
    result["reconstruction"] = (
        result["human_reconstruction"] + result["object_reconstruction"] + result["contact"]
    )
    result["weighted_fk"] = float(weights["fk"]) * raw["fk"]
    result["weighted_object_surface"] = (
        float(weights["object_surface"]) * raw["object_surface"]
    )
    result["weighted_velocity"] = float(weights["velocity"]) * raw["velocity"]
    result["terminal_goal"] = float(weights["terminal_goal"]) * raw["terminal_goal"]
    result["auxiliary_sum"] = sum(result[name] for name in (
        "weighted_fk", "weighted_object_surface", "weighted_velocity", "terminal_goal",
    ))
    result["total"] = result["reconstruction"] + result["auxiliary_sum"]
    return result


def gradient_block(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    batch_cpu: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    groups: Mapping[str, Sequence[int]],
    optimizer_states: Sequence[Mapping[str, object]],
    optimizer_group: Mapping[str, object],
    *,
    timestep: int,
    block_index: int,
    device: torch.device,
) -> Dict[str, object]:
    batch = _to_device(batch_cpu, device)
    count = batch["x"].shape[0]
    times = torch.full((count,), timestep, device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(f"D2L:primary:{timestep}:{block_index}"))
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
        prediction, batch["x"], batch["goals"], batch["rest_human_offsets"], parents,
        dataset.codec.position_minimum.to(device), dataset.codec.position_maximum.to(device),
        dataset.codec.object_minimum.to(device), dataset.codec.object_maximum.to(device),
        batch["terminal_window"], batch["rest_object_points"],
        batch["world_to_local_rotation"], batch["object_rotation_reference"],
    )
    raw_tensors = _raw_loss_tensors(losses)
    candidate_tensors = {
        name: _candidate_loss_tensors(raw_tensors, WEIGHTS[name]) for name in CANDIDATES
    }
    raw_gradients = {
        name: torch.autograd.grad(
            raw_tensors[name], parameters, retain_graph=True, allow_unused=True,
        )
        for name in RAW_COMPONENTS
    }
    direct_totals = {
        "current": torch.autograd.grad(
            losses["total"], parameters, retain_graph=True, allow_unused=True,
        ),
        "balanced": torch.autograd.grad(
            candidate_tensors["balanced"]["total"],
            parameters,
            retain_graph=False,
            allow_unused=True,
        ),
    }
    geometry = paired_routing_geometry(
        raw_gradients, direct_totals, parameters, optimizer_states, optimizer_group, groups,
    )
    raw_values = {name: float(raw_tensors[name].detach()) for name in RAW_COMPONENTS}
    candidate_values = {
        candidate: {
            name: float(candidate_tensors[candidate][name].detach())
            for name in LOSS_COMPONENTS
        }
        for candidate in CANDIDATES
    }
    production_replay_abs = abs(
        candidate_values["current"]["total"] - float(losses["total"].detach())
    )
    result = {
        "timestep": timestep,
        "block_index": block_index,
        "windows": count,
        "q_noise_sha256": _noise_sha256(noise),
        "raw_loss_values": raw_values,
        "candidate_loss_values": candidate_values,
        "production_total_value_replay_abs": production_replay_abs,
        **geometry,
    }
    result["finite"] = bool(
        result["finite"]
        and math.isfinite(production_replay_abs)
        and all(math.isfinite(value) for value in raw_values.values())
        and all(
            math.isfinite(value)
            for candidate in candidate_values.values()
            for value in candidate.values()
        )
    )
    del (
        batch, noisy, prediction, losses, raw_tensors, candidate_tensors,
        raw_gradients, direct_totals,
    )
    return result


def audit_checkpoint(
    name, model, raw_optimizer, diffusion, dataset, prepared, device, provenance,
):
    model.eval()
    parameters, groups = parameter_group_indices(model)
    contract = validate_optimizer_contract(name, raw_optimizer, parameters)
    model_before = state_dict_sha256(model)
    optimizer_before = optimizer_state_sha256(raw_optimizer)
    states = mapped_optimizer_states(raw_optimizer, parameters)
    mapped_before = mapped_state_sha256(states)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("D2-L model unexpectedly has populated .grad buffers")
    timesteps = {}
    finite = True
    maxima = {
        candidate: {
            "gradient_formula": 0.0, "clip_formula": 0.0, "adamw_formula": 0.0,
        }
        for candidate in CANDIDATES
    }
    production_replay = 0.0
    for timestep in TIMESTEPS:
        blocks = [
            gradient_block(
                model, diffusion, dataset, batch, parameters, groups, states,
                raw_optimizer["param_groups"][0], timestep=timestep,
                block_index=index, device=device,
            )
            for index, batch in enumerate(prepared)
        ]
        finite = finite and all(block["finite"] for block in blocks)
        production_replay = max(
            production_replay,
            max(block["production_total_value_replay_abs"] for block in blocks),
        )
        for candidate in CANDIDATES:
            maxima[candidate]["gradient_formula"] = max(
                maxima[candidate]["gradient_formula"],
                max(
                    block["candidates"][candidate][
                        "total_gradient_formula_relative_l2"
                    ]
                    for block in blocks
                ),
            )
            maxima[candidate]["clip_formula"] = max(
                maxima[candidate]["clip_formula"],
                max(
                    block["candidates"][candidate]["clipping"]["formula_replay_max_abs"]
                    for block in blocks
                ),
            )
            maxima[candidate]["adamw_formula"] = max(
                maxima[candidate]["adamw_formula"],
                max(
                    block["candidates"][candidate][
                        "adamw_decomposition_relative_l2"
                    ]
                    for block in blocks
                ),
            )
        timesteps[str(timestep)] = {"timestep": timestep, "blocks": blocks}
    return {
        "timesteps": timesteps,
        "finite": bool(finite),
        "optimizer_contract": contract,
        "optimizer_contract_exact": True,
        "weight_provenance": provenance,
        "weight_provenance_exact": True,
        "maximum_production_total_value_replay_abs": production_replay,
        "maximum_formula_replay": maxima,
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": state_dict_sha256(model),
        "optimizer_state_sha256_before": optimizer_before,
        "optimizer_state_sha256_after": optimizer_state_sha256(raw_optimizer),
        "mapped_state_sha256_before": mapped_before,
        "mapped_state_sha256_after": mapped_state_sha256(states),
        "parameter_grad_buffers_clear": all(parameter.grad is None for parameter in parameters),
        "optimizer_created": False,
        "training_updates": 0,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-L0",
        "mode": "fixed-gradient-balanced-auxiliary-counterfactual-routing",
        "repo_root": str(REPO),
        "seed": 42,
        "device": args.device,
        "block_size": BLOCK_SIZE,
        "primary_windows": PRIMARY_WINDOWS,
        "primary_selection_sha256": EXPECTED_PRIMARY_SHA256,
        "timesteps": list(TIMESTEPS),
        "gate_timesteps": list(GATE_TIMESTEPS),
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoints": {
            "R-1024": {
                "path": str(Path(args.checkpoint_r1024).resolve()),
                "sha256": args.sha256_r1024,
            },
            "R-3072": {
                "path": str(Path(args.checkpoint_r3072).resolve()),
                "sha256": args.sha256_r3072,
            },
        },
        "weight_source": {
            "path": str(Path(args.weight_source_metrics).resolve()),
            "run_id": WEIGHT_SOURCE_RUN,
            "sha256": WEIGHT_SOURCE_METRICS_SHA256,
            "records": 32,
            "target_norm": DERIVATION_TARGET_NORM,
            "raw_fk_norm_geomean": DERIVATION_RAW_FK_NORM,
            "raw_object_surface_norm_geomean": DERIVATION_RAW_OBJECT_SURFACE_NORM,
        },
        "weights": WEIGHTS,
        "weight_sweep": False,
        "optimizer_contract": EXPECTED_OPTIMIZER,
        "raw_components": list(RAW_COMPONENTS),
        "field_components": list(FIELD_COMPONENTS),
        "loss_components": list(LOSS_COMPONENTS),
        "candidates": list(CANDIDATES),
        "directions": list(DIRECTIONS),
        "parameter_groups": list(PARAMETER_GROUPS),
        "gate": {
            "formula_relative_l2_max": 1e-5,
            "human_delta_bootstrap_lower_min": 0.10,
            "human_bootstrap_lower_min": 0.15,
            "object_bootstrap_lower_min": 0.15,
            "required_paths": ["clipped_total", "adamw_full"],
        },
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "bps_sha256": EXPECTED_BPS_SHA256,
        "model_eval": True,
        "autograd_api": "torch.autograd.grad",
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "production_loss_change": False,
        "model_change": False,
        "representation_change": False,
        "condition_change": False,
        "sampler_change": False,
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
    parser.add_argument("--weight-source-metrics", required=True)
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
        raise ValueError(f"D2-L0 run id must be {RUN_ID}")
    if args.block_size != BLOCK_SIZE:
        raise ValueError(f"D2-L0 block size must be {BLOCK_SIZE}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match archived D2-L0 resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-L0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
    ).strip():
        raise RuntimeError("D2-L0 refuses a dirty worker checkout")
    if sha256_file(REPO / "data/train/norm.npy") != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("D2-L0 normalization hash mismatch")
    if sha256_file(REPO / "code/bps.pt") != EXPECTED_BPS_SHA256:
        raise ValueError("D2-L0 BPS hash mismatch")
    provenance = verify_weight_source(Path(args.weight_source_metrics))
    checkpoint_paths = {
        "R-1024": Path(args.checkpoint_r1024).resolve(),
        "R-3072": Path(args.checkpoint_r3072).resolve(),
    }
    requested_hashes = {
        "R-1024": args.sha256_r1024, "R-3072": args.sha256_r3072,
    }
    for name in CHECKPOINTS:
        actual = sha256_file(checkpoint_paths[name])
        if requested_hashes[name] != EXPECTED_CHECKPOINTS[name] or actual != requested_hashes[name]:
            raise ValueError(f"{name} checkpoint hash mismatch: {actual}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-L0 is a four-GPU-worker CUDA diagnostic")
    seed_everything(42)
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    selection = select_fresh_primary(dataset)
    if selection["sha256"] != EXPECTED_PRIMARY_SHA256:
        raise ValueError("D2-L0 primary selection hash mismatch")
    if (
        selection["terminal_windows"] != 0
        or selection["selected_ranks"] != list(range(898, 1026))
    ):
        raise ValueError("D2-L0 rank/nonterminal selection contract mismatch")
    prepared = prepare_blocks(dataset, selection["positions"])
    diffusion = GaussianDiffusion(500).to(device)
    output: Dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": "1B-D2-L0",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "windows": len(selection["positions"]),
            "blocks": len(prepared),
            "block_size": BLOCK_SIZE,
            "window_indices_sha256": selection["sha256"],
            "terminal_windows": selection["terminal_windows"],
            "first_rank": selection["selected_ranks"][0],
            "last_rank": selection["selected_ranks"][-1],
        },
        "selection_disjoint_from_d2h0_d2i0_d2j0_d2k0": True,
        "weight_provenance": provenance,
        "weights": WEIGHTS,
        "weight_sweep": False,
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
            "bps_sha256": EXPECTED_BPS_SHA256,
            "weight_source_metrics_sha256": WEIGHT_SOURCE_METRICS_SHA256,
        },
        "checkpoints": {},
        "training_updates": 0,
        "optimizer_created": False,
        "checkpoint_write": False,
        "production_loss_change": False,
        "model_change": False,
        "representation_change": False,
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
        raw = torch.load(checkpoint_paths[name], map_location="cpu")
        optimizer_state = raw["optimizer"]
        parameter_names = tuple(name_ for name_, _ in model.named_parameters())
        named_parameters = dict(model.named_parameters())
        checkpoint_parameter_names = tuple(
            name_ for name_ in raw["model"] if name_ in named_parameters
        )
        if checkpoint_parameter_names != parameter_names:
            raise ValueError(f"{name} model/optimizer parameter-name order mismatch")
        output["checkpoints"][name] = {
            "checkpoint": metadata,
            **audit_checkpoint(
                name, model, optimizer_state, diffusion, dataset, prepared, device,
                provenance,
            ),
        }
        del raw, optimizer_state, model
        torch.cuda.empty_cache()
    output["paired_noise_identity"] = bool(all(
        output["checkpoints"]["R-1024"]["timesteps"][str(timestep)]["blocks"][block][
            "q_noise_sha256"
        ]
        == output["checkpoints"]["R-3072"]["timesteps"][str(timestep)]["blocks"][block][
            "q_noise_sha256"
        ]
        for timestep in TIMESTEPS for block in range(len(prepared))
    ))
    output["decision"] = mechanism_gate(output["checkpoints"])
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
