#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-H0 paired exposure diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import (  # noqa: E402
    GaussianDiffusion,
    _extract,
    normalize_progress,
    prepare_clean_x0,
)
from priors.exposure import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINTS,
    CONDITION_VARIANTS,
    TARGET_TIMESTEPS,
    deterministic_condition_variants,
    fieldwise_mse_per_sample,
    mechanism_gate,
    paired_bootstrap_model_minus_oracle,
)
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import _time_embedding, load_trained_hoi_prior  # noqa: E402
from priors.remediation import select_teacher_windows, selection_sha256  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import BPS_SHA256, WindowFrame, rotation_geodesic  # noqa: E402
from datasets.utils import get_smpl_parents  # noqa: E402
from tools.diagnose_hoi_remediation import stable_seed  # noqa: E402
from tools.evaluate_hoi_remediation import global_goals, stack_frames  # noqa: E402


RUN_ID = "p1-hoi-d2h-exposure-paired-s42-20260715"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_NORMALIZATION_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
EXPECTED_SELECTION_SHA256 = "9d3f8cc4647018fdf285481ffef95df6eb3c4e6f6ad0b680f85e23b1edeebd71"
EXPECTED_CHECKPOINTS = {
    "R-1024": "d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23",
    "R-3072": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
}
AUTHOR_BASELINE_COMMIT = "b9a158f75ab0740c91c9cfc8863a65fa381b014c"
PHYSICAL_METRICS = (
    "object_goal_error_cm",
    "pelvis_goal_error_cm",
    "mpjpe_cm",
    "object_translation_mae_cm",
    "pelvis_translation_mae_cm",
    "object_rotation_geodesic_deg",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _tensor_scale(value: torch.Tensor) -> Dict[str, float]:
    detached = value.detach().float()
    return {
        "rms": float(detached.square().mean().sqrt()),
        "mean_abs": float(detached.abs().mean()),
        "max_abs": float(detached.abs().max()),
    }


def _history_max(value: torch.Tensor, fixed: torch.Tensor) -> float:
    return float(
        (value[:, :REPRESENTATION.history_frames] - fixed).abs().max()
    )


def stack_diagnostic_batch(
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Sequence[Mapping[str, torch.Tensor]], WindowFrame]:
    items = [dataset[position] for position in positions]
    keys = (
        "x", "text_embedding", "object_bps", "goals", "progress",
        "rest_human_offsets", "terminal_window", "rest_object_points",
        "world_to_local_rotation", "object_rotation_reference",
    )
    batch = {key: torch.stack([item[key] for item in items]).to(device) for key in keys}
    return batch, items, stack_frames(items, device)


@torch.no_grad()
def physical_errors_per_sample(
    dataset: PriorWindowDataset,
    prediction: torch.Tensor,
    target: torch.Tensor,
    items: Sequence[Mapping[str, torch.Tensor]],
    frames: WindowFrame,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Decode only for reference metrics; reverse states remain unprojected."""
    decoded = dataset.codec.decode(prediction, frames)
    truth = dataset.codec.decode(target, frames)
    pelvis_goal, object_goal = global_goals(dataset, items, frames, device)
    active = slice(REPRESENTATION.history_frames, None)
    relative_prediction = decoded["joints"][:, active] - decoded["joints"][:, active, :1]
    relative_truth = truth["joints"][:, active] - truth["joints"][:, active, :1]
    rotation_error = rotation_geodesic(
        decoded["object_rotation"][:, active], truth["object_rotation"][:, active],
    )
    return {
        "object_goal_error_cm": torch.linalg.vector_norm(
            decoded["object_translation"][:, -1] - object_goal, dim=-1,
        ) * 100.0,
        "pelvis_goal_error_cm": torch.linalg.vector_norm(
            decoded["joints"][:, -1, 0, (0, 2)] - pelvis_goal[:, (0, 2)], dim=-1,
        ) * 100.0,
        "mpjpe_cm": torch.linalg.vector_norm(
            relative_prediction - relative_truth, dim=-1,
        ).mean(dim=(1, 2)) * 100.0,
        "object_translation_mae_cm": torch.linalg.vector_norm(
            decoded["object_translation"][:, active] - truth["object_translation"][:, active], dim=-1,
        ).mean(dim=1) * 100.0,
        "pelvis_translation_mae_cm": torch.linalg.vector_norm(
            decoded["joints"][:, active, 0] - truth["joints"][:, active, 0], dim=-1,
        ).mean(dim=1) * 100.0,
        "object_rotation_geodesic_deg": rotation_error.mean(dim=1) * (180.0 / math.pi),
    }


def _empty_accumulator() -> Dict[str, object]:
    return {
        "oracle": {field.name: [] for field in REPRESENTATION.fields},
        "model": {field.name: [] for field in REPRESENTATION.fields},
        "physical_oracle": {name: [] for name in PHYSICAL_METRICS},
        "physical_model": {name: [] for name in PHYSICAL_METRICS},
        "state_displacement": {
            "all_channels": [],
            **{field.name: [] for field in REPRESENTATION.fields},
        },
    }


def _append(target: Dict[str, List[float]], values: Mapping[str, torch.Tensor]) -> None:
    for name, value in values.items():
        target[name].extend(value.detach().cpu().double().tolist())


def _state_displacement(model_state: torch.Tensor, oracle_state: torch.Tensor) -> Dict[str, torch.Tensor]:
    difference = model_state[:, REPRESENTATION.history_frames:] - oracle_state[:, REPRESENTATION.history_frames:]
    result = {"all_channels": difference.square().flatten(1).mean(dim=1)}
    for field in REPRESENTATION.fields:
        result[field.name] = difference[..., field.slice].square().flatten(1).mean(dim=1)
    return result


def _summarize_accumulator(value: Mapping[str, object]) -> Dict[str, object]:
    fields = {
        field.name: paired_bootstrap_model_minus_oracle(
            value["oracle"][field.name], value["model"][field.name],
        )
        for field in REPRESENTATION.fields
    }
    physical = {
        name: paired_bootstrap_model_minus_oracle(
            value["physical_oracle"][name], value["physical_model"][name],
        )
        for name in PHYSICAL_METRICS
    }
    displacement = {}
    for name, values in value["state_displacement"].items():
        array = np.asarray(values, dtype=np.float64)
        displacement[name] = {
            "mean_mse": float(array.mean()),
            "rms_of_mean_mse": float(np.sqrt(array.mean())),
            "per_sample_mse": array.tolist(),
        }
    return {
        "field_comparison": fields,
        "physical_comparison": physical,
        "state_displacement_model_parent_vs_oracle_parent": displacement,
    }


@torch.no_grad()
def paired_exposure_audit(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "timesteps": {},
        "finite": True,
        "history_max_abs": 0.0,
        "posterior_formula_replay_max_abs": 0.0,
        "paired_parent_q_noise": True,
        "paired_posterior_noise": True,
        "object_so3_projection": False,
        "support_clamp": False,
        "cfg": False,
        "autograd_detached": True,
    }
    for target_timestep in TARGET_TIMESTEPS:
        parent_timestep = target_timestep + 1
        accumulators = {variant: _empty_accumulator() for variant in CONDITION_VARIANTS}
        parent_q_noise_hash = hashlib.sha256()
        posterior_noise_hash = hashlib.sha256()
        for offset in range(0, len(positions), batch_size):
            selected = positions[offset:offset + batch_size]
            batch, items, frames = stack_diagnostic_batch(dataset, selected, device)
            clean = batch["x"]
            fixed = clean[:, :REPRESENTATION.history_frames]
            count = len(selected)
            parent_times = torch.full((count,), parent_timestep, device=device, dtype=torch.long)
            target_times = torch.full((count,), target_timestep, device=device, dtype=torch.long)
            parent_generator = torch.Generator(device=device)
            parent_generator.manual_seed(stable_seed(f"D2H:parent-q:{target_timestep}:{offset}"))
            parent_noise = torch.randn(clean.shape, device=device, generator=parent_generator)
            posterior_generator = torch.Generator(device=device)
            posterior_generator.manual_seed(stable_seed(f"D2H:posterior:{target_timestep}:{offset}"))
            posterior_noise = torch.randn(clean.shape, device=device, generator=posterior_generator)
            parent_q_noise_hash.update(parent_noise.detach().cpu().numpy().tobytes())
            posterior_noise_hash.update(posterior_noise.detach().cpu().numpy().tobytes())
            parent_state = diffusion.q_sample(clean, parent_times, parent_noise)
            oracle_state = diffusion.posterior_sample(
                parent_state, clean, parent_times, posterior_noise, fixed,
            )
            manual_mean = (
                _extract(diffusion.posterior_mean_coef1, parent_times, clean.shape) * clean
                + _extract(diffusion.posterior_mean_coef2, parent_times, clean.shape) * parent_state
            )
            manual_oracle = manual_mean + (
                0.5 * _extract(diffusion.posterior_log_variance, parent_times, clean.shape)
            ).exp() * posterior_noise
            manual_oracle[:, :REPRESENTATION.history_frames] = fixed
            result["posterior_formula_replay_max_abs"] = max(
                float(result["posterior_formula_replay_max_abs"]),
                float((manual_oracle - oracle_state).abs().max()),
            )
            result["history_max_abs"] = max(
                float(result["history_max_abs"]),
                _history_max(parent_state, fixed),
                _history_max(oracle_state, fixed),
            )
            condition_variants, _ = deterministic_condition_variants(batch)
            repeats = len(CONDITION_VARIANTS)
            expanded_fixed = fixed.repeat(repeats, 1, 1)
            expanded_parent_state = parent_state.repeat(repeats, 1, 1)
            expanded_oracle_state = oracle_state.repeat(repeats, 1, 1)
            expanded_parent_noise = posterior_noise.repeat(repeats, 1, 1)
            expanded_parent_times = parent_times.repeat(repeats)
            expanded_target_times = target_times.repeat(repeats)
            texts = torch.cat([
                condition_variants[variant]["text_embedding"] for variant in CONDITION_VARIANTS
            ])
            bps = torch.cat([
                condition_variants[variant]["object_bps"] for variant in CONDITION_VARIANTS
            ])
            goals = torch.cat([
                condition_variants[variant]["goals"] for variant in CONDITION_VARIANTS
            ])
            progress = normalize_progress(torch.cat([
                condition_variants[variant]["progress"] for variant in CONDITION_VARIANTS
            ]))
            parent_prediction = model(
                expanded_parent_state, expanded_parent_times, texts, bps, goals, progress,
            )
            parent_prediction = prepare_clean_x0(
                parent_prediction, expanded_fixed, object_so3_x0=False,
            )
            model_state = diffusion.posterior_sample(
                expanded_parent_state, parent_prediction, expanded_parent_times,
                expanded_parent_noise, expanded_fixed,
            )
            manual_model = (
                _extract(
                    diffusion.posterior_mean_coef1, expanded_parent_times, expanded_parent_state.shape,
                ) * parent_prediction
                + _extract(
                    diffusion.posterior_mean_coef2, expanded_parent_times, expanded_parent_state.shape,
                ) * expanded_parent_state
                + (0.5 * _extract(
                    diffusion.posterior_log_variance,
                    expanded_parent_times,
                    expanded_parent_state.shape,
                )).exp() * expanded_parent_noise
            )
            manual_model[:, :REPRESENTATION.history_frames] = expanded_fixed
            result["posterior_formula_replay_max_abs"] = max(
                float(result["posterior_formula_replay_max_abs"]),
                float((manual_model - model_state).abs().max()),
            )
            oracle_prediction = model(
                expanded_oracle_state, expanded_target_times, texts, bps, goals, progress,
            )
            model_prediction = model(
                model_state, expanded_target_times, texts, bps, goals, progress,
            )
            oracle_prediction = prepare_clean_x0(
                oracle_prediction, expanded_fixed, object_so3_x0=False,
            )
            model_prediction = prepare_clean_x0(
                model_prediction, expanded_fixed, object_so3_x0=False,
            )
            result["history_max_abs"] = max(
                float(result["history_max_abs"]),
                _history_max(parent_prediction, expanded_fixed),
                _history_max(model_state, expanded_fixed),
                _history_max(oracle_prediction, expanded_fixed),
                _history_max(model_prediction, expanded_fixed),
            )
            finite_tensors = (
                clean, parent_noise, posterior_noise, parent_state, oracle_state,
                parent_prediction, model_state, oracle_prediction, model_prediction,
            )
            result["finite"] = bool(
                result["finite"] and all(torch.isfinite(value).all() for value in finite_tensors)
            )
            result["autograd_detached"] = bool(
                result["autograd_detached"]
                and not any(value.requires_grad for value in finite_tensors)
            )
            for variant_index, variant in enumerate(CONDITION_VARIANTS):
                selected_rows = slice(variant_index * count, (variant_index + 1) * count)
                variant_oracle = oracle_prediction[selected_rows]
                variant_model = model_prediction[selected_rows]
                variant_model_state = model_state[selected_rows]
                result["history_max_abs"] = max(
                    float(result["history_max_abs"]),
                    _history_max(variant_oracle, fixed),
                    _history_max(variant_model, fixed),
                )
                accumulator = accumulators[variant]
                _append(accumulator["oracle"], fieldwise_mse_per_sample(variant_oracle, clean))
                _append(accumulator["model"], fieldwise_mse_per_sample(variant_model, clean))
                _append(
                    accumulator["physical_oracle"],
                    physical_errors_per_sample(
                        dataset, variant_oracle, clean, items, frames, device,
                    ),
                )
                _append(
                    accumulator["physical_model"],
                    physical_errors_per_sample(
                        dataset, variant_model, clean, items, frames, device,
                    ),
                )
                _append(
                    accumulator["state_displacement"],
                    _state_displacement(variant_model_state, oracle_state),
                )
        result["timesteps"][str(target_timestep)] = {
            "target_timestep": target_timestep,
            "parent_timestep": parent_timestep,
            "parent_q_noise_sha256": parent_q_noise_hash.hexdigest(),
            "posterior_noise_sha256": posterior_noise_hash.hexdigest(),
            **{
                variant: _summarize_accumulator(accumulators[variant])
                for variant in CONDITION_VARIANTS
            },
        }
    return result


@torch.no_grad()
def representation_scale_audit(
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    clean_sums = {field.name: 0.0 for field in REPRESENTATION.fields}
    clean_counts = {field.name: 0 for field in REPRESENTATION.fields}
    per_timestep = {}
    for target_timestep in TARGET_TIMESTEPS:
        noise_sums = {field.name: 0.0 for field in REPRESENTATION.fields}
        contribution_sums = {field.name: 0.0 for field in REPRESENTATION.fields}
        noisy_sums = {field.name: 0.0 for field in REPRESENTATION.fields}
        counts = {field.name: 0 for field in REPRESENTATION.fields}
        parent_timestep = target_timestep + 1
        for offset in range(0, len(positions), batch_size):
            selected = positions[offset:offset + batch_size]
            batch, _, _ = stack_diagnostic_batch(dataset, selected, device)
            clean = batch["x"]
            times = torch.full((len(selected),), parent_timestep, device=device, dtype=torch.long)
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_seed(f"D2H:parent-q:{target_timestep}:{offset}"))
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy = diffusion.q_sample(clean, times, noise)
            contribution = _extract(diffusion.sqrt_one_minus_alpha_bar, times, clean.shape) * noise
            for field in REPRESENTATION.fields:
                active_clean = clean[:, REPRESENTATION.history_frames:, field.slice]
                active_noise = noise[:, REPRESENTATION.history_frames:, field.slice]
                active_contribution = contribution[:, REPRESENTATION.history_frames:, field.slice]
                active_noisy = noisy[:, REPRESENTATION.history_frames:, field.slice]
                if target_timestep == TARGET_TIMESTEPS[0]:
                    clean_sums[field.name] += float(active_clean.double().square().sum())
                    clean_counts[field.name] += active_clean.numel()
                noise_sums[field.name] += float(active_noise.double().square().sum())
                contribution_sums[field.name] += float(active_contribution.double().square().sum())
                noisy_sums[field.name] += float(active_noisy.double().square().sum())
                counts[field.name] += active_noise.numel()
        per_timestep[str(target_timestep)] = {
            "target_timestep": target_timestep,
            "parent_timestep": parent_timestep,
            "unit_noise_rms": {
                name: math.sqrt(noise_sums[name] / counts[name]) for name in noise_sums
            },
            "scaled_q_noise_contribution_rms": {
                name: math.sqrt(contribution_sums[name] / counts[name]) for name in contribution_sums
            },
            "parent_noisy_state_rms": {
                name: math.sqrt(noisy_sums[name] / counts[name]) for name in noisy_sums
            },
        }
    return {
        "clean_state_rms": {
            name: math.sqrt(clean_sums[name] / clean_counts[name]) for name in clean_sums
        },
        "by_target_timestep": per_timestep,
    }


@torch.no_grad()
def token_scale_audit(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    timestep = 100
    count = batch["x"].shape[0]
    times = torch.full((count,), timestep, device=batch["x"].device, dtype=torch.long)
    generator = torch.Generator(device=batch["x"].device)
    generator.manual_seed(stable_seed("D2H:parity-token-scale"))
    noise = torch.randn(batch["x"].shape, device=batch["x"].device, generator=generator)
    noisy = diffusion.q_sample(batch["x"], times, noise)
    progress = normalize_progress(batch["progress"])
    network = model.network
    raw_goal_progress = torch.cat((batch["goals"], progress), dim=-1)
    encoded = {
        "timestep": network.time(_time_embedding(times, network.dim_model)),
        "text": network.text(batch["text_embedding"]),
        "bps": network.bps(batch["object_bps"].flatten(1)),
        "goal_progress": network.goal_progress(raw_goal_progress),
        "motion": network.motion_input(noisy),
        "learned_position": network.position.expand(count, -1, -1),
    }
    goal_full = encoded["goal_progress"]
    zeroed_response = {}
    components = {
        "pelvis_goal": slice(0, 3),
        "unused_goal_middle": slice(3, 6),
        "object_goal": slice(6, 9),
        "progress": slice(9, 12),
    }
    for name, selected in components.items():
        ablated = raw_goal_progress.clone()
        ablated[:, selected] = 0.0
        zeroed_response[name] = _tensor_scale(goal_full - network.goal_progress(ablated))
    return {
        "fixed_timestep": timestep,
        "samples": count,
        "raw_inputs": {
            "text": _tensor_scale(batch["text_embedding"]),
            "bps": _tensor_scale(batch["object_bps"]),
            "pelvis_goal": _tensor_scale(batch["goals"][:, :3]),
            "unused_goal_middle": _tensor_scale(batch["goals"][:, 3:6]),
            "object_goal": _tensor_scale(batch["goals"][:, 6:9]),
            "progress_raw": _tensor_scale(batch["progress"]),
            "progress_normalized": _tensor_scale(progress),
            "motion_noisy": _tensor_scale(noisy),
        },
        "encoded_tokens": {name: _tensor_scale(value) for name, value in encoded.items()},
        "goal_progress_token_zero_ablation_response": zeroed_response,
    }


def _gradient_statistics(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> Tuple[float, Tuple[torch.Tensor | None, ...]]:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True,
    )
    squared = sum(
        float(gradient.detach().double().square().sum())
        for gradient in gradients if gradient is not None
    )
    return math.sqrt(squared), gradients


def _gradient_cosine(
    first: Sequence[torch.Tensor | None],
    second: Sequence[torch.Tensor | None],
) -> float:
    dot = 0.0
    first_squared = 0.0
    second_squared = 0.0
    for first_value, second_value in zip(first, second):
        if first_value is not None:
            first_squared += float(first_value.detach().double().square().sum())
        if second_value is not None:
            second_squared += float(second_value.detach().double().square().sum())
        if first_value is not None and second_value is not None:
            dot += float((first_value.detach().double() * second_value.detach().double()).sum())
    denominator = math.sqrt(first_squared * second_squared)
    return dot / denominator if denominator else float("nan")


def gradient_audit(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    minimum = dataset.codec.position_minimum.to(device)
    maximum = dataset.codec.position_maximum.to(device)
    object_minimum = dataset.codec.object_minimum.to(device)
    object_maximum = dataset.codec.object_maximum.to(device)
    result = {}
    model.eval()
    for target_timestep in TARGET_TIMESTEPS:
        times = torch.full(
            (batch["x"].shape[0],), target_timestep, device=device, dtype=torch.long,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(f"D2H:gradient:{target_timestep}"))
        noise = torch.randn(batch["x"].shape, device=device, generator=generator)
        noisy = diffusion.q_sample(batch["x"], times, noise)
        prediction = model(
            noisy, times, batch["text_embedding"], batch["object_bps"], batch["goals"],
            normalize_progress(batch["progress"]),
        )
        losses = hoi_training_losses(
            prediction, batch["x"], batch["goals"], batch["rest_human_offsets"],
            parents, minimum, maximum, object_minimum, object_maximum,
            batch["terminal_window"], batch["rest_object_points"],
            batch["world_to_local_rotation"], batch["object_rotation_reference"],
        )
        reconstruction_norm, reconstruction_gradients = _gradient_statistics(
            losses["reconstruction"], parameters, retain_graph=True,
        )
        gradient_norms = {"reconstruction": reconstruction_norm}
        cosine = {}
        names = (
            "joint_position", "joint_rotation", "object_translation", "object_rotation", "contact",
            "fk", "object_surface", "velocity", "object_goal", "total",
        )
        for index, name in enumerate(names):
            norm, gradients = _gradient_statistics(
                losses[name], parameters, retain_graph=index < len(names) - 1,
            )
            gradient_norms[name] = norm
            if name in {"fk", "object_surface", "velocity", "object_goal"}:
                cosine[f"{name}_vs_field_reconstruction"] = _gradient_cosine(
                    gradients, reconstruction_gradients,
                )
        result[str(target_timestep)] = {
            "loss_values": {
                name: float(value.detach()) for name, value in losses.items()
                if name != "contact_accuracy"
            },
            "parameter_gradient_l2_norm": gradient_norms,
            "gradient_cosine": cosine,
            "finite": bool(
                all(math.isfinite(value) for value in gradient_norms.values())
                and all(math.isfinite(value) for value in cosine.values())
            ),
        }
        del prediction, losses, reconstruction_gradients
    return result


def implementation_comparison() -> Dict[str, object]:
    author_source = subprocess.check_output(
        ["git", "show", f"{AUTHOR_BASELINE_COMMIT}:code/models/infbagel.py"],
        cwd=REPO,
    )
    current_source = (REPO / "code/priors/models.py").read_bytes()
    return {
        "author_baseline_commit": AUTHOR_BASELINE_COMMIT,
        "source_sha256": {
            "author_code_models_infbagel_py": hashlib.sha256(author_source).hexdigest(),
            "current_code_priors_models_py": hashlib.sha256(current_source).hexdigest(),
        },
        "current_hoi_prior": {
            "condition_tokens": ["timestep", "text", "stored_initial_object_bps", "goal_plus_progress"],
            "motion_tokens": 16,
            "timestep_routing": "one independently encoded condition token",
            "position_encoding": "learned 20-token parameter",
            "transformer": "norm_first=True; GELU; FFN=4*dim_model; output LayerNorm",
            "input_scale": "no sqrt(dim_model) multiplier",
            "goal_routing": "pelvis/object/progress concatenated into one 12-D MLP token",
            "surface_loss": "MSE on 100 transformed object points",
            "reverse_state_training": "teacher q(x_t|x0) only for sealed checkpoints",
        },
        "author_infbagel_non_scene_path": {
            "condition_tokens": [
                "scene", "language_plus_progress", "scene_goal", "pelvis_goal",
                "object_goal", "object_bps",
            ],
            "motion_tokens": 16,
            "timestep_routing": "same timestep embedding added to every condition token",
            "position_encoding": "sinusoidal sequence positions",
            "transformer": "norm_first=False default; GELU; FFN=dim_model; no output LayerNorm",
            "input_scale": "tokens multiplied by sqrt(dim_model)",
            "goal_routing": "separate pelvis/object goal encoders; progress fused with language",
            "surface_loss": "smooth-L1 on 100 transformed object points",
            "reverse_state_training": "consistency-distilled released baseline confirmed by user",
        },
        "interpretation": (
            "Descriptive competing explanations only; these measurements do not authorize any "
            "condition, loss, architecture, representation, or sampler intervention."
        ),
    }


def contract_checks(
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    variants, permutation = deterministic_condition_variants(batch)
    joint_min = dataset.codec.position_minimum.to(batch["x"])
    joint_max = dataset.codec.position_maximum.to(batch["x"])
    object_min = dataset.codec.object_minimum.to(batch["x"])
    object_max = dataset.codec.object_maximum.to(batch["x"])
    joint_value = batch["x"][..., :84].reshape(*batch["x"].shape[:2], 28, 3)
    joint_inverse = (joint_value + 1.0) * (joint_max - joint_min) / 2.0 + joint_min
    joint_replay = -1.0 + 2.0 * (joint_inverse - joint_min) / (joint_max - joint_min)
    object_value = batch["x"][..., 216:219]
    object_inverse = (object_value + 1.0) * (object_max - object_min) / 2.0 + object_min
    object_replay = -1.0 + 2.0 * (object_inverse - object_min) / (object_max - object_min)
    nonterminal = int((~batch["terminal_window"]).sum())
    terminal = int(batch["terminal_window"].sum())
    production_source = inspect.getsource(GaussianDiffusion.sample)
    diagnostic_source = inspect.getsource(paired_exposure_audit)
    checks = {
        "broadcasting_shapes": all(
            value["text_embedding"].shape[0] == len(batch["x"])
            and value["object_bps"].shape[0] == len(batch["x"])
            and value["goals"].shape == batch["goals"].shape
            for value in variants.values()
        ),
        "batch_indexing": bool(torch.equal(permutation, torch.roll(
            torch.arange(len(batch["x"]), device=batch["x"].device), shifts=1,
        ))),
        "history_mask": REPRESENTATION.history_frames == 2,
        "terminal_mask": terminal + nonterminal == len(batch["x"]),
        "normalization_inversion": bool(
            (joint_replay - joint_value).abs().max() <= 1e-5
            and (object_replay - object_value).abs().max() <= 1e-5
        ),
        "posterior_coefficients_shared_with_production": (
            "self.posterior_sample(" in production_source
            and "diffusion.posterior_sample(" in diagnostic_source
        ),
        "main_diagnostic_detached": (
            "@torch.no_grad()" in diagnostic_source
            and "parent_prediction = model(" in diagnostic_source
        ),
        "condition_permutation_independence": bool(
            torch.equal(variants["text_permuted"]["object_bps"], batch["object_bps"])
            and torch.equal(variants["bps_permuted"]["text_embedding"], batch["text_embedding"])
            and torch.equal(variants["pelvis_permuted"]["goals"][:, 6:], batch["goals"][:, 6:])
            and torch.equal(variants["object_goal_permuted"]["goals"][:, :6], batch["goals"][:, :6])
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "selection_prefix_global_indices": [int(dataset.indices[position]) for position in positions],
        "terminal_windows": terminal,
        "nonterminal_windows": nonterminal,
        "normalization_inversion_max_abs": {
            "joint_positions": float((joint_replay - joint_value).abs().max()),
            "object_translation": float((object_replay - object_value).abs().max()),
        },
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-H0",
        "mode": "paired-reverse-state-exposure-diagnostic-only",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "partition": "internal_validation",
        "windows": 512,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "target_timesteps": list(TARGET_TIMESTEPS),
        "parent_timestep_rule": "s=t+1",
        "condition_variants": list(CONDITION_VARIANTS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "batch_size": args.batch_size,
        "parity_batch_size": args.parity_batch_size,
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "bps_sha256": BPS_SHA256,
        "author_baseline_commit": AUTHOR_BASELINE_COMMIT,
        "checkpoints": {
            "R-1024": {
                "path": str(Path(args.checkpoint_r1024).resolve()),
                "sha256": args.sha256_r1024,
                "weights": "online",
            },
            "R-3072": {
                "path": str(Path(args.checkpoint_r3072).resolve()),
                "sha256": args.sha256_r3072,
                "weights": "online",
            },
        },
        "paired_parent_q_noise": True,
        "paired_posterior_noise": True,
        "oracle_gt_use": "diagnostic posterior and reference metrics only",
        "object_so3_projection": False,
        "support_clamp": False,
        "cfg": False,
        "condition_change_between_paths": False,
        "production_sampler_equation_change": False,
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
        "checkpoint_selection": False,
        "training_updates": 0,
        "official_test_used": False,
        "chois_used": False,
        "d2h1_started": False,
        "device": args.device,
        "output": str(Path(args.output).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-r1024", required=True)
    parser.add_argument("--sha256-r1024", default=EXPECTED_CHECKPOINTS["R-1024"])
    parser.add_argument("--checkpoint-r3072", required=True)
    parser.add_argument("--sha256-r3072", default=EXPECTED_CHECKPOINTS["R-3072"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--parity-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-H0 run id must be {RUN_ID}")
    if args.batch_size < 2 or args.parity_batch_size < 2:
        raise ValueError("D2-H0 batch sizes must be at least two")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-H0 resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-H0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("D2-H0 refuses a dirty worker checkout")
    if sha256_file(REPO / "data/train/norm.npy") != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("D2-H0 normalization hash mismatch")
    if sha256_file(REPO / "code/bps.pt") != BPS_SHA256:
        raise ValueError("D2-H0 BPS hash mismatch")
    checkpoint_paths = {
        "R-1024": Path(args.checkpoint_r1024).resolve(),
        "R-3072": Path(args.checkpoint_r3072).resolve(),
    }
    requested_hashes = {
        "R-1024": args.sha256_r1024,
        "R-3072": args.sha256_r3072,
    }
    for name in CHECKPOINTS:
        actual = sha256_file(checkpoint_paths[name])
        if requested_hashes[name] != EXPECTED_CHECKPOINTS[name] or actual != requested_hashes[name]:
            raise ValueError(f"{name} checkpoint hash mismatch: {actual}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-H0 is a four-GPU-worker CUDA diagnostic")
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    positions = select_teacher_windows(dataset, 512)
    selection_hash = selection_sha256(int(dataset.indices[position]) for position in positions)
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(f"D2-H0 selection mismatch: {selection_hash}")
    diffusion = GaussianDiffusion(500).to(device)
    parity_positions = list(positions[:args.parity_batch_size])
    terminal_position = next(
        (position for position in positions if bool(dataset[position]["terminal_window"])), None,
    )
    if terminal_position is None:
        raise ValueError("D2-H0 parity audit requires a terminal window")
    parity_positions[-1] = terminal_position
    parity_batch, _, _ = stack_diagnostic_batch(dataset, parity_positions, device)
    output: Dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-H0",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "windows": len(positions),
            "window_indices_sha256": selection_hash,
            "official_test_sequence_count": 0,
            "chois_sequence_count": 0,
        },
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
            "bps_sha256": BPS_SHA256,
        },
        "contract_checks": contract_checks(
            dataset, parity_positions, parity_batch,
        ),
        "implementation_parity": {
            "routing_comparison": implementation_comparison(),
            "representation_scales": representation_scale_audit(
                diffusion, dataset, positions, device, args.batch_size,
            ),
            "checkpoints": {},
            "selection_use": "descriptive_only_not_a_gate_or_fallback",
        },
        "candidates": {},
        "checkpoint_count_loaded": 0,
        "checkpoint_selection": False,
        "training_updates": 0,
        "oracle_gt_use": "diagnostic posterior and reference metrics only",
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
        "official_test_used": False,
        "chois_used": False,
        "d2h1_started": False,
    }
    if not output["contract_checks"]["passed"]:
        raise RuntimeError("D2-H0 implementation contract checks failed before checkpoint execution")
    for name in CHECKPOINTS:
        model, metadata = load_trained_hoi_prior(
            str(checkpoint_paths[name]), device, weight_variant="online",
        )
        if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError(f"{name} data contract mismatch")
        output["candidates"][name] = {
            "checkpoint": metadata,
            **paired_exposure_audit(
                model, diffusion, dataset, positions, device, args.batch_size,
            ),
        }
        output["implementation_parity"]["checkpoints"][name] = {
            "token_numeric_scales": token_scale_audit(model, diffusion, parity_batch),
            "loss_parameter_gradients": gradient_audit(
                model, diffusion, dataset, parity_batch, device,
            ),
        }
        output["checkpoint_count_loaded"] = len(output["candidates"])
        del model
        torch.cuda.empty_cache()
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
