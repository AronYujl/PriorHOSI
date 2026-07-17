"""Locked utilities for the Phase 1B D2-S0 denoiser-response frontier."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .contact_guidance import (
    AUTHOR_HAND_WEIGHT,
    FK_PALM_INDICES,
    SEMANTIC_THRESHOLD,
    SPATIAL_THRESHOLD_M,
)
from .optimizer_reset import paired_difference, paired_mean_ratio
from .remediation import selection_sha256, stable_digest
from .representation import REPRESENTATION
from .routed_guidance import UPPER_ROTATION_JOINTS, upper_rotation_mask


RUN_ID = "p1-hoi-d2s-denoiser-response-frontier-s42-20260717"
RETRY_RUN_ID = "p1-hoi-d2s-denoiser-response-frontier-r1-s42-20260717"
SUBPHASE = "1B-D2-S0"
RUN_SUBPHASES = {
    RUN_ID: SUBPHASE,
    RETRY_RUN_ID: f"{SUBPHASE}-r1",
}
RUN_IDS: Tuple[str, ...] = tuple(RUN_SUBPHASES)
CHECKPOINT_SHA256 = (
    "ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8"
)
PHASE_OFFSETS: Tuple[int, ...] = (21, 63, 105)
PRIOR_ROLLOUT_OFFSETS: Tuple[int, ...] = (
    0, 7, 14, 28, 42, 49, 56, 70, 84, 91, 98, 112,
)
SEQUENCES = 64
WINDOWS_PER_SEQUENCE = 3
SELECTION_SHA256 = (
    "77d493519b4f7e91a529e3be1b42c3e62d84d045d11bbf24acaab10c6a41a70d"
)
TARGET_TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100, 250, 498)
PARENT_TIMESTEPS: Tuple[int, ...] = tuple(value + 1 for value in TARGET_TIMESTEPS)
GATE_TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100)
DIRECTIONS: Tuple[str, ...] = ("author_all", "upper_raw")
SCALES: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)
NONZERO_SCALES: Tuple[float, ...] = SCALES[:-1]
TRUST_RATIO = 0.25
PROTECTED_GROUPS: Tuple[str, ...] = (
    "joint_positions",
    "non_upper_rotations",
    "object_translation",
    "object_rotation",
    "contact",
)
FIELD_METRICS: Tuple[str, ...] = tuple(
    f"{field.name}_mse" for field in REPRESENTATION.fields
)
PHYSICAL_METRICS: Tuple[str, ...] = (
    "fk_mpjpe_cm",
    "pelvis_goal_error_cm",
    "object_goal_error_cm",
    "object_translation_mae_cm",
    "object_rotation_geodesic",
    "fk_foot_sliding",
)
GATE_RATIO_METRICS: Tuple[str, ...] = (
    "joint_position_mse",
    "object_translation_mse",
    *PHYSICAL_METRICS,
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
HISTORY_MAX_ABS = 1e-5


def select_response_holdout(dataset) -> Dict[str, object]:
    """Return the locked fresh 64-sequence D2-S phase-offset cohort."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-S selection is internal-validation only")
    by_sequence = defaultdict(dict)
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        pi = int(dataset.language["pi"][global_index])
        by_sequence[sequence][pi] = position
    suffix = ",".join(str(value) for value in PHASE_OFFSETS)
    eligible = []
    for sequence, positions in by_sequence.items():
        if all(pi in positions for pi in PHASE_OFFSETS):
            name = str(dataset.scene_names[sequence])
            eligible.append((
                stable_digest(
                    f"42:d2s-denoiser-response:{name}:{suffix}"
                ),
                name,
                sequence,
                positions,
            ))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(eligible) < SEQUENCES:
        raise ValueError(f"D2-S requires {SEQUENCES} eligible sequences")
    rows = eligible[:SEQUENCES]
    triples = [
        tuple(row[3][pi] for pi in PHASE_OFFSETS)
        for row in rows
    ]
    global_indices = [
        int(dataset.indices[position])
        for triple in triples
        for position in triple
    ]
    result = {
        "triples": triples,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "phase_offsets": list(PHASE_OFFSETS),
        "prior_rollout_offsets": list(PRIOR_ROLLOUT_OFFSETS),
        "sequences": len(triples),
        "windows": len(global_indices),
        "eligible_sequences": len(eligible),
        "sequence_names": [row[1] for row in rows],
    }
    if result["sha256"] != SELECTION_SHA256:
        raise ValueError(f"D2-S selection mismatch: {result['sha256']}")
    return result


def fixed_mask_from_contact(contact: torch.Tensor) -> torch.Tensor:
    if contact.ndim != 3 or contact.shape[-1] != 4:
        raise ValueError("D2-S contact must be [B,T,4]")
    return (contact[..., :2] > SEMANTIC_THRESHOLD).detach()


def author_components_per_sample(
    human_joints: torch.Tensor,
    object_vertices: torch.Tensor,
    object_translation: torch.Tensor,
    object_rotation: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Replay the author objective as per-sample values under one fixed mask."""
    if human_joints.ndim != 4 or human_joints.shape[2:] != (24, 3):
        raise ValueError("D2-S author joints must be [B,T,24,3]")
    batch, frames = human_joints.shape[:2]
    if object_vertices.ndim != 4 or object_vertices.shape[:2] != (batch, frames):
        raise ValueError("D2-S object vertices differ from joints")
    if object_translation.shape != (batch, frames, 3):
        raise ValueError("D2-S object translation shape mismatch")
    if object_rotation.shape != (batch, frames, 3, 3):
        raise ValueError("D2-S object rotation shape mismatch")
    if fixed_mask.shape != (batch, frames, 2):
        raise ValueError("D2-S fixed mask must be [B,T,2]")
    mask = fixed_mask.detach().to(human_joints)
    palms = human_joints[:, :, FK_PALM_INDICES]
    distances = torch.cdist(
        palms.reshape(batch * frames, 2, 3),
        object_vertices.reshape(batch * frames, object_vertices.shape[2], 3),
    ).amin(dim=-1).reshape(batch, frames, 2)
    spatial = torch.maximum(
        distances * mask - SPATIAL_THRESHOLD_M,
        torch.zeros_like(distances),
    ).mean(dim=(1, 2))

    inverse = object_rotation.detach().transpose(2, 3)
    relative = []
    for palm in FK_PALM_INDICES:
        offset = human_joints[:, :, palm] - object_translation.detach()
        relative.append(torch.matmul(inverse, offset[..., None]).squeeze(-1))
    temporal_terms = []
    for hand, value in enumerate(relative):
        normalized = value / torch.linalg.vector_norm(
            value, dim=-1, keepdim=True,
        )
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        hand_mask = mask[:, :, hand:hand + 1]
        pair_mask = hand_mask * hand_mask.transpose(-1, -2)
        temporal_terms.append(1.0 - (similarity * pair_mask).mean(dim=(1, 2)))
    temporal = temporal_terms[0] + temporal_terms[1]
    raw = spatial + temporal
    return {
        "raw_total": raw,
        "effective_total": raw * AUTHOR_HAND_WEIGHT,
        "spatial": spatial,
        "temporal": temporal,
        "mask_coverage": mask.mean(dim=(1, 2)),
        "distance_mean_m": distances.mean(dim=(1, 2)),
    }


def direction_update(full_gradient: torch.Tensor, direction: str) -> torch.Tensor:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown D2-S direction: {direction}")
    if full_gradient.ndim != 3 or full_gradient.shape[-1] != REPRESENTATION.dimension:
        raise ValueError("D2-S direction expects [B,T,232]")
    result = full_gradient.clone()
    if direction == "upper_raw":
        result = result * upper_rotation_mask(
            device=result.device,
        ).reshape(1, 1, -1)
    result[:, :REPRESENTATION.history_frames] = 0.0
    return result


def scaled_candidate_batch(
    posterior: torch.Tensor,
    update: torch.Tensor,
    fixed_history: torch.Tensor,
    scales: Sequence[float] = SCALES,
) -> torch.Tensor:
    """Pack scale-major candidate states without touching the RNG."""
    if posterior.shape != update.shape:
        raise ValueError("D2-S posterior/update shapes differ")
    if fixed_history.shape != posterior[:, :REPRESENTATION.history_frames].shape:
        raise ValueError("D2-S fixed-history shape mismatch")
    values = []
    for scale in scales:
        candidate = posterior + float(scale) * update
        candidate[:, :REPRESENTATION.history_frames] = fixed_history
        values.append(candidate)
    return torch.cat(values, dim=0)


def unpack_scale_major(value: torch.Tensor, batch: int) -> torch.Tensor:
    if batch <= 0 or value.shape[0] != len(SCALES) * batch:
        raise ValueError("D2-S scale-major batch shape mismatch")
    return value.reshape(len(SCALES), batch, *value.shape[1:])


def _group_slices(device: torch.device) -> Dict[str, torch.Tensor]:
    masks = {}
    for name in PROTECTED_GROUPS:
        masks[name] = torch.zeros(
            REPRESENTATION.dimension, dtype=torch.bool, device=device,
        )
    masks["joint_positions"][REPRESENTATION.field("joint_positions").slice] = True
    rotation = REPRESENTATION.field("joint_rotations_6d")
    masks["non_upper_rotations"][rotation.slice] = True
    masks["non_upper_rotations"] &= ~upper_rotation_mask(device=device)
    for name in ("object_translation", "object_rotation", "contact"):
        masks[name][REPRESENTATION.field(name).slice] = True
    return masks


def response_group_rms(value: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return mutable-frame RMS per sample for each protected response group."""
    if value.ndim != 3 or value.shape[-1] != REPRESENTATION.dimension:
        raise ValueError("D2-S response must be [B,T,232]")
    mutable = value[:, REPRESENTATION.history_frames:]
    return {
        name: mutable[..., mask].square().flatten(1).mean(dim=1).sqrt()
        for name, mask in _group_slices(value.device).items()
    }


def select_largest_eligible_scale(
    baseline_loss: torch.Tensor,
    candidate_losses: torch.Tensor,
    natural_response: torch.Tensor,
    candidate_responses: torch.Tensor,
) -> Dict[str, object]:
    """Select the largest scale using only fixed-mask loss and local response."""
    if baseline_loss.ndim != 1:
        raise ValueError("D2-S baseline loss must be [B]")
    batch = baseline_loss.shape[0]
    if candidate_losses.shape != (len(SCALES), batch):
        raise ValueError("D2-S candidate losses must be [scales,B]")
    if candidate_responses.shape[:2] != (len(SCALES), batch):
        raise ValueError("D2-S candidate responses must be [scales,B,T,D]")
    natural = response_group_rms(natural_response)
    candidate = {
        name: torch.stack([
            response_group_rms(candidate_responses[index])[name]
            for index in range(len(SCALES))
        ])
        for name in PROTECTED_GROUPS
    }
    improved = candidate_losses < baseline_loss[None]
    group_passes = {
        name: values <= TRUST_RATIO * natural[name][None]
        for name, values in candidate.items()
    }
    eligible = improved.clone()
    for value in group_passes.values():
        eligible &= value
    eligible[-1] = True
    selected_index = torch.full(
        (batch,), len(SCALES) - 1, dtype=torch.long,
        device=baseline_loss.device,
    )
    unresolved = torch.ones(batch, dtype=torch.bool, device=baseline_loss.device)
    for scale_index in range(len(NONZERO_SCALES)):
        choose = unresolved & eligible[scale_index]
        selected_index[choose] = scale_index
        unresolved &= ~choose
    selected_scale = torch.tensor(
        SCALES, dtype=baseline_loss.dtype, device=baseline_loss.device,
    )[selected_index]
    rows = torch.arange(batch, device=baseline_loss.device)
    selected_loss = candidate_losses[selected_index, rows]
    replay_eligible = eligible[selected_index, rows]
    largest_replay = torch.ones(batch, dtype=torch.bool, device=baseline_loss.device)
    for row in range(batch):
        expected = next(
            (index for index in range(len(SCALES)) if bool(eligible[index, row])),
            len(SCALES) - 1,
        )
        largest_replay[row] = int(selected_index[row]) == expected
    return {
        "selected_index": selected_index,
        "selected_scale": selected_scale,
        "selected_loss": selected_loss,
        "improved": improved,
        "group_passes": group_passes,
        "eligible": eligible,
        "natural_group_rms": natural,
        "candidate_group_rms": candidate,
        "selected_is_eligible": replay_eligible,
        "largest_eligible_replay": largest_replay,
        "finite": bool(
            torch.isfinite(baseline_loss).all()
            and torch.isfinite(candidate_losses).all()
            and torch.isfinite(natural_response).all()
            and torch.isfinite(candidate_responses).all()
            and all(torch.isfinite(value).all() for value in natural.values())
            and all(torch.isfinite(value).all() for value in candidate.values())
        ),
    }


def aggregate_by_sequence(
    names: Sequence[str], values: Sequence[float],
) -> Tuple[Sequence[str], np.ndarray]:
    if len(names) != len(values) or not names:
        raise ValueError("D2-S sequence aggregation requires matching values")
    grouped = defaultdict(list)
    for name, value in zip(names, values):
        grouped[str(name)].append(float(value))
    order = sorted(grouped)
    array = np.asarray(
        [np.mean(grouped[name]) for name in order], dtype=np.float64,
    )
    if not np.isfinite(array).all():
        raise ValueError("D2-S sequence aggregate is nonfinite")
    return order, array


def paired_sequence_difference(
    names: Sequence[str], candidate: Sequence[float], control: Sequence[float],
) -> Dict[str, object]:
    first_names, first = aggregate_by_sequence(names, candidate)
    second_names, second = aggregate_by_sequence(names, control)
    if first_names != second_names:
        raise ValueError("D2-S paired sequence ordering differs")
    return paired_difference(first, second)


def paired_sequence_ratio(
    names: Sequence[str], candidate: Sequence[float], control: Sequence[float],
) -> Dict[str, object]:
    first_names, first = aggregate_by_sequence(names, candidate)
    second_names, second = aggregate_by_sequence(names, control)
    if first_names != second_names:
        raise ValueError("D2-S ratio sequence ordering differs")
    return paired_mean_ratio(first, second)


def mechanism_gate(
    contract: Mapping[str, bool], timesteps: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    contract_passed = bool(contract) and all(bool(value) for value in contract.values())
    timestep_results = {}
    for timestep in GATE_TIMESTEPS:
        record = timesteps.get(str(timestep), {})
        comparison = record.get("controller_comparison", {})
        fixed_loss_ci = comparison.get("fixed_mask_author_loss", {}).get(
            "bootstrap_95_ci", [float("nan"), float("nan")],
        )
        contact = comparison.get("fk_union_5cm", {})
        checks = {
            "nonzero_selected_fraction_ge_0.50": bool(
                math.isfinite(float(record.get("nonzero_selected_fraction", float("nan"))))
                and float(record["nonzero_selected_fraction"]) >= 0.50
            ),
            "fixed_mask_loss_improvement_ci_lower_gt_zero": bool(
                len(fixed_loss_ci) >= 1
                and math.isfinite(float(fixed_loss_ci[0]))
                and float(fixed_loss_ci[0]) > 0.0
            ),
        }
        for metric in ("recall", "f1"):
            ci = contact.get(metric, {}).get(
                "bootstrap_95_ci", [float("nan"), float("nan")],
            )
            checks[f"fk_union_5cm_{metric}_ci_lower_gt_zero"] = bool(
                len(ci) >= 1 and math.isfinite(float(ci[0])) and float(ci[0]) > 0.0
            )
        ratios = comparison.get("ratios", {})
        for metric in GATE_RATIO_METRICS:
            ci = ratios.get(metric, {}).get(
                "bootstrap_95_ci", [float("nan"), float("nan")],
            )
            checks[f"{metric}_ratio_ci_upper_le_1.05"] = bool(
                len(ci) >= 2 and math.isfinite(float(ci[1])) and float(ci[1]) <= 1.05
            )
        timestep_results[str(timestep)] = {
            "passed": all(checks.values()),
            "checks": checks,
        }
    passing = sum(int(value["passed"]) for value in timestep_results.values())
    positive = bool(contract_passed and passing >= 4)
    if not contract_passed:
        classification = "denoiser-response-frontier-contract-failure-stop"
    elif positive:
        classification = "denoiser-response-frontier-positive-stop"
    else:
        classification = "denoiser-response-frontier-negative-stop"
    return {
        "classification": classification,
        "contract_passed": contract_passed,
        "mechanism_positive": positive,
        "contract_checks": dict(contract),
        "gate_timesteps": list(GATE_TIMESTEPS),
        "minimum_passing_timesteps": 4,
        "passing_timesteps": passing,
        "timestep_results": timestep_results,
        "full_trajectory_controller_authorized": False,
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
    }
