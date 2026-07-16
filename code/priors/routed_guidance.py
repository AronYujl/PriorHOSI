"""Locked utilities for the Phase 1B D2-R0 state-routed guidance diagnostic."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .contact_guidance import (
    AUTHOR_HAND_WEIGHT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FK_PALM_INDICES,
    HISTORY_MAX_ABS,
    guidance_gradient,
)
from .diffusion import prepare_clean_x0
from .optimizer_reset import paired_difference, paired_mean_ratio
from .remediation import selection_sha256, stable_digest
from .representation import REPRESENTATION


RUN_ID = "p1-hoi-d2r-state-routed-guidance-s42-20260717"
SUBPHASE = "1B-D2-R0"
CHECKPOINT_SHA256 = (
    "ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8"
)
PHASE_OFFSETS: Tuple[int, ...] = (7, 49, 91)
PRIOR_ROLLOUT_OFFSETS: Tuple[int, ...] = (
    0, 14, 28, 42, 56, 70, 84, 98, 112,
)
SEQUENCES = 64
WINDOWS_PER_SEQUENCE = 3
SELECTION_SHA256 = (
    "189e3f05e28007b3ba3dab25a6cf6afd63ed981135722ae41987129219bfd9da"
)
VARIANTS: Tuple[str, ...] = (
    "unguided",
    "author_all",
    "human_only",
    "upper_raw",
    "upper_norm",
)
PRIMARY_VARIANT = "upper_norm"
UPPER_ROTATION_JOINTS: Tuple[int, ...] = (
    3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21,
)
MASKED_OFF_MAX_ABS = 1e-7
NORM_REPLAY_RELATIVE_ERROR = 1e-5
KINEMATIC_METRICS: Tuple[str, ...] = (
    "fk_mpjpe_cm",
    "pelvis_goal_error_cm",
    "object_goal_error_cm",
    "object_translation_mae_cm",
    "object_rotation_geodesic",
    "fk_foot_sliding",
)


def select_routed_holdout(dataset) -> Dict[str, object]:
    """Return the locked fresh 64-sequence D2-R phase-offset selection."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-R selection is internal-validation only")
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
                    f"42:d2r-routed-guidance:{name}:{suffix}"
                ),
                name,
                sequence,
                positions,
            ))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(eligible) < SEQUENCES:
        raise ValueError(f"D2-R requires {SEQUENCES} eligible sequences")
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
        raise ValueError(f"D2-R selection mismatch: {result['sha256']}")
    return result


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    if chunk_index < 0 or window_index not in range(WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-R sampler seed coordinates")
    return f"D2:d2r-shared:chunk:{chunk_index}:window:{window_index}"


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def upper_rotation_mask(*, device=None) -> torch.Tensor:
    """Return the preregistered 232-channel upper-chain projection mask."""
    mask = torch.zeros(REPRESENTATION.dimension, dtype=torch.bool, device=device)
    rotation = REPRESENTATION.field("joint_rotations_6d")
    for joint in UPPER_ROTATION_JOINTS:
        start = rotation.start + joint * 6
        mask[start:start + 6] = True
    return mask


def _variant_mask(variant: str, *, device=None) -> torch.Tensor:
    if variant not in VARIANTS:
        raise ValueError(f"unknown D2-R variant: {variant}")
    if variant == "unguided":
        return torch.zeros(
            REPRESENTATION.dimension, dtype=torch.bool, device=device,
        )
    if variant == "author_all":
        return torch.ones(
            REPRESENTATION.dimension, dtype=torch.bool, device=device,
        )
    if variant == "human_only":
        mask = torch.ones(
            REPRESENTATION.dimension, dtype=torch.bool, device=device,
        )
        mask[
            REPRESENTATION.field("object_translation").start:
            REPRESENTATION.field("object_rotation").stop
        ] = False
        return mask
    return upper_rotation_mask(device=device)


def _mutable_flat(value: torch.Tensor) -> torch.Tensor:
    return value[:, REPRESENTATION.history_frames:].reshape(value.shape[0], -1)


def _gradient_measurements(
    full_gradient: torch.Tensor,
    injected_gradient: torch.Tensor,
) -> Dict[str, object]:
    packed = []
    rotation = REPRESENTATION.field("joint_rotations_6d")
    for value in (full_gradient, injected_gradient):
        mutable = value[:, REPRESENTATION.history_frames:]
        packed.extend((
            torch.linalg.vector_norm(mutable),
            mutable.square().mean().sqrt(),
            mutable.abs().max(),
        ))
        packed.extend(
            mutable[..., field.slice].square().sum()
            for field in REPRESENTATION.fields
        )
        packed.extend(
            mutable[
                ..., rotation.start + joint * 6:
                rotation.start + (joint + 1) * 6
            ].square().sum()
            for joint in range(22)
        )
    values = torch.stack(tuple(packed)).detach().cpu().tolist()
    width = 3 + len(REPRESENTATION.fields) + 22
    result = {}
    for source_index, source in enumerate((
        "full_gradient", "injected_gradient",
    )):
        offset = source_index * width
        result[source] = {
            "norm": float(values[offset]),
            "rms": float(values[offset + 1]),
            "max_abs": float(values[offset + 2]),
            "fields": {
                field.name: float(values[offset + 3 + index])
                for index, field in enumerate(REPRESENTATION.fields)
            },
            "rotation_joints": {
                str(joint): float(
                    values[
                        offset + 3 + len(REPRESENTATION.fields) + joint
                    ]
                )
                for joint in range(22)
            },
        }
    return result


def route_gradient(
    full_gradient: torch.Tensor,
    variant: str,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Project the fixed author hand gradient into a preregistered state subspace."""
    if full_gradient.ndim != 3 or full_gradient.shape[-1] != REPRESENTATION.dimension:
        raise ValueError("D2-R gradient must be [B,T,232]")
    if variant == "unguided":
        raise ValueError("unguided has no routed gradient")
    mask = _variant_mask(variant, device=full_gradient.device)
    routed = full_gradient * mask.reshape(1, 1, -1)
    routed[:, :REPRESENTATION.history_frames] = 0.0
    full_mutable = _mutable_flat(full_gradient)
    projected_mutable = _mutable_flat(routed)
    full_norm = torch.linalg.vector_norm(full_mutable, dim=1)
    projected_norm = torch.linalg.vector_norm(projected_mutable, dim=1)
    invalid_zero = (full_norm > 0.0) & (projected_norm == 0.0)
    scale = torch.ones_like(full_norm)
    if variant == "upper_norm":
        valid = projected_norm > 0.0
        scale[valid] = full_norm[valid] / projected_norm[valid]
        routed = routed * scale[:, None, None]
    expected = full_gradient * mask.reshape(1, 1, -1)
    expected[:, :REPRESENTATION.history_frames] = 0.0
    if variant == "upper_norm":
        expected = expected * scale[:, None, None]
    routed_norm = torch.linalg.vector_norm(_mutable_flat(routed), dim=1)
    replay = torch.zeros_like(full_norm)
    nonzero = full_norm > 0.0
    replay[nonzero] = (routed_norm[nonzero] - full_norm[nonzero]).abs() / full_norm[nonzero]
    if variant != "upper_norm":
        replay.zero_()
    disallowed = routed * (~mask).reshape(1, 1, -1)
    per_sample = torch.stack(
        (scale, full_norm, projected_norm, routed_norm), dim=1,
    ).detach().cpu().tolist()
    scalar_values = torch.stack((
        replay.max(),
        disallowed.abs().max(),
        routed[:, :REPRESENTATION.history_frames].abs().max(),
        (routed - expected).abs().max(),
    )).detach().cpu().tolist()
    measurements = _gradient_measurements(full_gradient, routed)
    audit = {
        "variant": variant,
        "routing_scale": [float(value[0]) for value in per_sample],
        "full_mutable_norm_per_sample": [
            float(value[1]) for value in per_sample
        ],
        "projected_mutable_norm_per_sample": [
            float(value[2]) for value in per_sample
        ],
        "routed_mutable_norm_per_sample": [
            float(value[3]) for value in per_sample
        ],
        "invalid_nonzero_full_zero_projection": bool(invalid_zero.any()),
        "norm_replay_relative_error_max": float(scalar_values[0]),
        "masked_off_max_abs": float(scalar_values[1]),
        "routed_history_max_abs": float(scalar_values[2]),
        "routing_formula_replay_max_abs": float(scalar_values[3]),
        **measurements,
        "finite": bool(
            torch.isfinite(routed).all()
            and torch.isfinite(scale).all()
            and torch.isfinite(replay).all()
        ),
    }
    return routed, audit


def apply_routed_guidance_update(
    posterior: torch.Tensor,
    clean: torch.Tensor,
    fixed_history: torch.Tensor,
    *,
    reverse_step: int,
    variant: str,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[Dict[str, object]]]:
    """Apply one routed author-hand update and restore immutable history."""
    if reverse_step == 0 or variant == "unguided":
        result = posterior.clone()
        result[:, :REPRESENTATION.history_frames] = fixed_history
        return result, None
    full_gradient, author_audit = guidance_gradient(
        clean,
        codec,
        frame,
        rest_human_offsets,
        parents_24,
        rest_vertices,
    )
    routed, routing_audit = route_gradient(full_gradient, variant)
    result = posterior + routed
    result[:, :REPRESENTATION.history_frames] = fixed_history
    return result, {**author_audit, **routing_audit}


def sample_routed_counterfactual(
    diffusion,
    model,
    fixed_history: torch.Tensor,
    text_embedding: torch.Tensor,
    object_bps: torch.Tensor,
    goals: torch.Tensor,
    progress: torch.Tensor,
    *,
    generator: torch.Generator,
    variant: str,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Run one paired trajectory with production posterior and routed guidance."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown D2-R variant: {variant}")
    batch = fixed_history.shape[0]
    current = torch.randn(
        (batch, REPRESENTATION.window_frames, REPRESENTATION.dimension),
        device=fixed_history.device,
        generator=generator,
    )
    current[:, :REPRESENTATION.history_frames] = fixed_history
    per_step = []
    history_max_abs = 0.0
    for step in reversed(range(diffusion.timesteps)):
        timesteps = torch.full(
            (batch,), step, dtype=torch.long, device=current.device,
        )
        with torch.no_grad():
            clean = model(
                current, timesteps, text_embedding, object_bps, goals, progress,
            )
            clean = prepare_clean_x0(
                clean, fixed_history, object_so3_x0=False,
            )
            if step:
                noise = torch.randn(
                    current.shape, device=current.device, generator=generator,
                )
            else:
                noise = torch.zeros_like(current)
            posterior = diffusion.posterior_sample(
                current, clean, timesteps, noise, fixed_history,
            )
        current, audit = apply_routed_guidance_update(
            posterior,
            clean,
            fixed_history,
            reverse_step=step,
            variant=variant,
            codec=codec,
            frame=frame,
            rest_human_offsets=rest_human_offsets,
            parents_24=parents_24,
            rest_vertices=rest_vertices,
        )
        if audit is not None:
            per_step.append({"reverse_step": step, **audit})
        history_max_abs = max(
            history_max_abs,
            float(
                (
                    current[:, :REPRESENTATION.history_frames] - fixed_history
                ).abs().max().detach().cpu()
            ),
        )
    return current, {
        "variant": variant,
        "guided": variant != "unguided",
        "applied_steps": len(per_step),
        "step_zero_guidance_applied": False,
        "history_max_abs": history_max_abs,
        "finite": bool(
            torch.isfinite(current).all()
            and all(bool(value["finite"]) for value in per_step)
        ),
        "per_step": per_step,
    }


def _contact_values(
    records: Sequence[Mapping[str, object]],
    metric: str,
) -> np.ndarray:
    values = []
    for record in records:
        union = record["fk_physical_geometry_vs_gt"][
            "thresholds_cm"
        ]["5"]["union"]
        if metric == "prediction_run_mean_frames":
            value = union["prediction_run_lengths"]["mean_frames"]
        else:
            value = union[metric]
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def paired_variant_comparison(
    control_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
    control_kinematics: Mapping[str, object],
    candidate_kinematics: Mapping[str, object],
) -> Dict[str, object]:
    if [value["sequence"] for value in control_records] != [
        value["sequence"] for value in candidate_records
    ]:
        raise ValueError("D2-R paired records differ in sequence ordering")
    control_rows = sorted(
        control_kinematics["per_sequence"],
        key=lambda value: value["sequence"],
    )
    candidate_rows = sorted(
        candidate_kinematics["per_sequence"],
        key=lambda value: value["sequence"],
    )
    if [row["sequence"] for row in control_rows] != [
        row["sequence"] for row in candidate_rows
    ]:
        raise ValueError("D2-R paired kinematics differ in sequence ordering")
    result = {"contact": {}, "kinematics": {}}
    for metric in (
        "recall",
        "f1",
        "prediction_percent",
        "precision",
        "prediction_run_mean_frames",
    ):
        candidate = _contact_values(candidate_records, metric)
        control = _contact_values(control_records, metric)
        value = paired_difference(candidate, control)
        if metric == "prediction_run_mean_frames":
            denominator = float(control.mean())
            value["candidate_over_control_mean_ratio"] = (
                float(candidate.mean() / denominator)
                if denominator > 0.0 else None
            )
        result["contact"][metric] = value
    for metric in KINEMATIC_METRICS:
        candidate = np.asarray(
            [float(row[metric]) for row in candidate_rows], dtype=np.float64,
        )
        control = np.asarray(
            [float(row[metric]) for row in control_rows], dtype=np.float64,
        )
        result["kinematics"][metric] = paired_mean_ratio(
            candidate, control,
        )
    return result


def mechanism_gate(
    contract: Mapping[str, bool],
    comparison: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    contract_passed = bool(contract) and all(
        bool(value) for value in contract.values()
    )
    contact_checks = {}
    for metric in (
        "recall",
        "f1",
        "prediction_percent",
        "prediction_run_mean_frames",
    ):
        ci = comparison.get("contact", {}).get(metric, {}).get(
            "bootstrap_95_ci", [float("nan")],
        )
        contact_checks[f"{metric}_ci_lower_gt_zero"] = bool(
            len(ci) >= 1
            and math.isfinite(float(ci[0]))
            and float(ci[0]) > 0.0
        )
    precision_ci = comparison.get("contact", {}).get(
        "precision", {},
    ).get("bootstrap_95_ci", [float("nan")])
    contact_checks["precision_ci_lower_ge_minus_0.02"] = bool(
        len(precision_ci) >= 1
        and math.isfinite(float(precision_ci[0]))
        and float(precision_ci[0]) >= -0.02
    )
    run_ratio = comparison.get("contact", {}).get(
        "prediction_run_mean_frames", {},
    ).get("candidate_over_control_mean_ratio")
    contact_checks["run_mean_ratio_ge_1.5"] = bool(
        run_ratio is not None
        and math.isfinite(float(run_ratio))
        and float(run_ratio) >= 1.5
    )
    kinematic_checks = {}
    for metric in KINEMATIC_METRICS:
        ci = comparison.get("kinematics", {}).get(metric, {}).get(
            "bootstrap_95_ci", [float("nan"), float("nan")],
        )
        kinematic_checks[f"{metric}_ratio_ci_upper_le_1.10"] = bool(
            len(ci) >= 2
            and math.isfinite(float(ci[1]))
            and float(ci[1]) <= 1.10
        )
    positive = bool(
        contract_passed
        and all(contact_checks.values())
        and all(kinematic_checks.values())
    )
    if not contract_passed:
        classification = "state-routed-guidance-contract-failure-stop"
    elif positive:
        classification = "state-routed-guidance-positive-stop"
    else:
        classification = "state-routed-guidance-negative-stop"
    return {
        "classification": classification,
        "contract_passed": contract_passed,
        "mechanism_positive": positive,
        "contract_checks": dict(contract),
        "upper_norm_contact_checks": contact_checks,
        "upper_norm_kinematic_checks": kinematic_checks,
        "primary_gate_variant": PRIMARY_VARIANT,
        "checkpoint_selected": False,
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
    }
