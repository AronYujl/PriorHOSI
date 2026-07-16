"""Locked utilities for the Phase 1B D2-Q0 contact-guidance counterfactual."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .contact_alignment import geometry_report
from .diffusion import prepare_clean_x0
from .losses import _fk_positions
from .optimizer_reset import paired_difference, paired_mean_ratio
from .remediation import selection_sha256, stable_digest
from .representation import REPRESENTATION


RUN_ID = "p1-hoi-d2q-author-contact-guidance-s42-20260716"
MODELS: Tuple[str, ...] = ("source", "current", "balanced")
VARIANTS: Tuple[str, ...] = ("unguided", "guided")
EXPECTED_CHECKPOINT_SHA256 = {
    "source": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
    "current": "76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f",
    "balanced": "ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8",
}
AUTHOR_COMMIT = "b9a158f75ab0740c91c9cfc8863a65fa381b014c"
AUTHOR_BLOB_SHA256 = {
    "code/guidance_loss.py": "5747721bcc015911c7999692079666f7e5d6204912761d91b8b82d4ba4c4ab24",
    "code/models/infbagel.py": "6dec84991c685e11d1f8e4f2f928497673210066d38a5901d0537a40504f9d76",
    "code/config/config_sample_infbagel.yaml": (
        "39490dc60c47da7273434d65fa95e44a2d34382c872fe7c1718c14886bdde388"
    ),
}
PHASE_OFFSETS: Tuple[int, ...] = (28, 70, 112)
PRIOR_OFFSETS: Tuple[int, ...] = (0, 14, 42, 56, 84, 98)
SEQUENCES = 64
WINDOWS_PER_SEQUENCE = 3
SELECTION_SHA256 = "337ba964d4384bc66664ceeb148eb960632bac3861718063b6f86f43c59c5344"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
HISTORY_MAX_ABS = 1e-5
SEMANTIC_THRESHOLD = 0.95
SPATIAL_THRESHOLD_M = 0.02
GUIDANCE_SCALE = 1.0
AUTHOR_HAND_WEIGHT = 10.0
REST_VERTEX_COUNT = 2048
FK_PALM_INDICES: Tuple[int, int] = (22, 23)
DIRECT_HAND_INDICES: Tuple[int, int] = (24, 26)
KINEMATIC_METRICS: Tuple[str, ...] = (
    "mpjpe_cm",
    "object_goal_error_cm",
    "pelvis_goal_error_cm",
    "object_translation_mae_cm",
    "foot_sliding",
)


def select_guidance_holdout(dataset) -> Dict[str, object]:
    """Return the locked fresh 64-sequence D2-Q phase-offset selection."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-Q selection is internal-validation only")
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
                    f"42:d2q-author-contact-guidance:{name}:{suffix}"
                ),
                name,
                sequence,
                positions,
            ))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(eligible) < SEQUENCES:
        raise ValueError(f"D2-Q requires {SEQUENCES} eligible sequences")
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
        "prior_offsets": list(PRIOR_OFFSETS),
        "sequences": len(triples),
        "windows": len(global_indices),
        "eligible_sequences": len(eligible),
        "sequence_names": [row[1] for row in rows],
    }
    if result["sha256"] != SELECTION_SHA256:
        raise ValueError(f"D2-Q selection mismatch: {result['sha256']}")
    return result


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    if chunk_index < 0 or window_index not in range(WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-Q sampler seed coordinates")
    return f"D2:d2q-shared:chunk:{chunk_index}:window:{window_index}"


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def deterministic_vertex_subset(rest_vertices: torch.Tensor) -> torch.Tensor:
    """Use the preregistered uniform-index 2,048-vertex surface subset."""
    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        raise ValueError("D2-Q rest vertices must be [vertices,3]")
    if rest_vertices.shape[0] < REST_VERTEX_COUNT:
        raise ValueError(
            f"D2-Q requires at least {REST_VERTEX_COUNT} rest vertices"
        )
    indices = torch.linspace(
        0,
        rest_vertices.shape[0] - 1,
        REST_VERTEX_COUNT,
        device=rest_vertices.device,
    ).round().long()
    return rest_vertices[indices]


def transformed_object_vertices(
    rest_vertices: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    if rest_vertices.ndim != 3 or rest_vertices.shape[2] != 3:
        raise ValueError("D2-Q batched rest vertices must be [B,V,3]")
    if rotation.ndim != 4 or rotation.shape[-2:] != (3, 3):
        raise ValueError("D2-Q object rotation must be [B,T,3,3]")
    if translation.shape != rotation.shape[:2] + (3,):
        raise ValueError("D2-Q object translation shape differs from rotation")
    if rest_vertices.shape[0] != rotation.shape[0]:
        raise ValueError("D2-Q rest-vertex batch differs from object pose")
    return (
        torch.einsum("bvc,btdc->btvd", rest_vertices.to(rotation), rotation)
        + translation[:, :, None]
    )


def author_hand_object_components(
    human_joints: torch.Tensor,
    object_vertices: torch.Tensor,
    object_translation: torch.Tensor,
    object_rotation: torch.Tensor,
    contact: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Replay the author hand-object formula without its outer HOI weights."""
    if human_joints.ndim != 4 or human_joints.shape[2:] != (24, 3):
        raise ValueError("D2-Q author guidance expects [B,T,24,3] FK joints")
    batch, frames = human_joints.shape[:2]
    if object_vertices.ndim != 4 or object_vertices.shape[:2] != (batch, frames):
        raise ValueError("D2-Q author guidance object vertices differ from joints")
    if object_translation.shape != (batch, frames, 3):
        raise ValueError("D2-Q author guidance object translation differs")
    if object_rotation.shape != (batch, frames, 3, 3):
        raise ValueError("D2-Q author guidance object rotation differs")
    if contact.shape != (batch, frames, 4):
        raise ValueError("D2-Q author guidance contact must be [B,T,4]")

    palms = human_joints[:, :, FK_PALM_INDICES]
    distances = torch.cdist(
        palms.reshape(batch * frames, 2, 3),
        object_vertices.reshape(batch * frames, object_vertices.shape[2], 3),
    ).amin(dim=-1)
    contact_mask = (
        contact[:, :, :2] > SEMANTIC_THRESHOLD
    ).reshape(batch * frames, 2).detach().to(distances)
    zeros = torch.zeros_like(distances)
    spatial = F.l1_loss(
        torch.maximum(
            distances * contact_mask - SPATIAL_THRESHOLD_M,
            zeros,
        ),
        zeros,
    )

    left_relative = human_joints[:, :, FK_PALM_INDICES[0]] - object_translation.detach()
    right_relative = human_joints[:, :, FK_PALM_INDICES[1]] - object_translation.detach()
    detached_inverse_rotation = object_rotation.detach().transpose(2, 3)
    left_relative = torch.matmul(
        detached_inverse_rotation, left_relative[:, :, :, None],
    ).squeeze(-1)
    right_relative = torch.matmul(
        detached_inverse_rotation, right_relative[:, :, :, None],
    ).squeeze(-1)
    sequence_mask = contact_mask.reshape(batch, frames, 2)
    left_mask = (
        sequence_mask[:, :, 0:1]
        * sequence_mask[:, :, 0:1].transpose(-1, -2)
    )
    right_mask = (
        sequence_mask[:, :, 1:2]
        * sequence_mask[:, :, 1:2].transpose(-1, -2)
    )
    left_normalized = left_relative / torch.norm(
        left_relative, dim=-1, keepdim=True,
    )
    right_normalized = right_relative / torch.norm(
        right_relative, dim=-1, keepdim=True,
    )
    left_similarity = torch.matmul(
        left_normalized, left_normalized.transpose(-1, -2),
    )
    right_similarity = torch.matmul(
        right_normalized, right_normalized.transpose(-1, -2),
    )
    temporal = (
        1.0 - torch.mean(left_similarity * left_mask)
        + 1.0 - torch.mean(right_similarity * right_mask)
    )
    total = batch * (spatial + temporal)
    return {
        "total": total,
        "spatial": spatial,
        "temporal": temporal,
        "mask_coverage": sequence_mask.float().mean(),
        "distance_mean_m": distances.mean(),
    }


def decoded_fk_positions(
    decoded: Mapping[str, torch.Tensor],
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
) -> torch.Tensor:
    return _fk_positions(
        decoded["joints"][..., 0, :],
        decoded["human_rotation"],
        rest_human_offsets,
        parents_24,
    )


def guidance_gradient(
    clean: torch.Tensor,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return the author-formula negative loss gradient and scalar audit."""
    with torch.enable_grad():
        differentiable_clean = clean.detach().requires_grad_(True)
        decoded = codec.decode(differentiable_clean, frame)
        fk_joints = decoded_fk_positions(
            decoded, rest_human_offsets, parents_24,
        )
        vertices = transformed_object_vertices(
            rest_vertices,
            decoded["object_rotation"],
            decoded["object_translation"],
        )
        components = author_hand_object_components(
            fk_joints,
            vertices,
            decoded["object_translation"],
            decoded["object_rotation"],
            decoded["contact"],
        )
        effective_loss = AUTHOR_HAND_WEIGHT * components["total"]
        gradient = torch.autograd.grad(
            -effective_loss, differentiable_clean,
        )[0] * GUIDANCE_SCALE
    finite = bool(
        torch.isfinite(gradient).all()
        and all(torch.isfinite(value) for value in components.values())
    )
    audit = {
        "loss": float(effective_loss.detach().cpu()),
        "raw_hand_loss": float(components["total"].detach().cpu()),
        "author_hand_weight": AUTHOR_HAND_WEIGHT,
        "spatial": float(components["spatial"].detach().cpu()),
        "temporal": float(components["temporal"].detach().cpu()),
        "mask_coverage": float(components["mask_coverage"].detach().cpu()),
        "distance_mean_m": float(components["distance_mean_m"].detach().cpu()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
        "gradient_rms": float(gradient.square().mean().sqrt().detach().cpu()),
        "gradient_max_abs": float(gradient.abs().max().detach().cpu()),
        "finite": finite,
    }
    return gradient.detach(), audit


def apply_guidance_update(
    posterior: torch.Tensor,
    clean: torch.Tensor,
    fixed_history: torch.Tensor,
    *,
    reverse_step: int,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[Dict[str, object]]]:
    """Apply the locked author update for steps 499..1 and restore history."""
    if reverse_step == 0:
        result = posterior.clone()
        result[:, :REPRESENTATION.history_frames] = fixed_history
        return result, None
    gradient, audit = guidance_gradient(
        clean,
        codec,
        frame,
        rest_human_offsets,
        parents_24,
        rest_vertices,
    )
    result = posterior + gradient
    result[:, :REPRESENTATION.history_frames] = fixed_history
    return result, audit


def sample_contact_counterfactual(
    diffusion,
    model,
    fixed_history: torch.Tensor,
    text_embedding: torch.Tensor,
    object_bps: torch.Tensor,
    goals: torch.Tensor,
    progress: torch.Tensor,
    *,
    generator: torch.Generator,
    guided: bool,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Run one analysis-only trajectory with the production posterior helper."""
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
        if guided:
            current, audit = apply_guidance_update(
                posterior,
                clean,
                fixed_history,
                reverse_step=step,
                codec=codec,
                frame=frame,
                rest_human_offsets=rest_human_offsets,
                parents_24=parents_24,
                rest_vertices=rest_vertices,
            )
            if audit is not None:
                per_step.append({"reverse_step": step, **audit})
        else:
            current = posterior
        history_max_abs = max(
            history_max_abs,
            float(
                (
                    current[:, :REPRESENTATION.history_frames] - fixed_history
                ).abs().max().detach().cpu()
            ),
        )
    return current, {
        "guided": guided,
        "applied_steps": len(per_step),
        "step_zero_guidance_applied": False,
        "history_max_abs": history_max_abs,
        "finite": bool(
            torch.isfinite(current).all()
            and all(bool(value["finite"]) for value in per_step)
        ),
        "per_step": per_step,
    }


def hand_distances(
    joints: torch.Tensor,
    vertices: torch.Tensor,
    indices: Sequence[int],
) -> torch.Tensor:
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError("D2-Q hand distances require [frames,joints,3]")
    if tuple(indices) not in (FK_PALM_INDICES, DIRECT_HAND_INDICES):
        raise ValueError("D2-Q hand indices are not preregistered")
    if joints.shape[1] <= max(indices):
        raise ValueError("D2-Q joint tensor lacks requested hand indices")
    if vertices.ndim != 3 or vertices.shape[:1] != joints.shape[:1]:
        raise ValueError("D2-Q object vertices differ from joint frames")
    return torch.cdist(joints[:, indices], vertices).amin(dim=-1)


def paired_guidance_comparison(
    unguided_records: Sequence[Mapping[str, object]],
    guided_records: Sequence[Mapping[str, object]],
    unguided_kinematics: Mapping[str, object],
    guided_kinematics: Mapping[str, object],
) -> Dict[str, object]:
    if [value["sequence"] for value in unguided_records] != [
        value["sequence"] for value in guided_records
    ]:
        raise ValueError("D2-Q paired records differ in sequence ordering")

    def contact_values(records, geometry: str, metric: str) -> np.ndarray:
        values = []
        for record in records:
            union = record[geometry]["thresholds_cm"]["5"]["union"]
            if metric == "prediction_run_mean_frames":
                value = union["prediction_run_lengths"]["mean_frames"]
            else:
                value = union[metric]
            values.append(float(value))
        return np.asarray(values, dtype=np.float64)

    def kinematic_values(metrics, key: str) -> np.ndarray:
        if key == "foot_sliding":
            rows = sorted(
                metrics["per_sequence"], key=lambda value: value["sequence"],
            )
            return np.asarray(
                [float(value[key]) for value in rows], dtype=np.float64,
            )
        grouped = defaultdict(list)
        for value in metrics["per_sequence_window"]:
            if key == "object_goal_error_cm" and int(value["window"]) != 3:
                continue
            grouped[value["sequence"]].append(float(value[key]))
        return np.asarray(
            [np.mean(grouped[name]) for name in sorted(grouped)],
            dtype=np.float64,
        )

    result = {"contact": {}, "kinematics": {}}
    for geometry in ("fk_physical_geometry_vs_gt", "direct_physical_geometry_vs_gt"):
        result["contact"][geometry] = {}
        for metric in (
            "recall",
            "f1",
            "prediction_percent",
            "precision",
            "prediction_run_mean_frames",
        ):
            guided = contact_values(guided_records, geometry, metric)
            unguided = contact_values(unguided_records, geometry, metric)
            value = paired_difference(guided, unguided)
            if metric == "prediction_run_mean_frames":
                denominator = float(unguided.mean())
                value["guided_over_unguided_mean_ratio"] = (
                    float(guided.mean() / denominator)
                    if denominator > 0.0 else None
                )
            result["contact"][geometry][metric] = value
    for metric in KINEMATIC_METRICS:
        guided = kinematic_values(guided_kinematics, metric)
        unguided = kinematic_values(unguided_kinematics, metric)
        result["kinematics"][metric] = paired_mean_ratio(
            guided, unguided,
        )
    return result


def mechanism_gate(
    contract: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    contract_passed = bool(contract) and all(
        bool(value) for value in contract.values()
    )
    balanced = comparisons.get("balanced", {})
    fk = balanced.get("contact", {}).get(
        "fk_physical_geometry_vs_gt", {},
    )
    contact_checks = {}
    for metric in (
        "recall",
        "f1",
        "prediction_percent",
        "prediction_run_mean_frames",
    ):
        ci = fk.get(metric, {}).get("bootstrap_95_ci", [float("nan")])
        contact_checks[f"{metric}_ci_lower_gt_zero"] = bool(
            len(ci) >= 1 and math.isfinite(float(ci[0]))
            and float(ci[0]) > 0.0
        )
    precision_ci = fk.get("precision", {}).get(
        "bootstrap_95_ci", [float("nan")],
    )
    contact_checks["precision_ci_lower_ge_minus_0.02"] = bool(
        len(precision_ci) >= 1
        and math.isfinite(float(precision_ci[0]))
        and float(precision_ci[0]) >= -0.02
    )
    run_ratio = fk.get("prediction_run_mean_frames", {}).get(
        "guided_over_unguided_mean_ratio",
    )
    contact_checks["run_mean_ratio_ge_1.5"] = bool(
        run_ratio is not None
        and math.isfinite(float(run_ratio))
        and float(run_ratio) >= 1.5
    )
    kinematic_checks = {}
    for metric in KINEMATIC_METRICS:
        ci = balanced.get("kinematics", {}).get(metric, {}).get(
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
        classification = "author-contact-guidance-contract-failure-stop"
    elif positive:
        classification = "author-contact-guidance-positive-stop"
    else:
        classification = "author-contact-guidance-negative-stop"
    return {
        "classification": classification,
        "contract_passed": contract_passed,
        "mechanism_positive": positive,
        "contract_checks": dict(contract),
        "balanced_contact_checks": contact_checks,
        "balanced_kinematic_checks": kinematic_checks,
        "primary_gate_checkpoint": "balanced",
        "checkpoint_selected": False,
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
    }
