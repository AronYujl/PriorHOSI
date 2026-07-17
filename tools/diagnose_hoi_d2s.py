#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-S0 denoiser-response frontier."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from pytorch3d import transforms


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.contact_alignment import (  # noqa: E402
    PHYSICAL_THRESHOLDS_CM,
    geometry_report,
)
from priors.contact_guidance import (  # noqa: E402
    AUTHOR_BLOB_SHA256,
    AUTHOR_COMMIT,
    AUTHOR_HAND_WEIGHT,
    DIRECT_HAND_INDICES,
    FK_PALM_INDICES,
    REST_VERTEX_COUNT,
    SEMANTIC_THRESHOLD,
    SPATIAL_THRESHOLD_M,
    author_hand_object_components,
    decoded_fk_positions,
    deterministic_vertex_subset,
    guidance_gradient,
    transformed_object_vertices,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.denoiser_response import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINT_SHA256,
    DIRECTIONS,
    FIELD_METRICS,
    GATE_RATIO_METRICS,
    GATE_TIMESTEPS,
    HISTORY_MAX_ABS,
    PARENT_TIMESTEPS,
    PHASE_OFFSETS,
    PHYSICAL_METRICS,
    PRIOR_ROLLOUT_OFFSETS,
    PROTECTED_GROUPS,
    RUN_IDS,
    RUN_SUBPHASES,
    SCALES,
    SELECTION_SHA256,
    TARGET_TIMESTEPS,
    TRUST_RATIO,
    UPPER_ROTATION_JOINTS,
    WINDOWS_PER_SEQUENCE,
    author_components_per_sample,
    direction_update,
    fixed_mask_from_contact,
    mechanism_gate,
    paired_sequence_difference,
    paired_sequence_ratio,
    scaled_candidate_batch,
    select_largest_eligible_scale,
    select_response_holdout,
    unpack_scale_major,
)
from priors.diffusion import (  # noqa: E402
    GaussianDiffusion,
    _extract,
    normalize_progress,
    prepare_clean_x0,
)
from priors.exposure import fieldwise_mse_per_sample  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import BPS_SHA256, WindowFrame, project_to_so3  # noqa: E402
from tools.diagnose_hoi_d2q import (  # noqa: E402
    _rest_batch,
    _sequence_name,
    author_blob_hashes,
    exclusive_json,
    git_output,
    rest_mesh_contract,
    sha256_file,
    sha256_tensor_state,
)
from tools.diagnose_hoi_remediation import seed_everything, stable_seed  # noqa: E402
from tools.evaluate_hoi_remediation import (  # noqa: E402
    current_bps,
    global_goals,
    load_rest_vertices,
    stack_frames,
)


EXPECTED_DATA_CONTRACT_SHA256 = (
    "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
)
DEFAULT_BATCH_SIZE = 8


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": RUN_SUBPHASES[args.run_id],
        "mode": "next-denoiser-local-response-frontier",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(Path(sys.executable).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": args.checkpoint_sha256,
            "weight_variant": "online",
        },
        "selection": {
            "partition": "internal_validation",
            "phase_offsets": list(PHASE_OFFSETS),
            "prior_rollout_offsets": list(PRIOR_ROLLOUT_OFFSETS),
            "ordering": (
                "SHA256(42:d2s-denoiser-response:sequence_name:21,63,105), "
                "sequence_name, sequence_id"
            ),
            "sequences": 64,
            "windows_per_sequence": WINDOWS_PER_SEQUENCE,
            "windows": 192,
            "global_window_indices_sha256": SELECTION_SHA256,
        },
        "probe": {
            "trajectory": "unguided production DDPM; candidates never written back",
            "target_timesteps": list(TARGET_TIMESTEPS),
            "parent_timesteps": list(PARENT_TIMESTEPS),
            "directions": list(DIRECTIONS),
            "scales": list(SCALES),
            "primary_direction": "upper_raw",
            "upper_rotation_joints": list(UPPER_ROTATION_JOINTS),
            "protected_groups": list(PROTECTED_GROUPS),
            "trust_ratio": TRUST_RATIO,
            "fixed_mask_source": "baseline-next clean contact channels 0/1 > 0.95",
            "selection_order": "largest eligible nonzero scale else zero",
            "selection_uses_gt": False,
            "candidate_consumes_sampler_rng": False,
            "author_hand_weight": AUTHOR_HAND_WEIGHT,
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "spatial_threshold_m": SPATIAL_THRESHOLD_M,
            "rest_vertex_count": REST_VERTEX_COUNT,
            "posterior_helper": "priors.diffusion.GaussianDiffusion.posterior_sample",
        },
        "author_parity": {
            "commit": AUTHOR_COMMIT,
            "blob_sha256": author_blob_hashes(),
            "retained": [
                "fk_24_joint_palms_22_23",
                "semantic_mask_gt_0.95_detached",
                "spatial_hinge_0.02m",
                "object_com_and_rotation_temporal_detach",
                "contact_pair_temporal_cosine",
                "batch_size_multiplier",
                "outer_hand_object_weight_10",
            ],
            "registered_response_deviations": [
                "upper_raw_state_projection",
                "fixed_baseline_next_semantic_mask_for_scale_selection",
                "counterfactual_next_denoiser_calls_not_written_to_trajectory",
                "deterministic_2048_vertex_surface",
            ],
            "other_deviations": [
                "feet_floor_weight_500_omitted",
                "scene_and_penetration_terms_omitted",
                "codec_differentiable_so3_decode",
                "ddpm_500_step_checkpoint_instead_of_consistency_sampler",
            ],
        },
        "evaluation": {
            "physical_thresholds_cm": list(PHYSICAL_THRESHOLDS_CM),
            "units": ["left_hand", "right_hand", "union"],
            "field_metrics": list(FIELD_METRICS),
            "physical_metrics": list(PHYSICAL_METRICS),
            "contact_surface": "same deterministic 2048 vertices for all paired candidates",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_unit": "sequence",
        },
        "gate": {
            "target_timesteps": list(GATE_TIMESTEPS),
            "minimum_passing_timesteps": 4,
            "nonzero_selected_fraction_min": 0.50,
            "fixed_mask_loss_improvement_ci_lower_gt": 0.0,
            "fk_union_5cm_recall_and_f1_ci_lower_gt": 0.0,
            "ratio_metrics": list(GATE_RATIO_METRICS),
            "ratio_ci_upper_max": 1.05,
        },
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            "normalization": {
                "path": str((REPO / "data/train/norm.npy").resolve()),
                "sha256": EXPECTED_NORMALIZATION_SHA256,
            },
            "bps": {
                "path": str((REPO / "code/bps.pt").resolve()),
                "sha256": BPS_SHA256,
            },
            "rest_meshes": rest_mesh_contract(),
        },
        "sampler_contract": {
            "production_default_changed": False,
            "future_gt": False,
            "stored_per_frame_bps": False,
            "rollout_bps": "recomputed_from_current window reference",
            "cfg": False,
            "support_clamp": False,
            "reverse_so3_projection": False,
        },
        "released_checkpoint_loaded": False,
        "ema_used": False,
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selection": False,
        "full_trajectory_controller_authorized": False,
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
        "official_test_used": False,
        "chois_used": False,
        "output": str(args.output.resolve()),
    }


def _repeat_frame(frame: WindowFrame, repeats: int) -> WindowFrame:
    return WindowFrame(
        frame.origin.repeat(repeats, 1),
        frame.world_to_local.repeat(repeats, 1, 1),
        frame.object_reference.repeat(repeats, 1, 1),
    )


def _foot_sliding_per_sample(fk_joints: torch.Tensor) -> torch.Tensor:
    joints = fk_joints[:, REPRESENTATION.history_frames:].clone()
    floor = torch.minimum(
        joints[:, :, 10, 1].amin(dim=1),
        joints[:, :, 11, 1].amin(dim=1),
    )
    joints[..., 1] -= floor[:, None, None]
    terms = []
    for joint, height in ((7, 0.08), (10, 0.04), (8, 0.08), (11, 0.04)):
        displacement = torch.linalg.vector_norm(
            joints[:, 1:, joint][:, :, (0, 2)]
            - joints[:, :-1, joint][:, :, (0, 2)],
            dim=-1,
        )
        y = joints[:, :-1, joint, 1]
        active = y < height
        weighted = (
            displacement * (2.0 - torch.pow(2.0, y / height))
        ).abs() * active
        terms.append(weighted.sum(dim=1) / joints.shape[1] * 100.0)
    return torch.stack(terms).mean(dim=0)


def reference_metrics(
    dataset: PriorWindowDataset,
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame: WindowFrame,
    rest_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_surface: torch.Tensor,
    pelvis_goal: torch.Tensor,
    object_goal: torch.Tensor,
) -> List[Dict[str, object]]:
    """Compute GT reference metrics after controller scale selection."""
    decoded = dataset.codec.decode(prediction, frame)
    truth = dataset.codec.decode(target, frame)
    predicted_fk = decoded_fk_positions(decoded, rest_offsets, parents_24)
    target_fk = decoded_fk_positions(truth, rest_offsets, parents_24)
    predicted_vertices = transformed_object_vertices(
        rest_surface, decoded["object_rotation"], decoded["object_translation"],
    )
    target_vertices = transformed_object_vertices(
        rest_surface, truth["object_rotation"], truth["object_translation"],
    )
    predicted_fk_distance = _batched_hand_distances(
        predicted_fk, predicted_vertices, FK_PALM_INDICES,
    )
    target_fk_distance = _batched_hand_distances(
        target_fk, target_vertices, FK_PALM_INDICES,
    )
    predicted_direct_distance = _batched_hand_distances(
        decoded["joints"], predicted_vertices, DIRECT_HAND_INDICES,
    )
    target_direct_distance = _batched_hand_distances(
        truth["joints"], target_vertices, DIRECT_HAND_INDICES,
    )
    field_mse = fieldwise_mse_per_sample(prediction, target)
    active = slice(REPRESENTATION.history_frames, None)
    relative_prediction = predicted_fk[:, active] - predicted_fk[:, active, :1]
    relative_target = target_fk[:, active] - target_fk[:, active, :1]
    physical = {
        "fk_mpjpe_cm": torch.linalg.vector_norm(
            relative_prediction - relative_target, dim=-1,
        ).mean(dim=(1, 2)) * 100.0,
        "pelvis_goal_error_cm": torch.linalg.vector_norm(
            predicted_fk[:, -1, 0][:, (0, 2)] - pelvis_goal[:, (0, 2)], dim=-1,
        ) * 100.0,
        "object_goal_error_cm": torch.linalg.vector_norm(
            decoded["object_translation"][:, -1] - object_goal, dim=-1,
        ) * 100.0,
        "object_translation_mae_cm": torch.linalg.vector_norm(
            decoded["object_translation"][:, active]
            - truth["object_translation"][:, active], dim=-1,
        ).mean(dim=1) * 100.0,
        "object_rotation_geodesic": transforms.so3_relative_angle(
            project_to_so3(decoded["object_rotation"][:, active]).flatten(0, 1),
            project_to_so3(truth["object_rotation"][:, active]).flatten(0, 1),
            cos_bound=1e-7,
        ).reshape(prediction.shape[0], -1).mean(dim=1) * (180.0 / math.pi),
        "fk_foot_sliding": _foot_sliding_per_sample(predicted_fk),
    }
    result = []
    for row in range(prediction.shape[0]):
        result.append({
            **{
                f"{name}_mse": float(values[row].detach().cpu())
                for name, values in field_mse.items()
            },
            **{
                name: float(values[row].detach().cpu())
                for name, values in physical.items()
            },
            "fk_contact": geometry_report(
                predicted_fk_distance[row, active].detach().cpu().numpy(),
                target_fk_distance[row, active].detach().cpu().numpy(),
            ),
            "direct_contact": geometry_report(
                predicted_direct_distance[row, active].detach().cpu().numpy(),
                target_direct_distance[row, active].detach().cpu().numpy(),
            ),
        })
    return result


def _tensor_measurements(value: torch.Tensor) -> Dict[str, object]:
    mutable = value[:, REPRESENTATION.history_frames:]
    rotation = REPRESENTATION.field("joint_rotations_6d")
    upper = torch.zeros(22, dtype=torch.bool, device=value.device)
    upper[list(UPPER_ROTATION_JOINTS)] = True
    reshaped_rotation = mutable[..., rotation.slice].reshape(
        value.shape[0], mutable.shape[1], 22, 6,
    )
    return {
        "norm": torch.linalg.vector_norm(mutable.flatten(1), dim=1),
        "rms": mutable.square().flatten(1).mean(dim=1).sqrt(),
        "max_abs": mutable.abs().flatten(1).amax(dim=1),
        "fields": {
            field.name: mutable[..., field.slice].square().flatten(1).mean(dim=1).sqrt()
            for field in REPRESENTATION.fields
        },
        "upper_rotation_rms": reshaped_rotation[:, :, upper].square().flatten(1).mean(dim=1).sqrt(),
        "non_upper_rotation_rms": reshaped_rotation[:, :, ~upper].square().flatten(1).mean(dim=1).sqrt(),
    }


def _row_measurements(value: Mapping[str, object], row: int) -> Dict[str, object]:
    return {
        "norm": float(value["norm"][row].detach().cpu()),
        "rms": float(value["rms"][row].detach().cpu()),
        "max_abs": float(value["max_abs"][row].detach().cpu()),
        "fields": {
            name: float(field[row].detach().cpu())
            for name, field in value["fields"].items()
        },
        "upper_rotation_rms": float(value["upper_rotation_rms"][row].detach().cpu()),
        "non_upper_rotation_rms": float(value["non_upper_rotation_rms"][row].detach().cpu()),
    }


def _batched_hand_distances(
    joints: torch.Tensor, vertices: torch.Tensor, indices: Sequence[int],
) -> torch.Tensor:
    if joints.ndim != 4 or vertices.ndim != 4 or joints.shape[:2] != vertices.shape[:2]:
        raise ValueError("D2-S batched hand-distance shapes differ")
    batch, frames = joints.shape[:2]
    return torch.cdist(
        joints[:, :, indices].reshape(batch * frames, len(indices), 3),
        vertices.reshape(batch * frames, vertices.shape[2], 3),
    ).amin(dim=-1).reshape(batch, frames, len(indices))


def probe_parent_response(
    model: torch.nn.Module,
    dataset: PriorWindowDataset,
    posterior: torch.Tensor,
    parent_clean: torch.Tensor,
    fixed_history: torch.Tensor,
    text: torch.Tensor,
    bps: torch.Tensor,
    goals: torch.Tensor,
    progress: torch.Tensor,
    target: torch.Tensor,
    frame: WindowFrame,
    rest_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_surface: torch.Tensor,
    pelvis_goal: torch.Tensor,
    object_goal: torch.Tensor,
    names: Sequence[str],
    positions: Sequence[int],
    *,
    parent_timestep: int,
) -> Dict[str, object]:
    """Measure every fixed candidate without writing one into the trajectory."""
    batch = posterior.shape[0]
    target_timestep = parent_timestep - 1
    target_times = torch.full(
        (batch,), target_timestep, dtype=torch.long, device=posterior.device,
    )
    with torch.no_grad():
        baseline_next = model(
            posterior, target_times, text, bps, goals, progress,
        )
        baseline_next = prepare_clean_x0(
            baseline_next, fixed_history, object_so3_x0=False,
        )
    baseline_decoded = dataset.codec.decode(baseline_next, frame)
    baseline_fk = decoded_fk_positions(
        baseline_decoded, rest_offsets, parents_24,
    )
    baseline_vertices = transformed_object_vertices(
        rest_surface,
        baseline_decoded["object_rotation"],
        baseline_decoded["object_translation"],
    )
    fixed_mask = fixed_mask_from_contact(baseline_decoded["contact"])
    baseline_components = author_components_per_sample(
        baseline_fk,
        baseline_vertices,
        baseline_decoded["object_translation"],
        baseline_decoded["object_rotation"],
        fixed_mask,
    )
    aggregate_components = author_hand_object_components(
        baseline_fk,
        baseline_vertices,
        baseline_decoded["object_translation"],
        baseline_decoded["object_rotation"],
        baseline_decoded["contact"],
    )
    author_sum_replay = abs(
        float(baseline_components["raw_total"].sum().detach().cpu())
        - float(aggregate_components["total"].detach().cpu())
    )
    baseline_records = []
    for row in range(batch):
        position = int(positions[row])
        baseline_records.append({
            "sequence": str(names[row]),
            "position": position,
            "global_index": int(dataset.indices[position]),
            "pi": int(dataset.language["pi"][int(dataset.indices[position])]),
            "fixed_mask_author": {
                key: float(value[row].detach().cpu())
                for key, value in baseline_components.items()
            },
        })

    full_gradient, gradient_audit = guidance_gradient(
        parent_clean,
        dataset.codec,
        frame,
        rest_offsets,
        parents_24,
        rest_surface,
    )
    direction_results = {}
    max_history = 0.0
    formula_replay = 0.0
    controller_payload = None
    baseline_reference = None
    # The primary no-GT upper_raw selection is completed before any GT metric.
    for direction in ("upper_raw", "author_all"):
        update = direction_update(full_gradient, direction)
        candidates = scaled_candidate_batch(
            posterior, update, fixed_history,
        )
        repeated = len(SCALES)
        expanded_fixed = fixed_history.repeat(repeated, 1, 1)
        expanded_times = target_times.repeat(repeated)
        with torch.no_grad():
            candidate_next = model(
                candidates,
                expanded_times,
                text.repeat(repeated, 1),
                bps.repeat(repeated, 1, 1),
                goals.repeat(repeated, 1),
                progress.repeat(repeated, 1),
            )
            candidate_next = prepare_clean_x0(
                candidate_next, expanded_fixed, object_so3_x0=False,
            )
        candidate_next_by_scale = unpack_scale_major(candidate_next, batch)
        candidate_states_by_scale = unpack_scale_major(candidates, batch)
        responses = candidate_next_by_scale - baseline_next[None]
        expanded_frame = _repeat_frame(frame, repeated)
        expanded_offsets = rest_offsets.repeat(repeated, 1, 1)
        expanded_surface = rest_surface.repeat(repeated, 1, 1)
        decoded = dataset.codec.decode(candidate_next, expanded_frame)
        candidate_fk = decoded_fk_positions(
            decoded, expanded_offsets, parents_24,
        )
        candidate_vertices = transformed_object_vertices(
            expanded_surface,
            decoded["object_rotation"],
            decoded["object_translation"],
        )
        fixed_mask_expanded = fixed_mask.repeat(repeated, 1, 1)
        candidate_components_flat = author_components_per_sample(
            candidate_fk,
            candidate_vertices,
            decoded["object_translation"],
            decoded["object_rotation"],
            fixed_mask_expanded,
        )
        candidate_components = {
            key: value.reshape(repeated, batch)
            for key, value in candidate_components_flat.items()
        }
        selection = select_largest_eligible_scale(
            baseline_components["effective_total"],
            candidate_components["effective_total"],
            baseline_next - parent_clean,
            responses,
        )

        # GT/reference tensors are first consumed after the no-GT selection above.
        if baseline_reference is None:
            baseline_reference = reference_metrics(
                dataset,
                baseline_next,
                target,
                frame,
                rest_offsets,
                parents_24,
                rest_surface,
                pelvis_goal,
                object_goal,
            )
            for row in range(batch):
                baseline_records[row]["reference"] = baseline_reference[row]
        expanded_target = target.repeat(repeated, 1, 1)
        expanded_pelvis_goal = pelvis_goal.repeat(repeated, 1)
        expanded_object_goal = object_goal.repeat(repeated, 1)
        candidate_reference = reference_metrics(
            dataset,
            candidate_next,
            expanded_target,
            expanded_frame,
            expanded_offsets,
            parents_24,
            expanded_surface,
            expanded_pelvis_goal,
            expanded_object_goal,
        )
        rows = torch.arange(batch, device=posterior.device)
        selected_index = selection["selected_index"]
        selected_next = candidate_next_by_scale[selected_index, rows]
        selected_reference = reference_metrics(
            dataset,
            selected_next,
            target,
            frame,
            rest_offsets,
            parents_24,
            rest_surface,
            pelvis_goal,
            object_goal,
        )
        update_measurements = [
            _tensor_measurements(float(scale) * update)
            for scale in SCALES
        ]
        response_measurements = [
            _tensor_measurements(responses[index])
            for index in range(len(SCALES))
        ]
        scale_one_response = responses[0]
        scale_records = {}
        for scale_index, scale in enumerate(SCALES):
            records = []
            candidate_mask = fixed_mask_from_contact(
                decoded["contact"].reshape(
                    repeated, batch, REPRESENTATION.window_frames, 4,
                )[scale_index]
            )
            for row in range(batch):
                response_norm = float(
                    torch.linalg.vector_norm(
                        responses[scale_index, row, REPRESENTATION.history_frames:],
                    ).detach().cpu()
                )
                input_norm = float(
                    torch.linalg.vector_norm(
                        (float(scale) * update)[row, REPRESENTATION.history_frames:],
                    ).detach().cpu()
                )
                if scale == 0.0:
                    linearity = 0.0
                else:
                    expected = float(scale) * scale_one_response[row]
                    denominator = float(
                        torch.linalg.vector_norm(expected).detach().cpu()
                    )
                    numerator = float(
                        torch.linalg.vector_norm(
                            responses[scale_index, row] - expected,
                        ).detach().cpu()
                    )
                    linearity = numerator / denominator if denominator else 0.0
                records.append({
                    "sequence": str(names[row]),
                    "position": int(positions[row]),
                    "scale": float(scale),
                    "fixed_mask_author": {
                        key: float(value[scale_index, row].detach().cpu())
                        for key, value in candidate_components.items()
                    },
                    "fixed_mask_flip_fraction": float(
                        (candidate_mask[row] != fixed_mask[row]).float().mean().detach().cpu()
                    ),
                    "input_update": _row_measurements(
                        update_measurements[scale_index], row,
                    ),
                    "next_denoiser_response": _row_measurements(
                        response_measurements[scale_index], row,
                    ),
                    "response_over_input_norm": (
                        response_norm / input_norm if input_norm else 0.0
                    ),
                    "local_linearity_relative_l2": linearity,
                    "eligible": bool(selection["eligible"][scale_index, row]),
                    "loss_improved": bool(selection["improved"][scale_index, row]),
                    "protected_group_pass": {
                        name: bool(values[scale_index, row])
                        for name, values in selection["group_passes"].items()
                    },
                    "reference": candidate_reference[scale_index * batch + row],
                })
            scale_records[f"{scale:g}"] = records
        selected_records = []
        for row in range(batch):
            index = int(selected_index[row])
            selected_records.append({
                "sequence": str(names[row]),
                "position": int(positions[row]),
                "selected_scale": float(selection["selected_scale"][row].detach().cpu()),
                "selected_scale_index": index,
                "selected_is_eligible": bool(selection["selected_is_eligible"][row]),
                "largest_eligible_replay": bool(selection["largest_eligible_replay"][row]),
                "baseline_fixed_mask_author_loss": float(
                    baseline_components["effective_total"][row].detach().cpu()
                ),
                "selected_fixed_mask_author_loss": float(
                    selection["selected_loss"][row].detach().cpu()
                ),
                "natural_group_rms": {
                    name: float(value[row].detach().cpu())
                    for name, value in selection["natural_group_rms"].items()
                },
                "selected_group_rms": {
                    name: float(
                        selection["candidate_group_rms"][name][index, row].detach().cpu()
                    )
                    for name in PROTECTED_GROUPS
                },
                "reference": selected_reference[row],
            })
        formula_expected = torch.cat([
            posterior + float(scale) * update for scale in SCALES
        ])
        formula_expected[:, :REPRESENTATION.history_frames] = expanded_fixed
        formula_replay = max(
            formula_replay,
            float((candidates - formula_expected).abs().max().detach().cpu()),
        )
        max_history = max(
            max_history,
            float(
                (candidates[:, :REPRESENTATION.history_frames] - expanded_fixed)
                .abs().max().detach().cpu()
            ),
            float(
                (candidate_next[:, :REPRESENTATION.history_frames] - expanded_fixed)
                .abs().max().detach().cpu()
            ),
        )
        direction_result = {
            "gradient_audit": dict(gradient_audit),
            "scale_records": scale_records,
            "selected_records": selected_records,
            "selection_finite": bool(selection["finite"]),
        }
        direction_results[direction] = direction_result
        if direction == "upper_raw":
            controller_payload = {
                "selected_next": selected_next.detach(),
                "selected_records": selected_records,
            }
    if controller_payload is None:
        raise AssertionError("D2-S primary upper_raw direction is missing")
    return {
        "target_timestep": target_timestep,
        "parent_timestep": parent_timestep,
        "baseline_next": baseline_next.detach(),
        "baseline_records": baseline_records,
        "directions": direction_results,
        "author_per_sample_sum_replay_max_abs": author_sum_replay,
        "candidate_formula_replay_max_abs": formula_replay,
        "history_max_abs": max_history,
        "finite": bool(
            torch.isfinite(baseline_next).all()
            and torch.isfinite(full_gradient).all()
            and all(
                bool(value["selection_finite"])
                for value in direction_results.values()
            )
        ),
    }


def run_unguided_chunk(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    full_rest_vertices: Mapping[str, torch.Tensor],
    rest_subsets: Mapping[str, torch.Tensor],
    parents_24: torch.Tensor,
    *,
    chunk_index: int,
) -> Dict[str, object]:
    items = [dataset[position] for position in positions]
    names = [_sequence_name(dataset, position) for position in positions]
    frame = stack_frames(items, device)
    target = torch.stack([item["x"] for item in items]).to(device)
    fixed = target[:, :REPRESENTATION.history_frames]
    gt_frame = stack_frames(items, device)
    pelvis_goal, object_goal = global_goals(dataset, items, gt_frame, device)
    goals = torch.zeros(len(items), 9, device=device)
    goals[:, :3] = dataset.codec.pelvis_goal(pelvis_goal, frame)
    goals[:, 6:9] = dataset.codec.object_goal(object_goal, frame)
    text = torch.stack([item["text_embedding"] for item in items]).to(device)
    bps = current_bps(dataset, frame.object_reference, names, full_rest_vertices)
    progress = normalize_progress(
        torch.stack([item["progress"] for item in items]).to(device)
    )
    rest_offsets = torch.stack([
        item["rest_human_offsets"] for item in items
    ]).to(device)
    rest_surface = _rest_batch(names, rest_subsets, device)
    label = f"D2:d2s-shared:chunk:{chunk_index}"
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(label))
    initial_generator_state = sha256_tensor_state(generator.get_state())
    current = torch.randn(
        (len(items), REPRESENTATION.window_frames, REPRESENTATION.dimension),
        device=device,
        generator=generator,
    )
    current[:, :REPRESENTATION.history_frames] = fixed
    probes = {}
    expected_clean = {}
    history_max_abs = 0.0
    posterior_formula_max_abs = 0.0
    baseline_replay_max_abs = 0.0
    probe_rng_unchanged = True
    for step in reversed(range(diffusion.timesteps)):
        timesteps = torch.full(
            (len(items),), step, dtype=torch.long, device=device,
        )
        with torch.no_grad():
            clean = model(current, timesteps, text, bps, goals, progress)
            clean = prepare_clean_x0(clean, fixed, object_so3_x0=False)
        if step in expected_clean:
            baseline_replay_max_abs = max(
                baseline_replay_max_abs,
                float((clean - expected_clean.pop(step)).abs().max().detach().cpu()),
            )
        if step:
            noise = torch.randn(
                current.shape, device=device, generator=generator,
            )
        else:
            noise = torch.zeros_like(current)
        with torch.no_grad():
            posterior = diffusion.posterior_sample(
                current, clean, timesteps, noise, fixed,
            )
        if step in PARENT_TIMESTEPS:
            manual = (
                _extract(diffusion.posterior_mean_coef1, timesteps, current.shape) * clean
                + _extract(diffusion.posterior_mean_coef2, timesteps, current.shape) * current
                + (0.5 * _extract(
                    diffusion.posterior_log_variance, timesteps, current.shape,
                )).exp() * noise
            )
            manual[:, :REPRESENTATION.history_frames] = fixed
            posterior_formula_max_abs = max(
                posterior_formula_max_abs,
                float((manual - posterior).abs().max().detach().cpu()),
            )
            before_probe = sha256_tensor_state(generator.get_state())
            probe = probe_parent_response(
                model,
                dataset,
                posterior,
                clean,
                fixed,
                text,
                bps,
                goals,
                progress,
                target,
                frame,
                rest_offsets,
                parents_24,
                rest_surface,
                pelvis_goal,
                object_goal,
                names,
                positions,
                parent_timestep=step,
            )
            after_probe = sha256_tensor_state(generator.get_state())
            probe_rng_unchanged &= before_probe == after_probe
            expected_clean[step - 1] = probe.pop("baseline_next")
            probes[str(step - 1)] = probe
        current = posterior
        history_max_abs = max(
            history_max_abs,
            float((current[:, :REPRESENTATION.history_frames] - fixed).abs().max().detach().cpu()),
        )
    if expected_clean:
        raise AssertionError(f"D2-S baseline replay cache not consumed: {expected_clean}")
    return {
        "probes": probes,
        "noise_stream": {
            "chunk_index": chunk_index,
            "label": label,
            "seed": stable_seed(label),
            "generator_initial_state_sha256": initial_generator_state,
            "generator_final_state_sha256": sha256_tensor_state(generator.get_state()),
        },
        "history_max_abs": history_max_abs,
        "posterior_formula_replay_max_abs": posterior_formula_max_abs,
        "baseline_next_replay_max_abs": baseline_replay_max_abs,
        "probe_rng_unchanged": probe_rng_unchanged,
        "finite": bool(
            torch.isfinite(current).all()
            and all(bool(value["finite"]) for value in probes.values())
        ),
    }


def _numeric_summary(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-S numeric summary requires finite values")
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _reference_summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    result = {name: _numeric_summary([
        float(record["reference"][name]) for record in records
    ]) for name in (*FIELD_METRICS, *PHYSICAL_METRICS)}
    result["contact"] = {}
    for geometry in ("fk_contact", "direct_contact"):
        result["contact"][geometry] = {}
        for threshold in PHYSICAL_THRESHOLDS_CM:
            key = f"{threshold:g}"
            result["contact"][geometry][key] = {}
            for unit in ("left_hand", "right_hand", "union"):
                result["contact"][geometry][key][unit] = {
                    metric: _numeric_summary([
                        float(record["reference"][geometry]["thresholds_cm"][key][unit][metric])
                        for record in records
                    ])
                    for metric in (
                        "accuracy", "precision", "recall", "f1",
                        "prediction_percent", "target_percent",
                    )
                }
    return result


def _contact_values(
    records: Sequence[Mapping[str, object]], geometry: str, metric: str,
) -> List[float]:
    return [
        float(record["reference"][geometry]["thresholds_cm"]["5"]["union"][metric])
        for record in records
    ]


def controller_comparison(
    baseline: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if [record["sequence"] for record in baseline] != [
        record["sequence"] for record in selected
    ]:
        raise ValueError("D2-S controller records differ in sequence ordering")
    names = [str(record["sequence"]) for record in baseline]
    result = {
        "fixed_mask_author_loss": paired_sequence_difference(
            names,
            [float(record["fixed_mask_author"]["effective_total"]) for record in baseline],
            [float(record["selected_fixed_mask_author_loss"]) for record in selected],
        ),
        "fk_union_5cm": {},
        "ratios": {},
    }
    for metric in ("recall", "f1", "precision", "prediction_percent"):
        result["fk_union_5cm"][metric] = paired_sequence_difference(
            names,
            _contact_values(selected, "fk_contact", metric),
            _contact_values(baseline, "fk_contact", metric),
        )
    for metric in GATE_RATIO_METRICS:
        source_metric = (
            "joint_positions_mse" if metric == "joint_position_mse" else metric
        )
        result["ratios"][metric] = paired_sequence_ratio(
            names,
            [float(record["reference"][source_metric]) for record in selected],
            [float(record["reference"][source_metric]) for record in baseline],
        )
    return result


def summarize_timestep(value: Mapping[str, object]) -> Dict[str, object]:
    baseline = value["baseline_records"]
    directions = {}
    for direction in DIRECTIONS:
        direction_value = value["directions"][direction]
        scale_summaries = {}
        for scale in SCALES:
            key = f"{scale:g}"
            records = direction_value["scale_records"][key]
            scale_summaries[key] = {
                "windows": len(records),
                "fixed_mask_author_effective_total": _numeric_summary([
                    float(record["fixed_mask_author"]["effective_total"])
                    for record in records
                ]),
                "fixed_mask_flip_fraction": _numeric_summary([
                    float(record["fixed_mask_flip_fraction"]) for record in records
                ]),
                "input_update_norm": _numeric_summary([
                    float(record["input_update"]["norm"]) for record in records
                ]),
                "response_norm": _numeric_summary([
                    float(record["next_denoiser_response"]["norm"]) for record in records
                ]),
                "response_over_input_norm": _numeric_summary([
                    float(record["response_over_input_norm"]) for record in records
                ]),
                "local_linearity_relative_l2": _numeric_summary([
                    float(record["local_linearity_relative_l2"]) for record in records
                ]),
                "reference": _reference_summary(records),
            }
        selected = direction_value["selected_records"]
        histogram = {f"{scale:g}": 0 for scale in SCALES}
        for record in selected:
            histogram[f"{float(record['selected_scale']):g}"] += 1
        directions[direction] = {
            "scales": scale_summaries,
            "selected_scale_histogram": histogram,
            "nonzero_selected_fraction": float(np.mean([
                float(record["selected_scale"]) > 0.0 for record in selected
            ])),
            "selected_reference": _reference_summary(selected),
        }
    primary = value["directions"]["upper_raw"]["selected_records"]
    return {
        "baseline_reference": _reference_summary(baseline),
        "directions": directions,
        "nonzero_selected_fraction": directions["upper_raw"]["nonzero_selected_fraction"],
        "controller_comparison": controller_comparison(baseline, primary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id not in RUN_IDS:
        raise ValueError(f"D2-S0 run id must be one of {RUN_IDS}")
    if args.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("D2-S0 checkpoint SHA argument mismatch")
    if args.batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(f"D2-S0 batch size must be {DEFAULT_BATCH_SIZE}")
    config = resolved_config(args)
    config_path = args.resolved_config.resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-S0 runtime arguments do not match archived resolved config")
    if Path(sys.executable).resolve() != Path(
        os.environ.get("INFBAGEL_PYTHON", ""),
    ).resolve():
        raise ValueError("D2-S0 requires the absolute INFBAGEL_PYTHON interpreter")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-S0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if git_output("status", "--porcelain"):
        raise RuntimeError("D2-S0 refuses a dirty worker checkout")
    if sha256_file(args.checkpoint.resolve()) != CHECKPOINT_SHA256:
        raise ValueError("D2-S0 checkpoint hash mismatch")
    asset_hashes = {
        "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
        "bps": sha256_file((REPO / "code/bps.pt").resolve()),
    }
    if asset_hashes != {
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "bps": BPS_SHA256,
    }:
        raise ValueError(f"D2-S0 asset hash mismatch: {asset_hashes}")
    if author_blob_hashes() != AUTHOR_BLOB_SHA256:
        raise ValueError("D2-S0 author blob hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-S0 is a four-GPU-worker CUDA diagnostic")
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output.resolve()}")

    seed_everything(42)
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    selection = select_response_holdout(dataset)
    triples = selection["triples"]
    positions = [position for triple in triples for position in triple]
    parents_24 = torch.from_numpy(
        get_smpl_parents(use_joints24=True).copy(),
    ).long().to(device)
    full_rest_vertices = load_rest_vertices(dataset, triples, device)
    rest_subsets = {
        name: deterministic_vertex_subset(vertices)
        for name, vertices in full_rest_vertices.items()
    }
    diffusion = GaussianDiffusion(500).to(device)
    model, metadata = load_trained_hoi_prior(
        str(args.checkpoint.resolve()), device, weight_variant="online",
    )
    if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
        raise ValueError("D2-S0 checkpoint data-contract mismatch")
    model.eval()
    model_before = state_dict_sha256(model)
    raw = {
        str(timestep): {
            "baseline_records": [],
            "directions": {
                direction: {
                    "scale_records": {f"{scale:g}": [] for scale in SCALES},
                    "selected_records": [],
                }
                for direction in DIRECTIONS
            },
        }
        for timestep in TARGET_TIMESTEPS
    }
    noise_streams = []
    history_max_abs = 0.0
    posterior_formula_max_abs = 0.0
    baseline_replay_max_abs = 0.0
    author_sum_replay_max_abs = 0.0
    candidate_formula_max_abs = 0.0
    all_finite = True
    probe_rng_unchanged = True
    torch.cuda.reset_peak_memory_stats(device)
    for chunk_index, offset in enumerate(range(0, len(positions), args.batch_size)):
        selected_positions = positions[offset:offset + args.batch_size]
        chunk = run_unguided_chunk(
            model,
            diffusion,
            dataset,
            selected_positions,
            device,
            full_rest_vertices,
            rest_subsets,
            parents_24,
            chunk_index=chunk_index,
        )
        noise_streams.append(chunk["noise_stream"])
        history_max_abs = max(history_max_abs, float(chunk["history_max_abs"]))
        posterior_formula_max_abs = max(
            posterior_formula_max_abs,
            float(chunk["posterior_formula_replay_max_abs"]),
        )
        baseline_replay_max_abs = max(
            baseline_replay_max_abs,
            float(chunk["baseline_next_replay_max_abs"]),
        )
        probe_rng_unchanged &= bool(chunk["probe_rng_unchanged"])
        all_finite &= bool(chunk["finite"])
        for timestep, probe in chunk["probes"].items():
            target_raw = raw[timestep]
            target_raw["baseline_records"].extend(probe["baseline_records"])
            author_sum_replay_max_abs = max(
                author_sum_replay_max_abs,
                float(probe["author_per_sample_sum_replay_max_abs"]),
            )
            candidate_formula_max_abs = max(
                candidate_formula_max_abs,
                float(probe["candidate_formula_replay_max_abs"]),
            )
            for direction in DIRECTIONS:
                source = probe["directions"][direction]
                destination = target_raw["directions"][direction]
                for scale in SCALES:
                    key = f"{scale:g}"
                    destination["scale_records"][key].extend(
                        source["scale_records"][key]
                    )
                destination["selected_records"].extend(source["selected_records"])
        print(
            json.dumps({
                "chunk": chunk_index + 1,
                "chunks": math.ceil(len(positions) / args.batch_size),
                "windows": min(offset + args.batch_size, len(positions)),
                "elapsed_seconds": time.time() - started,
            }),
            flush=True,
        )

    timestep_summaries = {
        timestep: summarize_timestep(value)
        for timestep, value in raw.items()
    }
    model_after = state_dict_sha256(model)
    parameter_grad_buffers_clear = all(
        parameter.grad is None for parameter in model.parameters()
    )
    helper_source = inspect.getsource(run_unguided_chunk)
    probe_source = inspect.getsource(probe_parent_response)
    production_source = inspect.getsource(GaussianDiffusion.sample)
    selection_source = inspect.getsource(select_largest_eligible_scale)
    expected_windows = 64 * WINDOWS_PER_SEQUENCE
    records_complete = all(
        len(raw[str(timestep)]["baseline_records"]) == expected_windows
        and all(
            len(raw[str(timestep)]["directions"][direction]["selected_records"])
            == expected_windows
            and all(
                len(raw[str(timestep)]["directions"][direction]["scale_records"][f"{scale:g}"])
                == expected_windows
                for scale in SCALES
            )
            for direction in DIRECTIONS
        )
        for timestep in TARGET_TIMESTEPS
    )
    largest_replay = all(
        bool(record["largest_eligible_replay"])
        and bool(record["selected_is_eligible"])
        for timestep in TARGET_TIMESTEPS
        for direction in DIRECTIONS
        for record in raw[str(timestep)]["directions"][direction]["selected_records"]
    )
    contract = {
        "checkpoint_hash_exact": True,
        "data_contract_exact": metadata["data_contract_sha256"] == EXPECTED_DATA_CONTRACT_SHA256,
        "asset_hashes_exact": asset_hashes == {
            "normalization": EXPECTED_NORMALIZATION_SHA256, "bps": BPS_SHA256,
        },
        "author_blob_hashes_exact": author_blob_hashes() == AUTHOR_BLOB_SHA256,
        "selection_exact": selection["sha256"] == SELECTION_SHA256,
        "selection_disjoint_from_prior_rollout_offsets": not bool(
            set(PHASE_OFFSETS) & set(PRIOR_ROLLOUT_OFFSETS)
        ),
        "target_parent_boundaries_exact": tuple(value + 1 for value in TARGET_TIMESTEPS) == PARENT_TIMESTEPS,
        "all_directions_scales_timesteps_fields_reported": records_complete,
        "all_finite": bool(all_finite),
        "history_max_abs_le_1e-5": history_max_abs <= HISTORY_MAX_ABS,
        "posterior_formula_replay_max_abs_le_1e-5": posterior_formula_max_abs <= 1e-5,
        "baseline_next_replay_max_abs_le_1e-7": baseline_replay_max_abs <= 1e-7,
        "author_per_sample_sum_replay_max_abs_le_1e-5": author_sum_replay_max_abs <= 1e-5,
        "candidate_formula_replay_max_abs_le_1e-7": candidate_formula_max_abs <= 1e-7,
        "candidate_probe_rng_unchanged": probe_rng_unchanged,
        "largest_eligible_scale_replay": largest_replay,
        "model_state_unchanged": model_before == model_after,
        "parameter_grad_buffers_clear": parameter_grad_buffers_clear,
        "posterior_helper_reused": "diffusion.posterior_sample(" in helper_source,
        "selection_uses_no_gt": (
            "target" not in inspect.signature(select_largest_eligible_scale).parameters
            and "target" not in selection_source
            and "gt" not in selection_source.lower()
            and probe_source.index("select_largest_eligible_scale(")
            < probe_source.index("expanded_target =")
        ),
        "fixed_baseline_mask_used": "fixed_mask_expanded" in probe_source,
        "candidates_not_written_to_trajectory": "current = posterior" in helper_source,
        "production_sampler_default_unchanged": "guidance" not in production_source,
        "sampler_future_gt_absent": (
            "future" not in production_source.lower()
            and "target" not in production_source.lower()
            and "dataset" not in production_source.lower()
            and "target" not in selection_source.lower()
        ),
        "sampler_stored_per_frame_bps_absent": (
            'item["object_bps"]' not in helper_source and "current_bps(" in helper_source
        ),
    }
    decision = mechanism_gate(contract, timestep_summaries)
    runtime = time.time() - started
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": RUN_SUBPHASES[args.run_id],
        "status": "completed",
        "git_commit": git_output("rev-parse", "HEAD"),
        "seed": 42,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": CHECKPOINT_SHA256,
            "metadata": metadata,
            "model_state_sha256_before": model_before,
            "model_state_sha256_after": model_after,
            "model_state_unchanged": model_before == model_after,
            "parameter_grad_buffers_clear": parameter_grad_buffers_clear,
        },
        "selection": {
            key: value for key, value in selection.items() if key != "triples"
        },
        "probe": {
            "target_timesteps": list(TARGET_TIMESTEPS),
            "parent_timesteps": list(PARENT_TIMESTEPS),
            "directions": list(DIRECTIONS),
            "scales": list(SCALES),
            "trust_ratio": TRUST_RATIO,
            "protected_groups": list(PROTECTED_GROUPS),
            "noise_streams": noise_streams,
        },
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            "normalization": asset_hashes["normalization"],
            "bps": asset_hashes["bps"],
            "rest_meshes": rest_mesh_contract(),
            "author_blob_sha256": author_blob_hashes(),
        },
        "contract": contract,
        "history_max_abs": history_max_abs,
        "posterior_formula_replay_max_abs": posterior_formula_max_abs,
        "baseline_next_replay_max_abs": baseline_replay_max_abs,
        "author_per_sample_sum_replay_max_abs": author_sum_replay_max_abs,
        "candidate_formula_replay_max_abs": candidate_formula_max_abs,
        "timesteps": {
            timestep: {**timestep_summaries[timestep], "raw": raw[timestep]}
            for timestep in raw
        },
        "decision": decision,
        "runtime_seconds": runtime,
        "gpu": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "sampler_contract": config["sampler_contract"],
        "released_checkpoint_used": False,
        "ema_used": False,
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selection": False,
        "full_trajectory_controller_authorized": False,
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
        "official_test_used": False,
        "chois_used": False,
    }
    exclusive_json(args.output.resolve(), result)
    print(json.dumps({
        "run_id": args.run_id,
        "classification": decision["classification"],
        "passing_timesteps": decision["passing_timesteps"],
        "contract_passed": decision["contract_passed"],
        "runtime_seconds": runtime,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
