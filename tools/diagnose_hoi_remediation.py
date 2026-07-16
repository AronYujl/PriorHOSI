#!/usr/bin/env python3
"""Run the preregistered Phase 1B D0 diagnostic on internal validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import trimesh
from pytorch3d import transforms
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from datasets.utils import zup_to_yup  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.models import build_expert  # noqa: E402
from priors.remediation import (  # noqa: E402
    D0_TIMESTEPS,
    deterministic_derangement,
    field_squared_error,
    select_internal_triples,
    select_teacher_windows,
    selection_sha256,
)
from priors.representation import REPRESENTATION  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = "e50d5e7f3081d7740e9df3658883013ca509f7f313667cc3ddbec418de582583"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def frame_contract(dataset: PriorWindowDataset, position: int, device: torch.device):
    global_index = int(dataset.indices[position])
    start = int(dataset.starts[global_index])
    sequence = int(dataset.sequence_ids[global_index])
    first_pelvis = np.asarray(dataset.joints[start, 0], dtype=np.float32)
    origin = np.asarray((first_pelvis[0], 0.0, first_pelvis[2]), dtype=np.float32)
    oriented = zup_to_yup(np.asarray(dataset.orient[start], dtype=np.float64).copy())
    yaw = Rotation.from_rotvec(oriented).as_euler("zxy")[2]
    world_to_local = Rotation.from_euler("zxy", (0.0, 0.0, -yaw)).as_matrix().astype(np.float32)
    reference = np.array(dataset.object_rot[start], dtype=np.float32, copy=True)
    return (
        torch.from_numpy(origin).to(device),
        torch.from_numpy(world_to_local).to(device),
        torch.from_numpy(reference).to(device),
        global_index,
        sequence,
    )


def decode_window(
    dataset: PriorWindowDataset, position: int, encoded: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = encoded.device
    origin, world_to_local, reference, global_index, sequence = frame_contract(dataset, position, device)
    joint_min = torch.from_numpy(dataset.minimum).to(device)
    joint_max = torch.from_numpy(dataset.maximum).to(device)
    object_min = torch.from_numpy(dataset.object_minimum).to(device)
    object_max = torch.from_numpy(dataset.object_maximum).to(device)
    local_joints = (encoded[..., :84].reshape(*encoded.shape[:-1], 28, 3) + 1.0) * 0.5
    local_joints = local_joints * (joint_max - joint_min) + joint_min
    joints = local_joints @ world_to_local + origin
    local_human_rotation = transforms.rotation_6d_to_matrix(
        encoded[..., 84:216].reshape(*encoded.shape[:-1], 22, 6)
    )
    human_rotation = world_to_local.transpose(-1, -2) @ local_human_rotation
    local_object = (encoded[..., 216:219] + 1.0) * 0.5
    local_object = local_object * (object_max - object_min) + object_min
    object_translation = local_object @ world_to_local + origin
    relative_rotation = encoded[..., 219:228].reshape(*encoded.shape[:-1], 3, 3)
    object_rotation = relative_rotation @ reference
    return {
        "joints": joints,
        "human_rotation": human_rotation,
        "object_translation": object_translation,
        "object_rotation": object_rotation,
        "contact": encoded[..., 228:232],
        "origin": origin,
        "world_to_local": world_to_local,
        "reference": reference,
        "global_index": torch.tensor(global_index, device=device),
        "sequence": torch.tensor(sequence, device=device),
    }


def raw_window_target(dataset: PriorWindowDataset, position: int, device: torch.device) -> Dict[str, torch.Tensor]:
    global_index = int(dataset.indices[position])
    start, end = int(dataset.starts[global_index]), int(dataset.ends[global_index])
    frames = np.arange(start, end, 3)
    sequence = int(dataset.sequence_ids[global_index])
    name = str(dataset.scene_names[sequence])
    offset = start - int(dataset.seq_starts[sequence])
    return {
        "joints": torch.from_numpy(np.asarray(dataset.joints[frames], dtype=np.float32)).to(device),
        "object_translation": torch.from_numpy(np.asarray(dataset.object_trans[frames], dtype=np.float32)).to(device),
        "object_rotation": torch.from_numpy(np.asarray(dataset.object_rot[frames], dtype=np.float32)).to(device),
        "contact": torch.from_numpy(
            np.asarray(dataset._contact(name)[offset:offset + 48:3], dtype=np.float32)
        ).to(device),
    }


def goal_globals(dataset: PriorWindowDataset, position: int, item: Mapping[str, torch.Tensor], device):
    decoded = decode_window(dataset, position, item["x"].to(device))
    object_min = torch.from_numpy(dataset.object_minimum).to(device)
    object_max = torch.from_numpy(dataset.object_maximum).to(device)
    local_goal = (item["goals"][6:9].to(device) + 1.0) * 0.5
    local_goal = local_goal * (object_max - object_min) + object_min
    object_goal = local_goal @ decoded["world_to_local"] + decoded["origin"]
    global_index = int(dataset.indices[position])
    end = int(dataset.ends[global_index])
    pelvis_goal = torch.from_numpy(
        np.array(dataset.joints[end - 1, 0], dtype=np.float32, copy=True)
    ).to(device)
    return pelvis_goal, object_goal


def generated_handoff(
    dataset: PriorWindowDataset,
    sample: torch.Tensor,
    decoded: Dict[str, torch.Tensor],
    next_items: Sequence[Mapping[str, torch.Tensor]],
    next_positions: Sequence[int],
    first_references: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Reproduce the failed evaluator's generated-history/first-reference handoff."""
    device = sample.device
    batch = sample.shape[0]
    joints = decoded["joints"][:, -2:]
    human_rotation = decoded["human_rotation"][:, -2:]
    object_translation = decoded["object_translation"][:, -2:]
    origin = joints[:, 0, 0].clone()
    origin[:, 1] = 0.0
    root_matrices = human_rotation[:, 0, 0].detach().cpu().numpy()
    yaws = Rotation.from_matrix(root_matrices).as_euler("zxy")[:, 2]
    values = np.zeros((batch, 3), dtype=np.float64)
    values[:, 2] = -yaws
    world_to_local = torch.from_numpy(Rotation.from_euler("zxy", values).as_matrix()).to(
        device=device, dtype=sample.dtype,
    )
    joint_min = torch.from_numpy(dataset.minimum).to(device)
    joint_max = torch.from_numpy(dataset.maximum).to(device)
    object_min = torch.from_numpy(dataset.object_minimum).to(device)
    object_max = torch.from_numpy(dataset.object_maximum).to(device)
    local_joints = (joints - origin[:, None, None]) @ world_to_local.transpose(-1, -2)[:, None]
    local_joints = -1.0 + 2.0 * (local_joints - joint_min) / (joint_max - joint_min)
    local_human_rotation = world_to_local[:, None, None] @ human_rotation
    local_object = (object_translation - origin[:, None]) @ world_to_local.transpose(-1, -2)
    local_object = -1.0 + 2.0 * (local_object - object_min) / (object_max - object_min)
    fixed = torch.zeros(batch, 2, REPRESENTATION.dimension, device=device, dtype=sample.dtype)
    fixed[..., :84] = local_joints.reshape(batch, 2, 84)
    fixed[..., 84:216] = transforms.matrix_to_rotation_6d(local_human_rotation).reshape(batch, 2, 132)
    fixed[..., 216:219] = local_object
    # The failed path kept every object's first-window reference.
    fixed[..., 219:228] = sample[:, -2:, 219:228]
    fixed[..., 228:232] = sample[:, -2:, 228:232]
    goals = torch.zeros(batch, 9, device=device, dtype=sample.dtype)
    for row, (item, position) in enumerate(zip(next_items, next_positions)):
        _, object_goal = goal_globals(dataset, position, item, device)
        local_goal = (object_goal - origin[row]) @ world_to_local[row].transpose(-1, -2)
        goals[row, 6:9] = -1.0 + 2.0 * (local_goal - object_min) / (object_max - object_min)
    return {
        "fixed": fixed,
        "goals": goals,
        "origin": origin,
        "world_to_local": world_to_local,
        "reference": first_references,
    }


def decode_generated_batch(
    dataset: PriorWindowDataset,
    sample: torch.Tensor,
    origin: torch.Tensor,
    world_to_local: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = sample.device
    joint_min = torch.from_numpy(dataset.minimum).to(device)
    joint_max = torch.from_numpy(dataset.maximum).to(device)
    object_min = torch.from_numpy(dataset.object_minimum).to(device)
    object_max = torch.from_numpy(dataset.object_maximum).to(device)
    local_joints = (sample[..., :84].reshape(sample.shape[0], 16, 28, 3) + 1.0) * 0.5
    local_joints = local_joints * (joint_max - joint_min) + joint_min
    joints = torch.einsum("btjc,bcd->btjd", local_joints, world_to_local) + origin[:, None, None]
    local_human_rotation = transforms.rotation_6d_to_matrix(sample[..., 84:216].reshape(-1, 22, 6))
    local_human_rotation = local_human_rotation.reshape(sample.shape[0], 16, 22, 3, 3)
    human_rotation = world_to_local.transpose(-1, -2)[:, None, None] @ local_human_rotation
    local_object = (sample[..., 216:219] + 1.0) * 0.5
    local_object = local_object * (object_max - object_min) + object_min
    object_translation = torch.einsum("btc,bcd->btd", local_object, world_to_local) + origin[:, None]
    relative_rotation = sample[..., 219:228].reshape(sample.shape[0], 16, 3, 3)
    object_rotation = relative_rotation @ reference[:, None]
    return {
        "joints": joints,
        "human_rotation": human_rotation,
        "object_translation": object_translation,
        "object_rotation": object_rotation,
        "contact": sample[..., 228:232],
    }


def stack_items(dataset, positions: Sequence[int], device: torch.device) -> Dict[str, torch.Tensor]:
    items = [dataset[position] for position in positions]
    keys = ("x", "text_embedding", "object_bps", "goals", "progress")
    return {key: torch.stack([item[key] for item in items]).to(device) for key in keys}


def teacher_diagnostic(model, diffusion, dataset, positions, device, weight_name, batch_size):
    result: Dict[str, object] = {}
    for timestep in D0_TIMESTEPS:
        totals = {field.name: 0.0 for field in REPRESENTATION.fields}
        sensitivity = {
            name: {field.name: 0.0 for field in REPRESENTATION.fields}
            for name in ("text", "bps", "object_goal")
        }
        seen = 0
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset:offset + batch_size]
            batch = stack_items(dataset, batch_positions, device)
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_seed(f"D0:{weight_name}:teacher:{timestep}:{offset}"))
            noise = torch.randn(batch["x"].shape, device=device, generator=generator)
            times = torch.full((len(batch_positions),), timestep, device=device, dtype=torch.long)
            noisy = diffusion.q_sample(batch["x"], times, noise)
            progress = normalize_progress(batch["progress"])
            matched = model(
                noisy, times, batch["text_embedding"], batch["object_bps"], batch["goals"], progress,
            )
            errors = field_squared_error(matched, batch["x"])
            for name, value in errors.items():
                totals[name] += float(value) * len(batch_positions)
            permutation = deterministic_derangement(len(batch_positions), device=device)
            variants = {
                "text": (batch["text_embedding"][permutation], batch["object_bps"], batch["goals"]),
                "bps": (batch["text_embedding"], batch["object_bps"][permutation], batch["goals"]),
                "object_goal": (
                    batch["text_embedding"], batch["object_bps"],
                    torch.cat((batch["goals"][:, :6], batch["goals"][permutation, 6:9]), dim=-1),
                ),
            }
            for condition, (text, bps, goals) in variants.items():
                prediction = model(noisy, times, text, bps, goals, progress)
                permuted_error = field_squared_error(prediction, batch["x"])
                for name, value in permuted_error.items():
                    sensitivity[condition][name] += float(value) * len(batch_positions)
            seen += len(batch_positions)
        matched_values = {name: value / seen for name, value in totals.items()}
        result[str(timestep)] = {
            "matched_fieldwise_mse": matched_values,
            "permutation": {
                condition: {
                    "fieldwise_mse": {name: value / seen for name, value in values.items()},
                    "delta_total_field_mean": float(np.mean([
                        values[name] / seen - matched_values[name] for name in values
                    ])),
                }
                for condition, values in sensitivity.items()
            },
        }
    return result


def project_so3(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix)
    candidate = u @ vh
    determinant = torch.det(candidate)
    correction = torch.eye(3, device=matrix.device, dtype=matrix.dtype).expand_as(matrix).clone()
    correction[..., 2, 2] = torch.where(determinant < 0, -1.0, 1.0)
    return u @ correction @ vh


def object_vertices(rest: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return rest[None] @ project_so3(rotation).transpose(-1, -2) + translation[:, None]


def physical_summary(dataset, triples, decoded_steps, device):
    per_window = []
    all_joint_relative_errors: List[torch.Tensor] = []
    object_goal_errors = []
    pelvis_goal_errors = []
    contact_counts = np.zeros(4, dtype=np.int64)  # tp/fp/tn/fn
    foot_sliding = []
    per_sequence = []
    rest_cache: Dict[str, torch.Tensor] = {}
    for row, triple in enumerate(triples):
        pred_joints = []
        gt_joints = []
        pred_objects = []
        gt_objects = []
        pred_rotations = []
        gt_rotations = []
        for step, position in enumerate(triple):
            prediction = {key: value[row, 2:] for key, value in decoded_steps[step].items() if torch.is_tensor(value)}
            target = raw_window_target(dataset, position, device)
            target = {key: value[2:] for key, value in target.items()}
            pred_joints.append(prediction["joints"])
            gt_joints.append(target["joints"])
            pred_objects.append(prediction["object_translation"])
            gt_objects.append(target["object_translation"])
            pred_rotations.append(prediction["object_rotation"])
            gt_rotations.append(target["object_rotation"])
            relative_pred = prediction["joints"] - prediction["joints"][:, :1]
            relative_gt = target["joints"] - target["joints"][:, :1]
            mpjpe = torch.linalg.vector_norm(relative_pred - relative_gt, dim=-1).mean() * 100.0
            pelvis_goal, object_goal = goal_globals(dataset, position, dataset[position], device)
            pelvis_error = torch.linalg.vector_norm(
                prediction["joints"][-1, 0][[0, 2]] - pelvis_goal[[0, 2]]
            ) * 100.0
            object_error = torch.linalg.vector_norm(prediction["object_translation"][-1] - object_goal) * 100.0
            object_rotation_error = transforms.so3_relative_angle(
                project_so3(prediction["object_rotation"]),
                project_so3(target["object_rotation"]),
                cos_bound=1e-7,
            ).mean() * (180.0 / math.pi)
            pelvis_goal_errors.append(float(pelvis_error))
            object_goal_errors.append(float(object_error))
            per_window.append({
                "sequence": str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])]),
                "window": step + 1,
                "pi": int(dataset.language["pi"][int(dataset.indices[position])]),
                "mpjpe_cm": float(mpjpe),
                "pelvis_goal_error_cm": float(pelvis_error),
                "object_goal_error_cm": float(object_error),
                "joint_position_mae_cm": float(
                    torch.linalg.vector_norm(prediction["joints"] - target["joints"], dim=-1).mean() * 100.0
                ),
                "pelvis_translation_mae_cm": float(
                    torch.linalg.vector_norm(
                        prediction["joints"][:, 0] - target["joints"][:, 0], dim=-1,
                    ).mean() * 100.0
                ),
                "object_translation_mae_cm": float(
                    torch.linalg.vector_norm(
                        prediction["object_translation"] - target["object_translation"], dim=-1,
                    ).mean() * 100.0
                ),
                "object_rotation_geodesic_deg": float(object_rotation_error),
                "contact_channel_mse": float((prediction["contact"] - target["contact"]).square().mean()),
            })
            all_joint_relative_errors.append(torch.linalg.vector_norm(relative_pred - relative_gt, dim=-1))
        predicted_joints = torch.cat(pred_joints)
        target_joints = torch.cat(gt_joints)
        predicted_objects = torch.cat(pred_objects)
        target_objects = torch.cat(gt_objects)
        predicted_rotations = torch.cat(pred_rotations)
        target_rotations = torch.cat(gt_rotations)
        name = per_window[-1]["sequence"]
        object_name = name.split("_")[1]
        if object_name not in rest_cache:
            mesh = trimesh.load(REPO / "data/object/rest_object_geo" / f"{object_name}.ply", process=False)
            vertices = zup_to_yup(np.asarray(mesh.vertices, dtype=np.float32).copy())
            # Fixed deterministic subset bounds the internal diagnostic cost.
            indices = np.linspace(0, len(vertices) - 1, min(2048, len(vertices))).round().astype(np.int64)
            rest_cache[object_name] = torch.from_numpy(vertices[indices]).to(device)
        rest = rest_cache[object_name]
        predicted_vertices = object_vertices(rest, predicted_rotations, predicted_objects)
        target_vertices = object_vertices(rest, target_rotations, target_objects)
        pred_hands = predicted_joints[:, (24, 26)]
        gt_hands = target_joints[:, (24, 26)]
        pred_contact = torch.cdist(pred_hands, predicted_vertices).amin(dim=(1, 2)) < 0.05
        gt_contact = torch.cdist(gt_hands, target_vertices).amin(dim=(1, 2)) < 0.05
        sequence_contact_counts = np.asarray((
            int((pred_contact & gt_contact).sum()),
            int((pred_contact & ~gt_contact).sum()),
            int((~pred_contact & ~gt_contact).sum()),
            int((~pred_contact & gt_contact).sum()),
        ))
        contact_counts += sequence_contact_counts
        joints_np = predicted_joints.detach().cpu().numpy().copy()
        floor = min(float(joints_np[:, 10, 1].min()), float(joints_np[:, 11, 1].min()))
        joints_np[..., 1] -= floor
        terms = []
        for joint, height in ((7, 0.08), (10, 0.04), (8, 0.08), (11, 0.04)):
            displacement = np.linalg.norm(np.diff(joints_np[:, joint][:, (0, 2)], axis=0), axis=1)
            y = joints_np[:-1, joint, 1]
            terms.append(float(np.abs(displacement * (2.0 - 2.0 ** (y / height)))[y < height].sum()) / len(joints_np) * 100.0)
        sequence_foot_sliding = float(np.mean(terms))
        foot_sliding.append(sequence_foot_sliding)
        sequence_tp, sequence_fp, sequence_tn, sequence_fn = sequence_contact_counts.tolist()
        sequence_precision = (
            sequence_tp / (sequence_tp + sequence_fp)
            if sequence_tp + sequence_fp else 0.0
        )
        sequence_recall = (
            sequence_tp / (sequence_tp + sequence_fn)
            if sequence_tp + sequence_fn else 0.0
        )
        sequence_f1 = (
            2.0 * sequence_precision * sequence_recall
            / (sequence_precision + sequence_recall)
            if sequence_precision + sequence_recall else 0.0
        )
        per_sequence.append({
            "sequence": name,
            "foot_sliding": sequence_foot_sliding,
            "physical_contact_f1": float(sequence_f1),
            "physical_contact_precision": float(sequence_precision),
            "physical_contact_recall": float(sequence_recall),
            "contact_counts": {
                "tp": sequence_tp,
                "fp": sequence_fp,
                "tn": sequence_tn,
                "fn": sequence_fn,
            },
        })
    tp, fp, tn, fn = contact_counts.tolist()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    windows = {
        str(step): {
            key: float(np.mean([value[key] for value in per_window if value["window"] == step]))
            for key in (
                "mpjpe_cm", "pelvis_goal_error_cm", "object_goal_error_cm",
                "joint_position_mae_cm", "pelvis_translation_mae_cm",
                "object_translation_mae_cm",
                "object_rotation_geodesic_deg", "contact_channel_mse",
            )
        }
        for step in (1, 2, 3)
    }
    third_object = [value["object_goal_error_cm"] for value in per_window if value["window"] == 3]
    return {
        "aggregate": {
            "object_goal_error_cm": float(np.mean(third_object)),
            "pelvis_goal_error_cm": float(np.mean(pelvis_goal_errors)),
            "mpjpe_cm": float(torch.cat(all_joint_relative_errors).mean() * 100.0),
            "pelvis_translation_mae_cm": float(np.mean([
                value["pelvis_translation_mae_cm"] for value in per_window
            ])),
            "object_translation_mae_cm": float(np.mean([
                value["object_translation_mae_cm"] for value in per_window
            ])),
            "object_rotation_geodesic_deg": float(np.mean([
                value["object_rotation_geodesic_deg"] for value in per_window
            ])),
            "foot_sliding": float(np.mean(foot_sliding)),
            "physical_contact_f1": float(f1),
            "physical_contact_precision": float(precision),
            "physical_contact_recall": float(recall),
            "finite": bool(all(math.isfinite(value) for value in (
                np.mean(third_object), np.mean(pelvis_goal_errors), np.mean(foot_sliding), f1,
            ))),
        },
        "by_window": windows,
        "per_sequence_window": per_window,
        "per_sequence": per_sequence,
        "contact_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "object_surface_vertex_sampling": "uniform-index-up-to-2048-from-hash-verified-rest-ply",
    }


@torch.no_grad()
def rollout_diagnostic(model, diffusion, dataset, triples, device, weight_name):
    positions_by_step = [[triple[step] for triple in triples] for step in range(3)]
    items_by_step = [[dataset[position] for position in positions] for positions in positions_by_step]
    # GT-history/rebased path: every window uses its own GT frame, reference and BPS.
    gt_decoded = []
    gt_samples = []
    for step in range(3):
        batch = stack_items(dataset, positions_by_step[step], device)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(f"D0:{weight_name}:gt-rebased:{step}"))
        sample = diffusion.sample(
            model, batch["x"][:, :2], batch["text_embedding"], batch["object_bps"],
            batch["goals"], normalize_progress(batch["progress"]), generator=generator,
        )
        gt_samples.append(sample)
        decoded_rows = [decode_window(dataset, position, sample[row]) for row, position in enumerate(positions_by_step[step])]
        gt_decoded.append({key: torch.stack([value[key] for value in decoded_rows]) for key in (
            "joints", "human_rotation", "object_translation", "object_rotation", "contact",
        )})
    gt_metrics = physical_summary(dataset, triples, gt_decoded, device)

    # Failed generated-history path: first BPS/reference persist across handoffs.
    first_batch = stack_items(dataset, positions_by_step[0], device)
    origins, rotations, references = [], [], []
    for position in positions_by_step[0]:
        origin, rotation, reference, _, _ = frame_contract(dataset, position, device)
        origins.append(origin)
        rotations.append(rotation)
        references.append(reference)
    origin = torch.stack(origins)
    world_to_local = torch.stack(rotations)
    first_reference = torch.stack(references)
    fixed = first_batch["x"][:, :2]
    text = first_batch["text_embedding"]
    bps = first_batch["object_bps"]
    goals = first_batch["goals"]
    generated_decoded = []
    for step in range(3):
        current_items = items_by_step[step]
        progress = torch.stack([item["progress"] for item in current_items]).to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(f"D0:{weight_name}:generated:{step}"))
        sample = diffusion.sample(
            model, fixed, text, bps, goals, normalize_progress(progress), generator=generator,
        )
        decoded = decode_generated_batch(dataset, sample, origin, world_to_local, first_reference)
        generated_decoded.append(decoded)
        if step < 2:
            handoff = generated_handoff(
                dataset, sample, decoded, items_by_step[step + 1], positions_by_step[step + 1], first_reference,
            )
            fixed, goals = handoff["fixed"], handoff["goals"]
            origin, world_to_local = handoff["origin"], handoff["world_to_local"]
    generated_metrics = physical_summary(dataset, triples, generated_decoded, device)
    return {
        "single_window_gt_history": {
            "aggregate": gt_metrics["by_window"]["1"],
            "definition": "first window of the fixed three-window set",
        },
        "three_window_gt_rebased_history": gt_metrics,
        "three_window_generated_history_legacy": generated_metrics,
    }


def resolved_config(args) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B",
        "mode": "D0-internal-diagnostic",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "split_manifest": str((REPO / args.split_manifest).resolve()),
        "partition": "internal_validation",
        "official_test_used": False,
        "chois_used": False,
        "weights": ["online", "ema_0.9999"],
        "timesteps": list(D0_TIMESTEPS),
        "teacher_windows": 512,
        "rollout_sequences": 128,
        "rollout_windows": 3,
        "teacher_batch_size": args.teacher_batch_size,
        "device": args.device,
        "output": str(Path(args.output).resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--split-manifest", default="experiments/splits/omomo_hoi_train_validation_seed42.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    archived = json.loads(config_path.read_text(encoding="utf-8"))
    if archived != config:
        raise ValueError("runtime arguments do not match the archived resolved D0 config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D0 requires INFBAGEL_WORKER_EXPERT=hoi")
    checkpoint_path = Path(args.checkpoint).resolve()
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    if actual_checkpoint_sha != args.checkpoint_sha256:
        raise ValueError(f"checkpoint hash mismatch: {actual_checkpoint_sha}")
    split_path = (REPO / args.split_manifest).resolve()
    if "omomo_hoi_train_validation_seed42" not in split_path.name:
        raise ValueError("D0 refuses non-Phase-1A HOI split manifests")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D0 is a worker CUDA workload")
    seed_everything(42)
    dataset = PriorWindowDataset(
        str(REPO), "hoi", "internal_validation", split_manifest=str(split_path),
    )
    triples = select_internal_triples(dataset, 128)
    teacher_positions = select_teacher_windows(dataset, 512)
    sequence_names = [
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[triple[0]])])])
        for triple in triples
    ]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "hoi_prior_phase1b" or checkpoint.get("initialization") != "random":
        raise ValueError("D0 accepts only a randomly initialized Phase 1B checkpoint")
    model_config = checkpoint["model_config"]
    diffusion = GaussianDiffusion(500).to(device)
    weights = {"online": "model", "ema_0.9999": "ema_model"}
    results = {}
    started = time.time()
    for weight_name, state_key in weights.items():
        model = build_expert("hoi", init_checkpoint=None, **model_config).to(device).eval()
        model.load_state_dict(checkpoint[state_key], strict=True)
        with torch.no_grad():
            teacher = teacher_diagnostic(
                model, diffusion, dataset, teacher_positions, device, weight_name, args.teacher_batch_size,
            )
            rollout = rollout_diagnostic(model, diffusion, dataset, triples, device, weight_name)
        results[weight_name] = {"teacher_forced_x0": teacher, "generation": rollout}
        del model
        torch.cuda.empty_cache()
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B",
        "diagnostic": "D0-existing-checkpoint-no-retraining",
        "seed": 42,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": actual_checkpoint_sha,
            "processed_windows": checkpoint.get("processed_windows"),
            "optimizer_updates": checkpoint.get("optimizer_updates"),
        },
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "split_manifest": str(split_path),
        "selection": {
            "partition": "internal_validation",
            "official_test_sequence_count": 0,
            "chois_sequence_count": 0,
            "sequence_algorithm": "SHA256('42:hoi-remediation:' + sequence_name), first 128 eligible",
            "triple_pi": [0, 42, 84],
            "sequence_count": len(triples),
            "sequence_names": sequence_names,
            "sequence_selection_sha256": selection_sha256(sequence_names),
            "teacher_algorithm": "SHA256('42:hoi-remediation-window:' + sequence_name + ':' + pi)",
            "teacher_window_count_per_timestep": len(teacher_positions),
            "teacher_window_indices_sha256": selection_sha256(
                int(dataset.indices[position]) for position in teacher_positions
            ),
        },
        "structural_failures": {
            "pelvis_condition_missing_windows": 512,
            "pelvis_condition_total_windows": 512,
            "pelvis_condition_missing_fraction": 1.0,
            "sampler_discards_pelvis_goal": True,
        },
        "weights": results,
        "runtime": {
            "seconds": time.time() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "external_contention_not_used_for_selection": True,
        },
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
