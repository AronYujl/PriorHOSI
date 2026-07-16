#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-R0 state-routed guidance diagnostic."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from pytorch3d import transforms


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.contact_alignment import (  # noqa: E402
    PHYSICAL_THRESHOLDS_CM,
    all_finite,
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
    deterministic_vertex_subset,
    decoded_fk_positions,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402
from priors.routed_guidance import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINT_SHA256,
    HISTORY_MAX_ABS,
    KINEMATIC_METRICS,
    MASKED_OFF_MAX_ABS,
    NORM_REPLAY_RELATIVE_ERROR,
    PHASE_OFFSETS,
    PRIMARY_VARIANT,
    PRIOR_ROLLOUT_OFFSETS,
    RUN_ID,
    SELECTION_SHA256,
    SUBPHASE,
    UPPER_ROTATION_JOINTS,
    VARIANTS,
    WINDOWS_PER_SEQUENCE,
    mechanism_gate,
    paired_variant_comparison,
    sample_routed_counterfactual,
    sampler_seed_label,
    select_routed_holdout,
    stable_seed,
    upper_rotation_mask,
)
from priors.window_codec import BPS_SHA256, project_to_so3  # noqa: E402
from tools.diagnose_hoi_d2q import (  # noqa: E402
    _rest_batch,
    _sequence_name,
    _summary_for_records,
    analyze_generated_sequence,
    author_blob_hashes,
    author_formula_replay_max_abs,
    exclusive_json,
    git_output,
    prepare_targets,
    reports_complete,
    rest_mesh_contract,
    sha256_file,
    sha256_tensor_state,
)
from tools.diagnose_hoi_remediation import (  # noqa: E402
    goal_globals,
    seed_everything,
)
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
MUTABLE_FRAMES_PER_WINDOW = (
    REPRESENTATION.window_frames - REPRESENTATION.history_frames
)


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": SUBPHASE,
        "mode": "state-subspace-routed-author-guidance",
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
                "SHA256(42:d2r-routed-guidance:sequence_name:7,49,91), "
                "sequence_name, sequence_id"
            ),
            "sequences": 64,
            "windows_per_sequence": WINDOWS_PER_SEQUENCE,
            "windows": 192,
            "global_window_indices_sha256": SELECTION_SHA256,
        },
        "sampling": {
            "variants": list(VARIANTS),
            "primary_gate_variant": PRIMARY_VARIANT,
            "diffusion_steps": 500,
            "condition_variant": "matched",
            "paired_initial_and_posterior_noise": True,
            "guidance_steps": list(range(499, 0, -1)),
            "step_zero_guidance": False,
            "author_hand_weight": AUTHOR_HAND_WEIGHT,
            "semantic_channels": [0, 1],
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "spatial_threshold_m": SPATIAL_THRESHOLD_M,
            "fk_palm_indices": list(FK_PALM_INDICES),
            "direct_hand_indices": list(DIRECT_HAND_INDICES),
            "upper_rotation_joints": list(UPPER_ROTATION_JOINTS),
            "upper_norm": (
                "per-sample mutable-frame L2 author_all/upper_raw; "
                "no clip or sweep"
            ),
            "rest_vertex_count": REST_VERTEX_COUNT,
            "rest_vertex_sampling": "deterministic-uniform-index",
            "posterior_helper": (
                "priors.diffusion.GaussianDiffusion.posterior_sample"
            ),
            "injection": "x_prev += routed_grad(-(10 * hand_core), pred_x0)",
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
                "guidance_scale_1",
                "x_prev_plus_negative_loss_gradient",
            ],
            "registered_routing_deviations": [
                "human_only_zeroes_object_translation_and_rotation",
                "upper_raw_projects_to_registered_upper_rotation_joints",
                "upper_norm_preserves_per_sample_mutable_state_l2",
            ],
            "other_deviations": [
                "feet_floor_weight_500_omitted",
                "scene_and_penetration_terms_omitted",
                "deterministic_2048_vertex_surface_instead_of_random_10000",
                "codec_differentiable_so3_decode",
                "ddpm_500_step_checkpoint_instead_of_consistency_sampler",
            ],
        },
        "evaluation": {
            "primary_contact_target": (
                "author-native-equivalent GT 24-joint FK palms 22/23"
            ),
            "descriptive_contact_target": (
                "28-joint direct representation hands 24/26"
            ),
            "physical_thresholds_cm": list(PHYSICAL_THRESHOLDS_CM),
            "units": ["left_hand", "right_hand", "union"],
            "kinematic_metrics": list(KINEMATIC_METRICS),
            "object_rotation_geodesic_unit": "degrees",
            "state_displacement": (
                "per-frame normalized-state L2 by representation field"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_unit": "sequence",
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
            "rollout_bps": "recomputed_from_current_generated_object_pose",
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
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
        "official_test_used": False,
        "chois_used": False,
        "output": str(args.output.resolve()),
    }


def rollout_chunk(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
    rest_vertices: Mapping[str, torch.Tensor],
    rest_subsets: Mapping[str, torch.Tensor],
    parents_24: torch.Tensor,
    *,
    chunk_index: int,
    variant: str,
) -> Dict[str, object]:
    positions_by_step = [
        [triple[step] for triple in triples]
        for step in range(WINDOWS_PER_SEQUENCE)
    ]
    items_by_step = [
        [dataset[position] for position in positions]
        for positions in positions_by_step
    ]
    names = [
        _sequence_name(dataset, position)
        for position in positions_by_step[0]
    ]
    first_items = items_by_step[0]
    frame = stack_frames(first_items, device)
    fixed = torch.stack([item["x"][:2] for item in first_items]).to(device)
    decoded_steps = []
    normalized_steps = []
    fk_steps = []
    noise_streams = []
    guidance_windows = []
    history_max_abs = 0.0
    for window_index in range(WINDOWS_PER_SEQUENCE):
        items = items_by_step[window_index]
        gt_frame = stack_frames(items, device)
        pelvis_global, object_global = global_goals(
            dataset, items, gt_frame, device,
        )
        goals = torch.zeros(len(triples), 9, device=device)
        goals[:, :3] = dataset.codec.pelvis_goal(pelvis_global, frame)
        goals[:, 6:9] = dataset.codec.object_goal(object_global, frame)
        text = torch.stack(
            [item["text_embedding"] for item in items]
        ).to(device)
        bps = current_bps(
            dataset, frame.object_reference, names, rest_vertices,
        )
        progress = normalize_progress(
            torch.stack([item["progress"] for item in items]).to(device)
        )
        rest_offsets = torch.stack([
            item["rest_human_offsets"] for item in items
        ]).to(device)
        surface = _rest_batch(names, rest_subsets, device)
        label = sampler_seed_label(chunk_index, window_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(label))
        initial_state = sha256_tensor_state(generator.get_state())
        sample, guidance_audit = sample_routed_counterfactual(
            diffusion,
            model,
            fixed,
            text,
            bps,
            goals,
            progress,
            generator=generator,
            variant=variant,
            codec=dataset.codec,
            frame=frame,
            rest_human_offsets=rest_offsets,
            parents_24=parents_24,
            rest_vertices=surface,
        )
        final_state = sha256_tensor_state(generator.get_state())
        normalized_steps.append(sample.detach().cpu())
        history_max_abs = max(
            history_max_abs,
            float(
                (
                    sample[:, :2] - fixed
                ).abs().max().detach().cpu()
            ),
            float(guidance_audit["history_max_abs"]),
        )
        decoded = dataset.codec.decode(sample, frame)
        fk = decoded_fk_positions(decoded, rest_offsets, parents_24)
        decoded_steps.append({
            key: value.detach().cpu()
            for key, value in decoded.items()
        })
        fk_steps.append(fk.detach().cpu())
        noise_streams.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "label": label,
            "seed": stable_seed(label),
            "generator_initial_state_sha256": initial_state,
            "generator_final_state_sha256": final_state,
        })
        guidance_windows.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            **guidance_audit,
        })
        if window_index < WINDOWS_PER_SEQUENCE - 1:
            fixed, frame = dataset.codec.encode(
                decoded["joints"][:, -2:],
                decoded["human_rotation"][:, -2:],
                global_object_translation=decoded[
                    "object_translation"
                ][:, -2:],
                global_object_rotation=decoded["object_rotation"][:, -2:],
                contact=decoded["contact"][:, -2:],
            )
    generated = []
    for row in range(len(triples)):
        generated.append({
            key: torch.cat([
                decoded_steps[step][key][row, 2:]
                for step in range(WINDOWS_PER_SEQUENCE)
            ])
            for key in (
                "joints",
                "human_rotation",
                "object_translation",
                "object_rotation",
                "contact",
            )
        })
        generated[-1]["fk_joints"] = torch.cat([
            fk_steps[step][row, 2:]
            for step in range(WINDOWS_PER_SEQUENCE)
        ])
    return {
        "generated": generated,
        "decoded_steps": decoded_steps,
        "normalized_steps": normalized_steps,
        "noise_streams": noise_streams,
        "guidance_windows": guidance_windows,
        "history_max_abs": history_max_abs,
    }


def _foot_sliding(joints: torch.Tensor) -> float:
    joints_np = joints.detach().cpu().numpy().copy()
    floor = min(
        float(joints_np[:, 10, 1].min()),
        float(joints_np[:, 11, 1].min()),
    )
    joints_np[..., 1] -= floor
    terms = []
    for joint, height in (
        (7, 0.08), (10, 0.04), (8, 0.08), (11, 0.04),
    ):
        displacement = np.linalg.norm(
            np.diff(joints_np[:, joint][:, (0, 2)], axis=0),
            axis=1,
        )
        y = joints_np[:-1, joint, 1]
        active = y < height
        terms.append(
            float(
                np.abs(
                    displacement * (2.0 - 2.0 ** (y / height))
                )[active].sum()
            )
            / len(joints_np)
            * 100.0
        )
    return float(np.mean(terms))


def native_like_kinematics(
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    targets: Sequence[Mapping[str, object]],
    generated: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, object]:
    per_sequence = []
    per_sequence_window = []
    for triple, target, prediction in zip(triples, targets, generated):
        rows = []
        for step, position in enumerate(triple):
            start = step * MUTABLE_FRAMES_PER_WINDOW
            stop = start + MUTABLE_FRAMES_PER_WINDOW
            pred_fk = prediction["fk_joints"][start:stop].to(device)
            target_fk = target["fk_joints"][start:stop].to(device)
            pred_object = prediction["object_translation"][start:stop].to(device)
            target_object = target["object_translation"][start:stop].to(device)
            pred_rotation = prediction["object_rotation"][start:stop].to(device)
            target_rotation = target["object_rotation"][start:stop].to(device)
            relative_pred = pred_fk - pred_fk[:, :1]
            relative_target = target_fk - target_fk[:, :1]
            fk_mpjpe = torch.linalg.vector_norm(
                relative_pred - relative_target, dim=-1,
            ).mean() * 100.0
            pelvis_goal, object_goal = goal_globals(
                dataset, position, dataset[position], device,
            )
            pelvis_error = torch.linalg.vector_norm(
                pred_fk[-1, 0][[0, 2]] - pelvis_goal[[0, 2]]
            ) * 100.0
            object_goal_error = torch.linalg.vector_norm(
                pred_object[-1] - object_goal,
            ) * 100.0
            object_translation = torch.linalg.vector_norm(
                pred_object - target_object, dim=-1,
            ).mean() * 100.0
            object_rotation = transforms.so3_relative_angle(
                project_to_so3(pred_rotation),
                project_to_so3(target_rotation),
                cos_bound=1e-7,
            ).mean() * (180.0 / math.pi)
            row = {
                "sequence": target["sequence"],
                "window": step + 1,
                "pi": int(
                    dataset.language["pi"][
                        int(dataset.indices[position])
                    ]
                ),
                "fk_mpjpe_cm": float(fk_mpjpe),
                "pelvis_goal_error_cm": float(pelvis_error),
                "object_goal_error_cm": float(object_goal_error),
                "object_translation_mae_cm": float(object_translation),
                "object_rotation_geodesic": float(object_rotation),
            }
            rows.append(row)
            per_sequence_window.append(row)
        sequence = {
            "sequence": target["sequence"],
            "fk_mpjpe_cm": float(np.mean([
                row["fk_mpjpe_cm"] for row in rows
            ])),
            "pelvis_goal_error_cm": float(np.mean([
                row["pelvis_goal_error_cm"] for row in rows
            ])),
            "object_goal_error_cm": float(rows[-1]["object_goal_error_cm"]),
            "object_translation_mae_cm": float(np.mean([
                row["object_translation_mae_cm"] for row in rows
            ])),
            "object_rotation_geodesic": float(np.mean([
                row["object_rotation_geodesic"] for row in rows
            ])),
            "fk_foot_sliding": _foot_sliding(
                prediction["fk_joints"].to(device)
            ),
        }
        sequence["finite"] = bool(all(
            math.isfinite(float(sequence[key]))
            for key in KINEMATIC_METRICS
        ))
        per_sequence.append(sequence)
    aggregate = {
        key: float(np.mean([
            row[key] for row in per_sequence
        ]))
        for key in KINEMATIC_METRICS
    }
    aggregate["finite"] = bool(
        all(row["finite"] for row in per_sequence)
        and all(math.isfinite(value) for value in aggregate.values())
    )
    return {
        "aggregate": aggregate,
        "per_sequence": per_sequence,
        "per_sequence_window": per_sequence_window,
        "target_contract": (
            "GT 24-joint FK from rotations/rest offsets; generated 24-joint FK"
        ),
    }


def state_displacement(
    control_steps: Sequence[torch.Tensor],
    candidate_steps: Sequence[torch.Tensor],
    sequence_names: Sequence[str],
) -> Dict[str, object]:
    if len(control_steps) != WINDOWS_PER_SEQUENCE or len(candidate_steps) != WINDOWS_PER_SEQUENCE:
        raise ValueError("D2-R state displacement requires three rollout windows")
    if any(first.shape != second.shape for first, second in zip(
        control_steps, candidate_steps,
    )):
        raise ValueError("D2-R paired normalized-state shapes differ")
    per_sequence = []
    for row, name in enumerate(sequence_names):
        difference = torch.cat([
            candidate_steps[step][
                row, REPRESENTATION.history_frames:
            ]
            - control_steps[step][
                row, REPRESENTATION.history_frames:
            ]
            for step in range(WINDOWS_PER_SEQUENCE)
        ])
        fields = {}
        for field in REPRESENTATION.fields:
            fields[field.name] = float(
                torch.linalg.vector_norm(
                    difference[..., field.slice], dim=-1,
                ).mean()
            )
        per_sequence.append({"sequence": name, "fields": fields})
    return {
        "definition": (
            "mean per-frame L2 in normalized pre-decode state, candidate-control"
        ),
        "aggregate": {
            field.name: float(np.mean([
                row["fields"][field.name] for row in per_sequence
            ]))
            for field in REPRESENTATION.fields
        },
        "per_sequence": per_sequence,
    }


def _numeric_summary(values: Sequence[float]) -> Dict[str, object]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


def guidance_audit_summary(
    windows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    variant = str(windows[0]["variant"]) if windows else ""
    per_step = [
        step
        for window in windows
        for step in window["per_step"]
    ]
    scalar_names = (
        "loss",
        "raw_hand_loss",
        "author_hand_weight",
        "spatial",
        "temporal",
        "mask_coverage",
        "distance_mean_m",
        "gradient_norm",
        "gradient_rms",
        "gradient_max_abs",
        "norm_replay_relative_error_max",
        "masked_off_max_abs",
        "routing_formula_replay_max_abs",
        "routed_history_max_abs",
    )
    aggregate = {
        name: _numeric_summary([
            float(value[name]) for value in per_step
        ])
        for name in scalar_names
    }
    scales = [
        float(scale)
        for value in per_step
        for scale in value["routing_scale"]
    ]
    aggregate["routing_scale"] = _numeric_summary(scales)
    energy = {}
    for source in ("full_gradient", "injected_gradient"):
        energy[source] = {
            "stats": {
                name: _numeric_summary([
                    float(value[source][name]) for value in per_step
                ])
                for name in ("norm", "rms", "max_abs")
            },
            "fields": {
                field.name: _numeric_summary([
                    float(value[source]["fields"][field.name])
                    for value in per_step
                ])
                for field in REPRESENTATION.fields
            },
            "rotation_joints": {
                str(joint): _numeric_summary([
                    float(value[source]["rotation_joints"][str(joint)])
                    for value in per_step
                ])
                for joint in range(22)
            },
        }
    guided = variant != "unguided"
    return {
        "variant": variant,
        "guided": guided,
        "windows": len(windows),
        "applied_steps": sum(
            int(value["applied_steps"]) for value in windows
        ),
        "expected_applied_steps": len(windows) * 499 if guided else 0,
        "step_zero_guidance_applied": any(
            bool(value["step_zero_guidance_applied"]) for value in windows
        ),
        "finite": all(bool(value["finite"]) for value in windows),
        "invalid_nonzero_full_zero_projection": any(
            bool(value["invalid_nonzero_full_zero_projection"])
            for value in per_step
        ),
        "aggregate": aggregate,
        "gradient_energy": energy,
        "per_window": list(windows),
    }


def _ancestor_rotation_contract(parents_24: torch.Tensor) -> bool:
    represented = set()
    parents = [int(value) for value in parents_24.detach().cpu().tolist()]
    for palm in FK_PALM_INDICES:
        parent = parents[palm]
        while parent >= 0:
            if parent != 0 and parent < 22:
                represented.add(parent)
            parent = parents[parent]
    return represented == set(UPPER_ROTATION_JOINTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256", default=CHECKPOINT_SHA256,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-R0 run id must be {RUN_ID}")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-R0 batch size must evenly divide 64")
    if args.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("D2-R0 checkpoint hash differs from preregistration")
    config = resolved_config(args)
    config_path = args.resolved_config.resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError(
            "D2-R0 runtime arguments do not match archived resolved config"
        )
    if Path(sys.executable).resolve() != Path(
        os.environ.get("INFBAGEL_PYTHON", ""),
    ).resolve():
        raise ValueError(
            "D2-R0 requires the absolute INFBAGEL_PYTHON interpreter"
        )
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-R0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if git_output("status", "--porcelain"):
        raise RuntimeError("D2-R0 refuses a dirty worker checkout")
    checkpoint_path = args.checkpoint.resolve()
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            f"D2-R0 checkpoint hash mismatch: {actual_checkpoint_sha256}"
        )
    asset_hashes = {
        "normalization": sha256_file(
            (REPO / "data/train/norm.npy").resolve()
        ),
        "bps": sha256_file((REPO / "code/bps.pt").resolve()),
    }
    expected_asset_hashes = {
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "bps": BPS_SHA256,
    }
    if asset_hashes != expected_asset_hashes:
        raise ValueError(
            f"D2-R0 asset hash mismatch: {asset_hashes} "
            f"!= {expected_asset_hashes}"
        )
    actual_author_hashes = author_blob_hashes()
    if actual_author_hashes != AUTHOR_BLOB_SHA256:
        raise ValueError("D2-R0 author blob hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-R0 is a four-GPU-worker CUDA diagnostic")
    if args.output.resolve().exists():
        raise FileExistsError(
            f"refusing to overwrite {args.output.resolve()}"
        )

    seed_everything(42)
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest=(
            "experiments/splits/omomo_hoi_train_validation_seed42.json"
        ),
    )
    selection = select_routed_holdout(dataset)
    triples = selection["triples"]
    parents_24 = torch.from_numpy(
        get_smpl_parents(use_joints24=True).copy(),
    ).long().to(device)
    targets = prepare_targets(dataset, triples, parents_24.cpu())
    full_rest_vertices = load_rest_vertices(dataset, triples, device)
    rest_subsets = {
        name: deterministic_vertex_subset(vertices)
        for name, vertices in full_rest_vertices.items()
    }
    diffusion = GaussianDiffusion(500).to(device)
    model, metadata = load_trained_hoi_prior(
        str(checkpoint_path), device, weight_variant="online",
    )
    if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
        raise ValueError("D2-R0 checkpoint data-contract mismatch")
    model.eval()
    model_before = state_dict_sha256(model)
    variants: Dict[str, object] = {}
    variant_records: Dict[str, List[Dict[str, object]]] = {}
    variant_kinematics: Dict[str, object] = {}
    variant_states: Dict[str, Sequence[torch.Tensor]] = {}
    sequence_names = [str(target["sequence"]) for target in targets]
    for variant in VARIANTS:
        records = []
        generated_all = []
        state_chunks = []
        noise_streams = []
        guidance_windows = []
        history_max_abs = 0.0
        variant_error = None
        try:
            for chunk_index, offset in enumerate(
                range(0, len(triples), args.batch_size)
            ):
                selected_triples = triples[offset:offset + args.batch_size]
                rollout = rollout_chunk(
                    model,
                    diffusion,
                    dataset,
                    selected_triples,
                    device,
                    full_rest_vertices,
                    rest_subsets,
                    parents_24,
                    chunk_index=chunk_index,
                    variant=variant,
                )
                state_chunks.append(rollout["normalized_steps"])
                noise_streams.extend(rollout["noise_streams"])
                guidance_windows.extend(rollout["guidance_windows"])
                history_max_abs = max(
                    history_max_abs, float(rollout["history_max_abs"]),
                )
                generated_all.extend(rollout["generated"])
                for target, generated in zip(
                    targets[offset:offset + args.batch_size],
                    rollout["generated"],
                ):
                    records.append(analyze_generated_sequence(
                        target,
                        generated,
                        full_rest_vertices,
                        device,
                    ))
            normalized_steps = [
                torch.cat([
                    chunk[step] for chunk in state_chunks
                ])
                for step in range(WINDOWS_PER_SEQUENCE)
            ]
            kinematics = native_like_kinematics(
                dataset, triples, targets, generated_all, device,
            )
            summary = _summary_for_records(
                records, include_categories=True,
            )
            audit = guidance_audit_summary(guidance_windows)
            finite = bool(
                all_finite(summary)
                and all_finite(kinematics)
                and all_finite(audit)
                and all(
                    torch.isfinite(value).all()
                    for value in normalized_steps
                )
            )
            complete = bool(
                len(records) == 64
                and reports_complete(summary)
                and all(reports_complete(record) for record in records)
                and set(kinematics["aggregate"]) >= (
                    set(KINEMATIC_METRICS) | {"finite"}
                )
                and len(kinematics["per_sequence"]) == 64
                and len(kinematics["per_sequence_window"]) == 192
            )
        except Exception as exc:  # preserve a contract-failure artifact
            variant_error = f"{type(exc).__name__}: {exc}"
            summary = {}
            kinematics = {}
            audit = guidance_audit_summary(guidance_windows)
            finite = False
            complete = False
            normalized_steps = []
        variants[variant] = {
            "error": variant_error,
            "history_max_abs": history_max_abs,
            "noise_streams": noise_streams,
            "guidance_audit": audit,
            "aggregate": summary,
            "native_like_kinematics": kinematics,
            "per_sequence": records,
            "finite": finite,
            "all_fields_thresholds_and_metrics_reported": complete,
        }
        variant_records[variant] = records
        variant_kinematics[variant] = kinematics
        variant_states[variant] = normalized_steps
        torch.cuda.empty_cache()

    comparisons = {}
    comparison_errors = {}
    state_displacements = {}
    for variant in VARIANTS:
        if variant == "unguided":
            continue
        try:
            comparisons[variant] = paired_variant_comparison(
                variant_records["unguided"],
                variant_records[variant],
                variant_kinematics["unguided"],
                variant_kinematics[variant],
            )
            state_displacements[variant] = state_displacement(
                variant_states["unguided"],
                variant_states[variant],
                sequence_names,
            )
            comparison_errors[variant] = None
        except Exception as exc:
            comparisons[variant] = {}
            state_displacements[variant] = {}
            comparison_errors[variant] = f"{type(exc).__name__}: {exc}"
    model_after = state_dict_sha256(model)

    all_noise_streams = [
        variants[variant]["noise_streams"] for variant in VARIANTS
    ]
    reference_noise = all_noise_streams[0]
    paired_noise_identity = bool(
        reference_noise
        and all(value == reference_noise for value in all_noise_streams[1:])
    )
    custom_source = inspect.getsource(sample_routed_counterfactual)
    production_source = inspect.getsource(GaussianDiffusion.sample)
    formula_replay_max_abs = author_formula_replay_max_abs(device)
    guided_variants = [variant for variant in VARIANTS if variant != "unguided"]

    def audit_stat(variant: str, name: str, statistic: str):
        try:
            return variants[variant]["guidance_audit"]["aggregate"][
                name
            ][statistic]
        except (KeyError, TypeError):
            return None

    contract = {
        "checkpoint_hash_exact": (
            actual_checkpoint_sha256 == CHECKPOINT_SHA256
        ),
        "asset_hashes_exact": asset_hashes == expected_asset_hashes,
        "author_blob_hashes_exact": (
            actual_author_hashes == AUTHOR_BLOB_SHA256
        ),
        "author_formula_replay_max_abs_le_1e-5": (
            formula_replay_max_abs <= 1e-5
        ),
        "data_contract_exact": (
            metadata["data_contract_sha256"]
            == EXPECTED_DATA_CONTRACT_SHA256
        ),
        "selection_exact": (
            selection["sha256"] == SELECTION_SHA256
            and selection["sequences"] == 64
            and selection["windows"] == 192
            and selection["phase_offsets"] == list(PHASE_OFFSETS)
        ),
        "selection_disjoint_from_prior_rollout_offsets": set(
            selection["phase_offsets"]
        ).isdisjoint(PRIOR_ROLLOUT_OFFSETS),
        "upper_chain_parent_mapping_exact": _ancestor_rotation_contract(
            parents_24
        ),
        "upper_mask_exact": int(upper_rotation_mask().sum()) == (
            len(UPPER_ROTATION_JOINTS) * 6
        ),
        "paired_sampler_noise_identity": paired_noise_identity,
        "history_restoration": all(
            float(variants[variant]["history_max_abs"])
            <= HISTORY_MAX_ABS
            for variant in VARIANTS
        ),
        "all_finite": all(
            bool(variants[variant]["finite"]) for variant in VARIANTS
        ),
        "all_variants_reported": set(variants) == set(VARIANTS),
        "all_fields_thresholds_and_metrics_reported": all(
            bool(variants[variant][
                "all_fields_thresholds_and_metrics_reported"
            ])
            for variant in VARIANTS
        ),
        "guided_steps_and_step_zero_exact": all(
            (
                int(variants[variant]["guidance_audit"]["applied_steps"])
                == int(variants[variant]["guidance_audit"][
                    "expected_applied_steps"
                ])
                and not bool(variants[variant]["guidance_audit"][
                    "step_zero_guidance_applied"
                ])
            )
            for variant in VARIANTS
        ),
        "author_hand_weight_exact": all(
            audit_stat(variant, "author_hand_weight", "min")
            == AUTHOR_HAND_WEIGHT
            and audit_stat(variant, "author_hand_weight", "max")
            == AUTHOR_HAND_WEIGHT
            for variant in guided_variants
        ),
        "routing_masked_off_max_abs_le_1e-7": all(
            audit_stat(variant, "masked_off_max_abs", "max") is not None
            and float(audit_stat(
                variant, "masked_off_max_abs", "max",
            )) <= MASKED_OFF_MAX_ABS
            for variant in guided_variants
        ),
        "routing_formula_replay_max_abs_le_1e-7": all(
            audit_stat(
                variant, "routing_formula_replay_max_abs", "max",
            ) is not None
            and float(audit_stat(
                variant, "routing_formula_replay_max_abs", "max",
            )) <= MASKED_OFF_MAX_ABS
            for variant in guided_variants
        ),
        "upper_norm_replay_relative_error_le_1e-5": (
            audit_stat(
                "upper_norm",
                "norm_replay_relative_error_max",
                "max",
            ) is not None
            and float(audit_stat(
                "upper_norm",
                "norm_replay_relative_error_max",
                "max",
            )) <= NORM_REPLAY_RELATIVE_ERROR
        ),
        "routing_zero_denominator_contract": not any(
            bool(variants[variant]["guidance_audit"][
                "invalid_nonzero_full_zero_projection"
            ])
            for variant in guided_variants
        ),
        "paired_comparisons_complete": all(
            comparison_errors[variant] is None
            and set(comparisons[variant]) == {"contact", "kinematics"}
            for variant in guided_variants
        ),
        "state_displacements_complete": all(
            set(state_displacements[variant].get("aggregate", {}))
            == {field.name for field in REPRESENTATION.fields}
            for variant in guided_variants
        ),
        "model_state_unchanged": model_before == model_after,
        "parameter_grad_buffers_clear": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "posterior_helper_reused": (
            "diffusion.posterior_sample(" in custom_source
        ),
        "production_sampler_default_unchanged": (
            "guidance" not in production_source
            and "sample_routed_counterfactual" not in production_source
        ),
        "sampler_future_gt_absent": (
            "future_gt" not in custom_source
            and "target[" not in custom_source
        ),
        "sampler_stored_per_frame_bps_absent": (
            "stored_per_frame_bps" not in custom_source
            and 'batch["object_bps"]' not in custom_source
        ),
    }
    decision = mechanism_gate(contract, comparisons.get(PRIMARY_VARIANT, {}))
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "status": "completed",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "selection": {
            key: value for key, value in selection.items()
            if key != "triples"
        },
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            **asset_hashes,
            "author_blob_sha256": actual_author_hashes,
            "rest_meshes": rest_mesh_contract(),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": CHECKPOINT_SHA256,
            "metadata": metadata,
            "model_state_sha256_before": model_before,
            "model_state_sha256_after": model_after,
            "model_state_unchanged": model_before == model_after,
            "parameter_grad_buffers_clear": all(
                parameter.grad is None for parameter in model.parameters()
            ),
        },
        "variants": variants,
        "paired_candidate_minus_unguided": comparisons,
        "paired_comparison_errors": comparison_errors,
        "state_displacements_vs_unguided": state_displacements,
        "contract": contract,
        "decision": decision,
        "sampler_contract": {
            "production_default_changed": False,
            "future_gt": False,
            "stored_per_frame_bps": False,
            "rollout_bps": (
                "recomputed_from_current_generated_object_pose"
            ),
            "paired_noise_identity": paired_noise_identity,
            "posterior_helper_reused": contract["posterior_helper_reused"],
            "reverse_so3_projection": False,
        },
        "author_parity": config["author_parity"],
        "author_formula_replay_max_abs": formula_replay_max_abs,
        "training_updates": 0,
        "optimizer_created": False,
        "checkpoint_write": False,
        "released_checkpoint_used": False,
        "ema_used": False,
        "checkpoint_selection": False,
        "production_guidance_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
        "official_test_used": False,
        "chois_used": False,
        "runtime_seconds": time.time() - started,
        "gpu": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "maximum_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "maximum_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
    }
    exclusive_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
