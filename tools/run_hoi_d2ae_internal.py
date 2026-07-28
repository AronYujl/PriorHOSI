#!/usr/bin/env python3
"""Run the fixed D2-AE0 sparse-relation internal causal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

import utils as author_utils  # noqa: E402
from datasets.utils import get_smpl_parents  # noqa: E402
from priors.contact_alignment import (  # noqa: E402
    PHASE_OFFSETS as COHORT_PHASE_OFFSETS,
    PRIOR_ROLLOUT_OFFSETS,
)
from priors.d2ae_diagnostic import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIRECT_HAND_INDICES,
    FK_PALM_INDICES,
    GT_CONTACT_FINITE_SEQUENCE_COUNT,
    GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256,
    HISTORY_MAX_ABS,
    PHYSICAL_THRESHOLDS_CM,
    ROLE_NAMES,
    SELECTION_SHA256,
    TEMPORAL_ANCHORS,
    VARIANTS,
    internal_mechanism_gate,
    paired_comparisons,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AE, load_trained_hoi_prior  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
)
from priors.window_codec import project_to_so3  # noqa: E402
from tools import run_hoi_d2ac_internal as base  # noqa: E402


SUBPHASE = "1B-D2-AE0-internal"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-sparse-relation-field-internal"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
TRAINING_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-sparse-relation-field"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
FAILURE_CLASSIFICATION = "sparse-relation-field-contract-failure-stop"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    if chunk_index < 0 or window_index not in range(base.WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-AE sampler seed coordinates")
    return f"D2:d2ae-shared:chunk:{chunk_index}:window:{window_index}"


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def checkpoint_contract(
    path: Path,
    expected_sha256: str,
    training_run_id: str,
) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_name = f"{training_run_id}_windows061440000.pth"
    if actual != expected_sha256:
        raise ValueError(f"D2-AE final checkpoint hash mismatch: {actual}")
    if path.name != expected_name:
        raise ValueError("D2-AE internal requires the fixed final checkpoint basename")
    checkpoint = torch.load(path, map_location="cpu")
    initialization = checkpoint.get("weight_initialization", {})
    relation = checkpoint.get("sparse_relation_contract", {})
    resume = checkpoint.get("resume_contract", {})
    checks = {
        "schema_version": checkpoint.get("schema_version") == 2,
        "checkpoint_type": checkpoint.get("checkpoint_type") == "hoi_prior_phase1b",
        "window_state_codec": checkpoint.get("window_state_codec")
        == "state-compositional-v1",
        "expert": checkpoint.get("expert") == "hoi",
        "run_id": checkpoint.get("run_id") == training_run_id,
        "seed": checkpoint.get("seed") == 42,
        "processed_windows": checkpoint.get("processed_windows") == 61_440_000,
        "processed_frames": checkpoint.get("processed_frames") == 983_040_000,
        "optimizer_updates": checkpoint.get("optimizer_updates") == 30_000,
        "world_size": checkpoint.get("world_size") == 4,
        "effective_batch_size": checkpoint.get("effective_batch_size") == 2048,
        "architecture_variant": (
            checkpoint.get("architecture_variant") == HOI_ARCHITECTURE_D2AE
            and checkpoint.get("model_config", {}).get("architecture_variant")
            == HOI_ARCHITECTURE_D2AE
        ),
        "sparse_relation_provenance": (
            relation.get("architecture_variant") == HOI_ARCHITECTURE_D2AE
            and relation.get("sparse_relation_parameters")
            == SPARSE_RELATION_PARAMETER_COUNT
            and relation.get("mapping_sha256") == SPARSE_POINT_MAPPING_SHA256
            and relation.get("manifest_sha256") == SPARSE_POINT_MANIFEST_SHA256
            and relation.get("stacked_tensor_sha256") == SPARSE_POINT_TENSOR_SHA256
            and relation.get("current_state_only") is True
            and relation.get("clean_target_used") is False
            and relation.get("future_gt_used") is False
            and relation.get("scene_used") is False
            and relation.get("contact_used") is False
            and relation.get("stored_relation_used") is False
        ),
        "data_contract": (
            checkpoint.get("data_contract_sha256")
            == base.EXPECTED_DATA_CONTRACT_SHA256
        ),
        "split": checkpoint.get("split_sha256") == base.EXPECTED_SPLIT_SHA256,
        "random_initialization": (
            checkpoint.get("initialization") == "random"
            and initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("source_checkpoint_sha256") is None
            and initialization.get("source_model_state_sha256") is None
            and initialization.get("restored_components") == []
            and all(
                initialization.get(name) == 0
                for name in (
                    "old_optimizer_states_loaded",
                    "old_ema_models_loaded",
                    "old_scheduler_states_loaded",
                    "old_scaler_states_loaded",
                    "old_rng_states_loaded",
                )
            )
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(initialization.get("initial_model_state_sha256", "")),
            ) is not None
        ),
        "no_ema": checkpoint.get("ema_models") == {},
        "online_model": isinstance(checkpoint.get("model"), dict),
        "d2x_routing": resume.get("fk_foot_temporal_routing") is True,
        "d2ab_disabled": resume.get("d2ab_predicted_support_no_slip") is False,
        "d2ac_disabled": resume.get("d2ac_interaction_adapter") is not True,
        "d2ad_disabled": resume.get("d2ad_local_frame_interaction_adapter") is not True,
        "d2ae_enabled": (
            resume.get("d2ae_sparse_relation_field") is True
            and resume.get("architecture_variant") == HOI_ARCHITECTURE_D2AE
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AE final checkpoint contract mismatch: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
        "run_id": training_run_id,
        "git_commit": checkpoint.get("git_commit"),
        "checks": checks,
        "initial_model_state_sha256": initialization.get(
            "initial_model_state_sha256"
        ),
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": SUBPHASE,
        "mode": "sparse-relation-field-internal-causal-diagnostic",
        "seed": 42,
        "git_commit": base.git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(Path(sys.executable).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "target_checkpoint": {
            "path": str(args.target_checkpoint.resolve()),
            "sha256": args.target_sha256,
            "run_id": args.training_run_id,
            "processed_windows": 61_440_000,
            "weight_variant": "online",
        },
        "selection": {
            "partition": "internal_validation",
            "source": "sealed D2-O cohort",
            "phase_offsets": [14, 56, 98],
            "sequences": 64,
            "windows_per_sequence": 3,
            "windows": 192,
            "sha256": SELECTION_SHA256,
        },
        "variants": list(VARIANTS),
        "sampling": {
            "diffusion_steps": 500,
            "paired_initial_latent_and_posterior_noise": True,
            "same_exogenous_conditions_and_window_order": True,
            "path_local_generated_history_restoration": True,
            "causal_window_overlap": (
                "previous sampled tail [start+42,start+45] equals next "
                "history [next_start,next_start+3]"
            ),
            "history_restoration": True,
            "global_bps": "recomputed_from_each_generated_object_reference",
            "relation_source": "current diffusion state x_t only",
            "relation_builder_shared_with_training": True,
            "future_gt": False,
            "previous_predicted_x0_relation": False,
            "stored_relation": False,
            "scene": False,
            "cfg": False,
            "guidance": False,
            "dynamic_perception": False,
            "generator_draws_per_window": {
                "initial_latent": 1,
                "posterior_noise": 499,
                "timestep_zero_noise": "zeros_without_generator_draw",
            },
        },
        "relation": {
            "rest_object_points": [100, 3],
            "temporal_anchors": list(TEMPORAL_ANCHORS),
            "roles": list(ROLE_NAMES),
            "gate_ablated": "force tanh(alpha)=0 at every model call",
            "temporal_correspondence_permuted": "geometry slot k receives (k+2) mod 4",
            "left_right_role_swapped": "swap pooled left/right blocks before projection",
            "capture": [
                "pooled_block_norm",
                "pooled_block_variance",
                "relation_norm",
                "temporal_permutation_sensitivity",
                "role_swap_sensitivity",
                "gate",
            ],
            "selection_use": False,
        },
        "metrics": {
            "semantic_contact_units": ["left_hand", "right_hand", "union"],
            "semantic_thresholds": [0.5, 0.75, 0.95],
            "physical_thresholds_cm": list(PHYSICAL_THRESHOLDS_CM),
            "direct_hand_indices": list(DIRECT_HAND_INDICES),
            "fk_palm_indices": list(FK_PALM_INDICES),
            "gt_contact_frame_definition": (
                "target direct-hand union physical distance below 5cm"
            ),
            "gt_contact_distance_finite_mask": (
                "fixed target-derived sequence mask; no missing-value imputation"
            ),
            "gt_contact_distance_finite_sequence_count": (
                GT_CONTACT_FINITE_SEQUENCE_COUNT
            ),
            "gt_contact_distance_finite_sequence_names_sha256": (
                GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256
            ),
            "penetration_zero_denominator": (
                "undefined ratio plus unchanged paired absolute difference"
            ),
            "paired_unit": "sequence",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "assets": {
            "data_contract_sha256": base.EXPECTED_DATA_CONTRACT_SHA256,
            "split_sha256": base.EXPECTED_SPLIT_SHA256,
            "normalization_sha256": base.EXPECTED_NORMALIZATION_SHA256,
            "bps_sha256": base.BPS_SHA256,
            "sparse_mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
            "sparse_manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
            "sparse_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        },
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_writes": 0,
        "checkpoint_selection": False,
        "official_test_used": False,
        "output_dir": str(args.output_dir.resolve()),
        "metrics_path": str(args.metrics.resolve()),
    }


class RelationCapture:
    """Accumulate the small diagnostic snapshots on their source device."""

    EXPECTED_SHAPES = {
        "pooled_block_norm": (4, 3),
        "pooled_block_variance": (4, 3),
        "relation_norm": (4,),
        "temporal_permutation_sensitivity": (4,),
        "role_swap_sensitivity": (4,),
        "gate": (1,),
    }

    def __init__(self) -> None:
        self.sums: Dict[str, torch.Tensor] = {}
        self.finite: torch.Tensor | None = None
        self.calls = 0

    def hook(self, module, inputs, output) -> None:
        del inputs, output
        snapshot = module.snapshot()
        if snapshot is None or set(snapshot) != set(self.EXPECTED_SHAPES):
            raise RuntimeError("D2-AE sparse relation capture is incomplete")
        for key, shape in self.EXPECTED_SHAPES.items():
            value = snapshot[key]
            if tuple(value.shape) != shape:
                raise ValueError(f"D2-AE relation snapshot {key} is invalid")
            value = value.detach().to(dtype=torch.float64)
            if key not in self.sums:
                self.sums[key] = torch.zeros_like(value)
            if self.finite is None:
                self.finite = torch.ones((), dtype=torch.bool, device=value.device)
            self.finite.logical_and_(torch.isfinite(value).all())
            self.sums[key].add_(value)
        self.calls += 1

    def result(self) -> Dict[str, object]:
        if self.calls != 500:
            raise ValueError(f"D2-AE expected 500 relation calls, got {self.calls}")
        if self.finite is None or not bool(self.finite.detach().cpu()):
            raise ValueError("D2-AE relation capture contains non-finite values")
        return {
            "forward_calls": self.calls,
            "axis": {
                "temporal_anchors": list(TEMPORAL_ANCHORS),
                "roles": list(ROLE_NAMES),
            },
            "values": {
                key: (value / self.calls).detach().cpu().tolist()
                for key, value in sorted(self.sums.items())
            },
        }


def _conditioning_hashes(values: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    return {
        "shapes": {key: list(value.shape) for key, value in sorted(values.items())},
        "sha256": {key: tensor_sha256(value) for key, value in sorted(values.items())},
    }


def causal_overlap_contract(
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
) -> Dict[str, object]:
    """Prove that the sealed three-window cohort is one causal rollout.

    Each 48-source-frame window is sampled every three frames into the 16-frame
    prior representation.  A 42-source-frame shift therefore makes the prior
    window's last two sampled frames exactly the next window's two history
    frames.
    """
    rows = []
    for cohort_index, triple in enumerate(triples):
        if len(triple) != base.WINDOWS_PER_SEQUENCE:
            raise ValueError("D2-AE causal cohort must contain three windows per sequence")
        global_indices = [int(dataset.indices[int(position)]) for position in triple]
        sequence_ids = [int(dataset.sequence_ids[index]) for index in global_indices]
        phase_offsets = [int(dataset.language["pi"][index]) for index in global_indices]
        starts = [int(dataset.starts[index]) for index in global_indices]
        ends = [int(dataset.ends[index]) for index in global_indices]
        sampled_frames = [
            list(range(start, end, 3))
            for start, end in zip(starts, ends)
        ]
        start_offsets = [start - starts[0] for start in starts]
        overlap = [
            {
                "previous_tail": sampled_frames[step][-2:],
                "next_history": sampled_frames[step + 1][:2],
                "exact": sampled_frames[step][-2:] == sampled_frames[step + 1][:2],
            }
            for step in range(base.WINDOWS_PER_SEQUENCE - 1)
        ]
        checks = {
            "single_sequence": len(set(sequence_ids)) == 1,
            "phase_offsets": tuple(phase_offsets) == tuple(COHORT_PHASE_OFFSETS),
            "source_window_lengths": all(
                end - start == 48 for start, end in zip(starts, ends)
            ),
            "sampled_window_lengths": all(
                len(frames) == 16 for frames in sampled_frames
            ),
            "rollout_offsets": tuple(start_offsets) == tuple(PRIOR_ROLLOUT_OFFSETS),
            "history_overlap": all(item["exact"] for item in overlap),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"D2-AE causal overlap mismatch for cohort row {cohort_index}: {failed}"
            )
        rows.append({
            "cohort_index": cohort_index,
            "sequence": str(dataset.scene_names[sequence_ids[0]]),
            "global_indices": global_indices,
            "phase_offsets": phase_offsets,
            "source_starts": starts,
            "source_ends": ends,
            "source_start_offsets": start_offsets,
            "sampled_tail_to_next_history": overlap,
            "checks": checks,
        })
    return {
        "schema_version": 1,
        "phase_offsets": list(COHORT_PHASE_OFFSETS),
        "prior_rollout_offsets": list(PRIOR_ROLLOUT_OFFSETS),
        "source_window_frames": 48,
        "subsample_stride": 3,
        "model_window_frames": 16,
        "history_frames": 2,
        "sequences": len(rows),
        "all_exact": True,
        "rows": rows,
    }


@torch.no_grad()
def rollout_chunk(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
    rest_vertices: Mapping[str, torch.Tensor],
    parents_24: torch.Tensor,
    *,
    chunk_index: int,
) -> Dict[str, object]:
    positions_by_step = [
        [triple[step] for triple in triples]
        for step in range(base.WINDOWS_PER_SEQUENCE)
    ]
    items_by_step = [
        [dataset[position] for position in positions]
        for positions in positions_by_step
    ]
    names = [
        base._sequence_name(dataset, position)
        for position in positions_by_step[0]
    ]
    first_items = items_by_step[0]
    frame = base.stack_frames(first_items, device)
    fixed = torch.stack([item["x"][:2] for item in first_items]).to(device)
    decoded_steps = []
    fk_steps = []
    noise_streams = []
    conditioning_streams = []
    relation_windows = []
    history_max_abs = 0.0
    position_minimum = dataset.codec.position_minimum.to(device=device)
    position_maximum = dataset.codec.position_maximum.to(device=device)
    object_minimum = dataset.codec.object_minimum.to(device=device)
    object_maximum = dataset.codec.object_maximum.to(device=device)
    relation_module = model.network.sparse_relation_field
    if relation_module is None:
        raise ValueError("D2-AE internal model has no sparse relation field")

    for window_index in range(base.WINDOWS_PER_SEQUENCE):
        items = items_by_step[window_index]
        gt_frame = base.stack_frames(items, device)
        pelvis_global, object_global = base.global_goals(
            dataset, items, gt_frame, device,
        )
        goals = torch.zeros(len(triples), 9, device=device)
        goals[:, :3] = dataset.codec.pelvis_goal(pelvis_global, frame)
        goals[:, 6:9] = dataset.codec.object_goal(object_global, frame)
        text = torch.stack([item["text_embedding"] for item in items]).to(device)
        bps = base.current_bps(
            dataset, frame.object_reference, names, rest_vertices,
        )
        raw_progress = torch.stack([item["progress"] for item in items]).to(device)
        progress = normalize_progress(raw_progress)
        rest_offsets = torch.stack([
            item["rest_human_offsets"] for item in items
        ]).to(device)
        rest_object_points = torch.stack([
            item["rest_object_points"] for item in items
        ]).to(device=device, dtype=torch.float32)
        relation_arguments = {
            "rest_object_points": rest_object_points,
            "world_to_local_rotation": frame.world_to_local,
            "object_rotation_reference": frame.object_reference,
            "position_minimum": position_minimum,
            "position_maximum": position_maximum,
            "object_minimum": object_minimum,
            "object_maximum": object_maximum,
        }
        label = sampler_seed_label(chunk_index, window_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(label))
        initial_state = base.sha256_tensor_state(generator.get_state())
        capture = RelationCapture()
        model.network.set_sparse_relation_capture(True)
        hook = relation_module.register_forward_hook(capture.hook)
        try:
            sample = diffusion.sample(
                model,
                fixed,
                text,
                bps,
                goals,
                progress,
                **relation_arguments,
                generator=generator,
            )
        finally:
            hook.remove()
            model.network.set_sparse_relation_capture(False)
        final_state = base.sha256_tensor_state(generator.get_state())
        relation_value = capture.result()
        relation_value.update({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "metadata": {
                "rest_object_points_shape": list(rest_object_points.shape),
                "world_to_local_rotation_shape": list(frame.world_to_local.shape),
                "object_rotation_reference_shape": list(frame.object_reference.shape),
                "device": str(rest_object_points.device),
                "dtype": str(rest_object_points.dtype),
                "finite": bool(
                    all(torch.isfinite(value).all() for value in relation_arguments.values())
                ),
            },
        })
        relation_windows.append(relation_value)
        conditioning_streams.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "path_local_provenance": {
                "fixed_history_source": (
                    "immutable_selected_window_history"
                    if window_index == 0
                    else "previous_generated_tail_from_same_variant"
                ),
                "frame_source": (
                    "immutable_selected_window_frame"
                    if window_index == 0
                    else "previous_generated_tail_from_same_variant"
                ),
                "global_bps_reference": "same_path_local_frame.object_reference",
                "local_goal_reference": "same_path_local_frame",
                "relation_rotation_reference": "same_path_local_frame",
                "intervention_scope_per_model_call": (
                    "gate_or_temporal_geometry_blocks_or_left_right_pooled_blocks_only"
                ),
            },
            "exogenous": _conditioning_hashes({
                "text": text,
                "raw_progress": raw_progress,
                "global_pelvis_goal": pelvis_global,
                "global_object_goal": object_global,
                "rest_object_points": rest_object_points,
            }),
            "path_local_model_inputs": _conditioning_hashes({
                "fixed_history": fixed,
                "global_bps": bps,
                "local_goals": goals,
                "normalized_progress": progress,
                **relation_arguments,
            }),
        })
        noise_streams.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "label": label,
            "seed": stable_seed(label),
            "generator_initial_state_sha256": initial_state,
            "generator_final_state_sha256": final_state,
            "draw_contract": {
                "initial_latent_draws": 1,
                "posterior_noise_draws": 499,
                "total_generator_draws": 500,
                "draw_shape": [len(triples), 16, 232],
                "timestep_zero_noise": "zeros_without_generator_draw",
            },
        })
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        history_max_abs = max(
            history_max_abs,
            float((sample[:, :2] - fixed).abs().max().detach().cpu()),
        )
        decoded = dataset.codec.decode(sample, frame)
        fk = base.decoded_fk_positions(decoded, rest_offsets, parents_24)
        decoded_steps.append({
            key: value.detach().cpu()
            for key, value in decoded.items()
        })
        fk_steps.append(fk.detach().cpu())
        if window_index < base.WINDOWS_PER_SEQUENCE - 1:
            fixed, frame = dataset.codec.encode(
                decoded["joints"][:, -2:],
                decoded["human_rotation"][:, -2:],
                global_object_translation=decoded["object_translation"][:, -2:],
                global_object_rotation=decoded["object_rotation"][:, -2:],
                contact=decoded["contact"][:, -2:],
            )

    generated = []
    for row in range(len(triples)):
        value = {
            key: torch.cat([
                decoded_steps[step][key][row, 2:]
                for step in range(base.WINDOWS_PER_SEQUENCE)
            ])
            for key in (
                "joints",
                "human_rotation",
                "object_translation",
                "object_rotation",
                "contact",
            )
        }
        value["fk_joints"] = torch.cat([
            fk_steps[step][row, 2:]
            for step in range(base.WINDOWS_PER_SEQUENCE)
        ])
        generated.append(value)
    return {
        "generated": generated,
        "decoded_steps": decoded_steps,
        "noise_streams": noise_streams,
        "conditioning_streams": conditioning_streams,
        "relation_windows": relation_windows,
        "history_max_abs": history_max_abs,
    }


def analyze_sequence(
    target: Mapping[str, object],
    generated: Mapping[str, torch.Tensor],
    rest_vertices: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    penetration: Mapping[str, object],
) -> Dict[str, object]:
    category = str(target["object_category"])
    target_vertices = rest_vertices[category].to(device)[None] @ (
        target["object_rotation"].to(device).transpose(-1, -2)
    ) + target["object_translation"].to(device)[:, None]
    predicted_vertices = rest_vertices[category].to(device)[None] @ (
        project_to_so3(generated["object_rotation"].to(device)).transpose(-1, -2)
    ) + generated["object_translation"].to(device)[:, None]
    target_direct = base.hand_distances(
        target["joints"].to(device), target_vertices, DIRECT_HAND_INDICES,
    ).cpu().numpy().astype(np.float64)
    predicted_direct = base.hand_distances(
        generated["joints"].to(device), predicted_vertices, DIRECT_HAND_INDICES,
    ).cpu().numpy().astype(np.float64)
    target_fk = base.hand_distances(
        target["fk_joints"].to(device), target_vertices, FK_PALM_INDICES,
    ).cpu().numpy().astype(np.float64)
    predicted_fk = base.hand_distances(
        generated["fk_joints"].to(device), predicted_vertices, FK_PALM_INDICES,
    ).cpu().numpy().astype(np.float64)
    target_contact = target["contact"].cpu().numpy().astype(np.float64)
    predicted_contact = generated["contact"].cpu().numpy().astype(np.float64)
    return {
        "sequence": target["sequence"],
        "sequence_index": int(target["sequence_index"]),
        "object_category": category,
        "positions": target["positions"],
        "pi": target["pi"],
        "semantic_vs_gt": base.semantic_report(predicted_contact, target_contact),
        "direct_physical_geometry_vs_gt": base.geometry_report(
            predicted_direct, target_direct,
        ),
        "fk_physical_geometry_vs_gt": base.geometry_report(
            predicted_fk, target_fk,
        ),
        "gt_contact_frame_direct_distance": base.gt_contact_frame_distance(
            predicted_direct, target_direct,
        ),
        "penetration": dict(penetration),
        "per_frame": {
            "target_contact": target_contact.tolist(),
            "predicted_contact": predicted_contact.tolist(),
            "target_direct_hand_object_distance_m": target_direct.tolist(),
            "predicted_direct_hand_object_distance_m": predicted_direct.tolist(),
            "target_fk_hand_object_distance_m": target_fk.tolist(),
            "predicted_fk_hand_object_distance_m": predicted_fk.tolist(),
        },
    }


def variant_complete(
    records: Sequence[Mapping[str, object]],
    kinematics: Mapping[str, object],
) -> bool:
    return bool(
        len(records) == 64
        and all(
            base.reports_complete({
                "semantic_vs_gt": record["semantic_vs_gt"],
                "fk_physical_geometry_vs_gt": record[
                    "fk_physical_geometry_vs_gt"
                ],
                "direct_physical_geometry_vs_gt": record[
                    "direct_physical_geometry_vs_gt"
                ],
            })
            for record in records
        )
        and set(kinematics["aggregate"]) >= {
            "object_goal_error_cm",
            "pelvis_goal_error_cm",
            "mpjpe_cm",
            "foot_sliding",
        }
    )


def aggregate_relation_windows(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not records:
        raise ValueError("D2-AE relation appendix is empty")
    keys = set(records[0]["values"])
    if any(set(record["values"]) != keys for record in records):
        raise ValueError("D2-AE relation appendix keys differ")
    return {
        "window_records": len(records),
        "forward_calls_per_window": 500,
        "axis": records[0]["axis"],
        "values": {
            key: np.mean(
                np.asarray([record["values"][key] for record in records], dtype=np.float64),
                axis=0,
            ).tolist()
            for key in sorted(keys)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AE internal run id must use the locked stem and actual date"
        )
    if not TRAINING_RUN_ID_RE.fullmatch(args.training_run_id):
        raise ValueError("invalid D2-AE formal training run id")
    if not re.fullmatch(r"[0-9a-f]{64}", args.target_sha256):
        raise ValueError("D2-AE target SHA-256 must be lowercase hexadecimal")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-AE internal batch size must evenly divide 64")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(sys.executable).resolve() != Path(configured_python).resolve()
    ):
        raise ValueError("D2-AE internal requires the absolute INFBAGEL_PYTHON")
    config = resolved_config(args)
    if args.resolve_only:
        base.exclusive_json(args.resolved_config.resolve(), config)
        return
    if (
        os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi"
        or socket.gethostname() != "node01"
    ):
        raise RuntimeError("D2-AE internal is restricted to the HOI worker")
    if base.git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-AE internal refuses a dirty worker checkout")
    if json.loads(args.resolved_config.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-AE internal runtime differs from archived config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-AE internal requires worker CUDA")
    if args.output_dir.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir.resolve()}")
    if args.metrics.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.metrics.resolve()}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    author_utils.SMPL_DIR = str((REPO / "smpl_models").resolve())
    try:
        checkpoint = checkpoint_contract(
            args.target_checkpoint.resolve(),
            args.target_sha256,
            args.training_run_id,
        )
        asset_hashes = {
            "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
            "bps": sha256_file((REPO / "code/bps.pt").resolve()),
            "split": sha256_file(
                REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
            "sparse_mapping": SPARSE_POINT_MAPPING_SHA256,
            "sparse_manifest": SPARSE_POINT_MANIFEST_SHA256,
            "sparse_tensor": SPARSE_POINT_TENSOR_SHA256,
        }
        if asset_hashes != {
            "normalization": base.EXPECTED_NORMALIZATION_SHA256,
            "bps": base.BPS_SHA256,
            "split": base.EXPECTED_SPLIT_SHA256,
            "sparse_mapping": SPARSE_POINT_MAPPING_SHA256,
            "sparse_manifest": SPARSE_POINT_MANIFEST_SHA256,
            "sparse_tensor": SPARSE_POINT_TENSOR_SHA256,
        }:
            raise ValueError(f"D2-AE internal asset hash mismatch: {asset_hashes}")

        base.seed_everything(42)
        dataset = PriorWindowDataset(
            str(REPO),
            "hoi",
            partition="internal_validation",
            split_manifest=(
                "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
        )
        selection = base.select_contact_holdout(dataset)
        if (
            selection["sha256"] != SELECTION_SHA256
            or selection["sequences"] != 64
            or selection["windows"] != 192
            or selection["phase_offsets"] != [14, 56, 98]
        ):
            raise ValueError("D2-AE internal selection contract mismatch")
        triples = selection["triples"]
        causal_overlap = causal_overlap_contract(dataset, triples)
        causal_overlap_path = output_dir / "causal_window_overlap.json"
        base.exclusive_json(causal_overlap_path, {
            **causal_overlap,
            "run_id": args.run_id,
        })
        parents_24 = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy()
        ).long().to(device)
        parents_22 = torch.from_numpy(
            get_smpl_parents(use_joints24=False).copy()
        ).long().to(device)
        targets = base.prepare_targets(dataset, triples, parents_24.cpu())
        for target, triple in zip(targets, triples):
            target["sequence_index"] = int(
                dataset[triple[0]]["sequence_index"].item()
            )
        rest_vertices = base.load_rest_vertices(dataset, triples, device)
        penetration_assets = base.load_penetration_assets(REPO)
        betas = np.load(REPO / "data/train/betas.npy", mmap_mode="r")
        translations = np.load(
            REPO / "data/train/transl_aligned.npy", mmap_mode="r"
        )
        with (REPO / "data/train/gender.pkl").open("rb") as handle:
            genders = pickle.load(handle)
        smpl_cache: Dict[str, torch.nn.Module] = {}
        diffusion = GaussianDiffusion(500).to(device)
        model, metadata = load_trained_hoi_prior(
            str(args.target_checkpoint.resolve()),
            device,
            weight_variant="online",
            expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
        )
        if metadata["data_contract_sha256"] != base.EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError("D2-AE internal checkpoint data-contract mismatch")
        model.eval()
        model_before = state_dict_sha256(model)

        variants: Dict[str, object] = {}
        records_by_variant: Dict[str, List[Dict[str, object]]] = {}
        noise_by_variant: Dict[str, object] = {}
        conditioning_by_variant: Dict[str, object] = {}
        relation_by_variant: Dict[str, object] = {}
        for variant in VARIANTS:
            model.network.set_sparse_relation_diagnostic_variant(variant)
            model.network.set_sparse_relation_gate_override(None)
            records: List[Dict[str, object]] = []
            decoded_chunks = []
            noise_streams = []
            conditioning_streams = []
            relation_windows = []
            history_max_abs = 0.0
            for chunk_index, offset in enumerate(
                range(0, len(triples), args.batch_size)
            ):
                selected = triples[offset:offset + args.batch_size]
                rollout = rollout_chunk(
                    model,
                    diffusion,
                    dataset,
                    selected,
                    device,
                    rest_vertices,
                    parents_24,
                    chunk_index=chunk_index,
                )
                decoded_chunks.append(rollout["decoded_steps"])
                noise_streams.extend(rollout["noise_streams"])
                conditioning_streams.extend(rollout["conditioning_streams"])
                relation_windows.extend(rollout["relation_windows"])
                history_max_abs = max(
                    history_max_abs, float(rollout["history_max_abs"])
                )
                for target, generated in zip(
                    targets[offset:offset + args.batch_size],
                    rollout["generated"],
                ):
                    penetration = base.sequence_penetration(
                        generated,
                        sequence_index=int(target["sequence_index"]),
                        object_name=str(target["object_category"]),
                        device=device,
                        parents_22=parents_22,
                        betas=betas,
                        genders=genders,
                        translations=translations,
                        penetration_assets=penetration_assets,
                        smpl_cache=smpl_cache,
                    )
                    records.append(analyze_sequence(
                        target,
                        generated,
                        rest_vertices,
                        device,
                        penetration=penetration,
                    ))
            decoded_steps = base.concatenate_decoded_steps(decoded_chunks, device)
            kinematics = base.physical_summary(
                dataset, triples, decoded_steps, device,
            )
            sequence_names = [str(record["sequence"]) for record in records]
            mapped_kinematics = base.kinematics_by_sequence(
                kinematics, sequence_names,
            )
            for record in records:
                record["kinematics"] = mapped_kinematics[str(record["sequence"])]
            semantic_geometry = base._summary_for_records(
                records, include_categories=True,
            )
            penetration_summary = base.aggregate_penetration(records)
            relation_summary = aggregate_relation_windows(relation_windows)
            finite = bool(
                base.all_finite(semantic_geometry)
                and base.all_finite(kinematics)
                and base.all_finite(relation_summary)
                and all(
                    all(
                        value is None or math.isfinite(float(value))
                        for key, value in record["penetration"].items()
                        if key not in {"finite", "excluded_by_official_contract"}
                    )
                    for record in records
                )
            )
            complete = variant_complete(records, kinematics)
            variant_value = {
                "variant": variant,
                "history_max_abs": history_max_abs,
                "aggregate": {
                    "semantic_and_geometry": semantic_geometry,
                    "kinematics": kinematics["aggregate"],
                    "penetration": penetration_summary,
                    "sparse_relation": relation_summary,
                },
                "kinematics_full": kinematics,
                "per_sequence": records,
                "noise_streams": noise_streams,
                "conditioning_streams": conditioning_streams,
                "relation_windows": relation_windows,
                "finite": finite,
                "all_fields_reported": complete,
            }
            variant_path = output_dir / f"{variant}.json"
            base.exclusive_json(variant_path, variant_value)
            variants[variant] = {
                "artifact": {
                    "path": str(variant_path),
                    "sha256": sha256_file(variant_path),
                    "bytes": variant_path.stat().st_size,
                },
                "history_max_abs": history_max_abs,
                "aggregate": variant_value["aggregate"],
                "finite": finite,
                "all_fields_reported": complete,
            }
            records_by_variant[variant] = records
            noise_by_variant[variant] = noise_streams
            conditioning_by_variant[variant] = conditioning_streams
            relation_by_variant[variant] = {
                "aggregate": relation_summary,
                "per_window": relation_windows,
            }
            torch.cuda.empty_cache()

        model.network.set_sparse_relation_diagnostic_variant("full")
        model.network.set_sparse_relation_gate_override(None)
        model.network.set_sparse_relation_capture(False)
        model_after = state_dict_sha256(model)
        comparisons = paired_comparisons(records_by_variant)
        finite_masks = [
            comparisons[f"full_vs_{variant}"][
                "other_minus_full_gt_contact_distance_cm"
            ]["finite_sequence_names"]
            for variant in VARIANTS[1:]
        ]
        gt_contact_mask_exact = bool(
            all(mask == finite_masks[0] for mask in finite_masks[1:])
            and len(finite_masks[0]) == GT_CONTACT_FINITE_SEQUENCE_COUNT
            and base.sequence_names_sha256(finite_masks[0])
            == GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256
        )
        paired_noise_identity = all(
            noise_by_variant[variant] == noise_by_variant["full"]
            for variant in VARIANTS[1:]
        )
        paired_exogenous_identity = all(
            [
                {
                    "chunk_index": row["chunk_index"],
                    "window_index": row["window_index"],
                    "exogenous": row["exogenous"],
                }
                for row in conditioning_by_variant[variant]
            ]
            == [
                {
                    "chunk_index": row["chunk_index"],
                    "window_index": row["window_index"],
                    "exogenous": row["exogenous"],
                }
                for row in conditioning_by_variant["full"]
            ]
            for variant in VARIANTS[1:]
        )
        initial_history_identity = all(
            [
                row["path_local_model_inputs"]["sha256"]["fixed_history"]
                for row in conditioning_by_variant[variant]
                if int(row["window_index"]) == 0
            ]
            == [
                row["path_local_model_inputs"]["sha256"]["fixed_history"]
                for row in conditioning_by_variant["full"]
                if int(row["window_index"]) == 0
            ]
            for variant in VARIANTS[1:]
        )
        path_local_provenance_exact = all(
            row["path_local_provenance"]["fixed_history_source"]
            == (
                "immutable_selected_window_history"
                if int(row["window_index"]) == 0
                else "previous_generated_tail_from_same_variant"
            )
            and row["path_local_provenance"]["frame_source"]
            == (
                "immutable_selected_window_frame"
                if int(row["window_index"]) == 0
                else "previous_generated_tail_from_same_variant"
            )
            and row["path_local_provenance"]["global_bps_reference"]
            == "same_path_local_frame.object_reference"
            and row["path_local_provenance"]["local_goal_reference"]
            == "same_path_local_frame"
            and row["path_local_provenance"]["relation_rotation_reference"]
            == "same_path_local_frame"
            for variant in VARIANTS
            for row in conditioning_by_variant[variant]
        )
        generator_draw_contract_exact = all(
            row["draw_contract"] == {
                "initial_latent_draws": 1,
                "posterior_noise_draws": 499,
                "total_generator_draws": 500,
                "draw_shape": [
                    len(triples[offset:offset + args.batch_size]), 16, 232,
                ],
                "timestep_zero_noise": "zeros_without_generator_draw",
            }
            for variant in VARIANTS
            for offset, row in zip(
                (
                    offset
                    for offset in range(0, len(triples), args.batch_size)
                    for _ in range(base.WINDOWS_PER_SEQUENCE)
                ),
                noise_by_variant[variant],
            )
        )
        paired_noise_path = output_dir / "paired_noise.json"
        base.exclusive_json(paired_noise_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "shared": paired_noise_identity,
            "variants": noise_by_variant,
        })
        conditioning_path = output_dir / "paired_conditioning.json"
        base.exclusive_json(conditioning_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "shared_exogenous": paired_exogenous_identity,
            "shared_initial_history": initial_history_identity,
            "later_model_inputs": "path-local after causal rollout divergence",
            "path_local_provenance_exact": path_local_provenance_exact,
            "variants": conditioning_by_variant,
        })
        relation_path = output_dir / "sparse_relation_appendix.json"
        base.exclusive_json(relation_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "selection_use": False,
            "temporal_anchors": list(TEMPORAL_ANCHORS),
            "roles": list(ROLE_NAMES),
            "variants": relation_by_variant,
        })

        sampler_source = inspect.getsource(rollout_chunk)
        diffusion_source = inspect.getsource(GaussianDiffusion.sample)
        contract = {
            "checkpoint_contract": all(checkpoint["checks"].values()),
            "checkpoint_architecture_variant": (
                metadata["architecture_variant"] == HOI_ARCHITECTURE_D2AE
            ),
            "asset_hashes_exact": True,
            "selection_exact": True,
            "causal_window_overlap_exact": causal_overlap["all_exact"] is True,
            "gt_contact_finite_mask_exact": gt_contact_mask_exact,
            "paired_noise_identity": paired_noise_identity,
            "generator_draw_contract_exact": generator_draw_contract_exact,
            "paired_exogenous_condition_identity": paired_exogenous_identity,
            "paired_initial_history_identity": initial_history_identity,
            "path_local_condition_provenance": path_local_provenance_exact,
            "history_restoration": all(
                float(variants[variant]["history_max_abs"]) <= HISTORY_MAX_ABS
                for variant in VARIANTS
            ),
            "all_variants_finite": all(
                bool(variants[variant]["finite"]) for variant in VARIANTS
            ),
            "all_fields_reported": all(
                bool(variants[variant]["all_fields_reported"])
                for variant in VARIANTS
            ),
            "relation_capture_complete": all(
                int(window["forward_calls"]) == 500
                and bool(window["metadata"]["finite"])
                and str(window["metadata"]["device"]).startswith("cuda")
                for variant in VARIANTS
                for window in relation_by_variant[variant]["per_window"]
            ),
            "model_state_unchanged": model_before == model_after,
            "parameter_grad_buffers_clear": all(
                parameter.grad is None for parameter in model.parameters()
            ),
            "relation_capture_descriptive_only": True,
            "penetration_zero_denominator_explicit": True,
            "current_state_relation_metadata_forwarded": all(
                name in diffusion_source
                for name in (
                    "rest_object_points",
                    "world_to_local_rotation",
                    "object_rotation_reference",
                    "position_minimum",
                    "position_maximum",
                    "object_minimum",
                    "object_maximum",
                )
            ),
            "sampler_future_gt_absent": "future_gt" not in sampler_source,
            "sampler_previous_x0_relation_absent": "previous_x0" not in sampler_source,
            "sampler_scene_absent": "Scene" not in sampler_source,
            "sampler_stored_relation_absent": "stored_relation" not in sampler_source,
            "sampler_cpu_dynamic_geometry_absent": all(
                token not in sampler_source
                for token in ("cKDTree", "scipy", "cdist", "full_mesh")
            ),
            "global_bps_recomputed_unchanged": "base.current_bps(" in sampler_source,
            "optimizer_absent": True,
            "checkpoint_write_absent": True,
            "official_test_absent": True,
        }
        decision = internal_mechanism_gate(contract, comparisons)
        relation_module = model.network.sparse_relation_field
        assert relation_module is not None
        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "completed",
            "seed": 42,
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "selection": {
                key: value
                for key, value in selection.items()
                if key != "triples"
            },
            "target_checkpoint": checkpoint,
            "checkpoint_metadata": metadata,
            "learned_sparse_relation": {
                "alpha": float(relation_module.alpha.detach().cpu()),
                "gate": float(torch.tanh(relation_module.alpha.detach()).cpu()),
                "contract": relation_module.contract_metadata(),
            },
            "assets": {
                **asset_hashes,
                "data_contract_sha256": base.EXPECTED_DATA_CONTRACT_SHA256,
                "penetration_hand_vertex_ids_sha256": penetration_assets[
                    "hand_ids_sha256"
                ],
            },
            "variants": variants,
            "comparisons": comparisons,
            "contract": contract,
            "decision": decision,
            "paired_noise": {
                "path": str(paired_noise_path),
                "sha256": sha256_file(paired_noise_path),
            },
            "paired_conditioning": {
                "path": str(conditioning_path),
                "sha256": sha256_file(conditioning_path),
            },
            "causal_window_overlap": {
                "path": str(causal_overlap_path),
                "sha256": sha256_file(causal_overlap_path),
                "all_exact": causal_overlap["all_exact"],
            },
            "sparse_relation_appendix": {
                "path": str(relation_path),
                "sha256": sha256_file(relation_path),
                "selection_use": False,
            },
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_writes": 0,
            "checkpoint_selection": False,
            "consistency_started": False,
            "official_test_used": False,
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
        base.exclusive_json(args.metrics.resolve(), result)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "failed",
            "seed": 42,
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "classification": FAILURE_CLASSIFICATION,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_writes": 0,
            "checkpoint_selection": False,
            "consistency_started": False,
            "official_test_used": False,
        }
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            base.exclusive_json(failure_path, failure)
        if not args.metrics.resolve().exists():
            base.exclusive_json(args.metrics.resolve(), failure)
        raise


if __name__ == "__main__":
    main()
