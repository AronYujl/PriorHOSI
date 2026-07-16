#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-Q0 contact-guidance counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from guidance_loss import apply_hand_object_interaction_guidance_loss  # noqa: E402
from priors.contact_alignment import (  # noqa: E402
    PHYSICAL_THRESHOLDS_CM,
    all_finite,
    geometry_report,
    object_vertices,
    semantic_report,
)
from priors.contact_guidance import (  # noqa: E402
    AUTHOR_BLOB_SHA256,
    AUTHOR_COMMIT,
    AUTHOR_HAND_WEIGHT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIRECT_HAND_INDICES,
    EXPECTED_CHECKPOINT_SHA256,
    FK_PALM_INDICES,
    GUIDANCE_SCALE,
    HISTORY_MAX_ABS,
    KINEMATIC_METRICS,
    MODELS,
    PHASE_OFFSETS,
    REST_VERTEX_COUNT,
    RUN_ID,
    SELECTION_SHA256,
    SEMANTIC_THRESHOLD,
    SPATIAL_THRESHOLD_M,
    VARIANTS,
    WINDOWS_PER_SEQUENCE,
    decoded_fk_positions,
    deterministic_vertex_subset,
    hand_distances,
    mechanism_gate,
    author_hand_object_components,
    paired_guidance_comparison,
    sample_contact_counterfactual,
    sampler_seed_label,
    select_guidance_holdout,
    stable_seed,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import BPS_SHA256, project_to_so3  # noqa: E402
from tools.diagnose_hoi_remediation import (  # noqa: E402
    physical_summary,
    raw_window_target,
    seed_everything,
)
from tools.evaluate_hoi_remediation import (  # noqa: E402
    current_bps,
    global_goals,
    load_rest_vertices,
    stack_frames,
)


SUBPHASE = "1B-D2-Q0"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_NORMALIZATION_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
DEFAULT_BATCH_SIZE = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor_state(value: torch.Tensor) -> str:
    return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def author_blob_hashes() -> Dict[str, str]:
    result = {}
    for path in AUTHOR_BLOB_SHA256:
        payload = subprocess.check_output(
            ["git", "show", f"{AUTHOR_COMMIT}:{path}"],
            cwd=REPO,
        )
        result[path] = hashlib.sha256(payload).hexdigest()
    return result


def rest_mesh_contract() -> Dict[str, Dict[str, object]]:
    root = (REPO / "data/object/rest_object_geo").resolve()
    return {
        path.stem: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path.resolve()),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.glob("*.ply"))
    }


def author_formula_replay_max_abs(device: torch.device) -> float:
    human = (
        torch.arange(1 * 3 * 24 * 3, device=device, dtype=torch.float32)
        .reshape(1, 3, 24, 3)
        / 100.0
        + 1.0
    )
    vertices = (
        torch.arange(1 * 3 * 5 * 3, device=device, dtype=torch.float32)
        .reshape(1, 3, 5, 3)
        / 50.0
    )
    translation = torch.zeros(1, 3, 3, device=device)
    rotation = torch.eye(3, device=device).repeat(1, 3, 1, 1)
    contact = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        device=device,
    )
    expected = apply_hand_object_interaction_guidance_loss(
        human, vertices, translation, rotation, contact,
    )
    actual = author_hand_object_components(
        human, vertices, translation, rotation, contact,
    )["total"]
    return float((actual - expected).abs().detach().cpu())


def checkpoint_paths(args) -> Dict[str, Path]:
    return {
        "source": args.source_checkpoint.resolve(),
        "current": args.current_checkpoint.resolve(),
        "balanced": args.balanced_checkpoint.resolve(),
    }


def checkpoint_hashes(args) -> Dict[str, str]:
    return {
        "source": args.source_sha256,
        "current": args.current_sha256,
        "balanced": args.balanced_sha256,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    paths = checkpoint_paths(args)
    hashes = checkpoint_hashes(args)
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": SUBPHASE,
        "mode": "author-contact-guidance-paired-counterfactual",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(Path(sys.executable).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "checkpoints": {
            name: {
                "path": str(paths[name]),
                "sha256": hashes[name],
                "weight_variant": "online",
            }
            for name in MODELS
        },
        "primary_gate_checkpoint": "balanced",
        "selection": {
            "partition": "internal_validation",
            "phase_offsets": list(PHASE_OFFSETS),
            "sequences": 64,
            "windows_per_sequence": WINDOWS_PER_SEQUENCE,
            "windows": 192,
            "global_window_indices_sha256": SELECTION_SHA256,
        },
        "sampling": {
            "variants": list(VARIANTS),
            "diffusion_steps": 500,
            "condition_variant": "matched",
            "paired_initial_and_posterior_noise": True,
            "guidance_steps": list(range(499, 0, -1)),
            "step_zero_guidance": False,
            "guidance_scale": GUIDANCE_SCALE,
            "author_hand_weight": AUTHOR_HAND_WEIGHT,
            "semantic_channels": [0, 1],
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "spatial_threshold_m": SPATIAL_THRESHOLD_M,
            "fk_palm_indices": list(FK_PALM_INDICES),
            "direct_hand_indices": list(DIRECT_HAND_INDICES),
            "rest_vertex_count": REST_VERTEX_COUNT,
            "rest_vertex_sampling": "deterministic-uniform-index",
            "posterior_helper": "priors.diffusion.GaussianDiffusion.posterior_sample",
            "injection": (
                "x_prev += grad(-(10 * author_hand_object_core_loss), pred_x0)"
            ),
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
            "deviations": [
                "feet_floor_weight_500_omitted",
                "scene_and_penetration_terms_omitted",
                "deterministic_2048_vertex_surface_instead_of_random_10000",
                "codec_differentiable_so3_decode",
                "ddpm_500_step_checkpoint_instead_of_consistency_sampler",
            ],
        },
        "evaluation": {
            "physical_thresholds_cm": list(PHYSICAL_THRESHOLDS_CM),
            "units": ["left_hand", "right_hand", "union"],
            "kinematic_metrics": list(KINEMATIC_METRICS),
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


def _sequence_name(dataset, position: int) -> str:
    global_index = int(dataset.indices[position])
    sequence = int(dataset.sequence_ids[global_index])
    return str(dataset.scene_names[sequence])


def prepare_targets(
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    parents_24: torch.Tensor,
) -> List[Dict[str, object]]:
    result = []
    device = torch.device("cpu")
    for triple in triples:
        target_steps = []
        for position in triple:
            item = dataset[position]
            frame = stack_frames([item], device)
            decoded = dataset.codec.decode(item["x"][None], frame)
            rest_offsets = item["rest_human_offsets"][None]
            fk = decoded_fk_positions(decoded, rest_offsets, parents_24)
            raw = raw_window_target(dataset, position, device)
            target_steps.append({
                "joints": raw["joints"],
                "human_rotation": decoded["human_rotation"][0],
                "fk_joints": fk[0],
                "object_translation": raw["object_translation"],
                "object_rotation": raw["object_rotation"],
                "contact": raw["contact"],
            })
        name = _sequence_name(dataset, triple[0])
        result.append({
            "sequence": name,
            "object_category": name.split("_")[1],
            "positions": list(triple),
            "pi": [
                int(dataset.language["pi"][int(dataset.indices[position])])
                for position in triple
            ],
            "rest_human_offsets": dataset[triple[0]]["rest_human_offsets"],
            **{
                key: torch.cat([
                    value[key][REPRESENTATION.history_frames:]
                    for value in target_steps
                ])
                for key in (
                    "joints",
                    "human_rotation",
                    "fk_joints",
                    "object_translation",
                    "object_rotation",
                    "contact",
                )
            },
        })
    return result


def _rest_batch(
    names: Sequence[str],
    rest_subsets: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([
        rest_subsets[name.split("_")[1]].to(device)
        for name in names
    ])


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
    guided: bool,
) -> Dict[str, object]:
    positions_by_step = [
        [triple[step] for triple in triples]
        for step in range(WINDOWS_PER_SEQUENCE)
    ]
    items_by_step = [
        [dataset[position] for position in positions]
        for positions in positions_by_step
    ]
    names = [_sequence_name(dataset, position) for position in positions_by_step[0]]
    first_items = items_by_step[0]
    frame = stack_frames(first_items, device)
    fixed = torch.stack([item["x"][:2] for item in first_items]).to(device)
    decoded_steps = []
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
        text = torch.stack([item["text_embedding"] for item in items]).to(device)
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
        sample, guidance_audit = sample_contact_counterfactual(
            diffusion,
            model,
            fixed,
            text,
            bps,
            goals,
            progress,
            generator=generator,
            guided=guided,
            codec=dataset.codec,
            frame=frame,
            rest_human_offsets=rest_offsets,
            parents_24=parents_24,
            rest_vertices=surface,
        )
        final_state = sha256_tensor_state(generator.get_state())
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        history_max_abs = max(
            history_max_abs,
            float((sample[:, :2] - fixed).abs().max().detach().cpu()),
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
                global_object_translation=decoded["object_translation"][:, -2:],
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
        "noise_streams": noise_streams,
        "guidance_windows": guidance_windows,
        "history_max_abs": history_max_abs,
    }


def _frame_lists(
    *,
    target_contact: np.ndarray,
    predicted_contact: np.ndarray,
    target_fk_distance: np.ndarray,
    predicted_fk_distance: np.ndarray,
    target_direct_distance: np.ndarray,
    predicted_direct_distance: np.ndarray,
) -> Dict[str, object]:
    return {
        "target_contact": target_contact.tolist(),
        "predicted_contact": predicted_contact.tolist(),
        "target_fk_hand_object_distance_m": target_fk_distance.tolist(),
        "predicted_fk_hand_object_distance_m": predicted_fk_distance.tolist(),
        "target_direct_hand_object_distance_m": target_direct_distance.tolist(),
        "predicted_direct_hand_object_distance_m": predicted_direct_distance.tolist(),
    }


def analyze_generated_sequence(
    target: Mapping[str, object],
    generated: Mapping[str, torch.Tensor],
    rest_vertices: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    category = str(target["object_category"])
    target_rotation = target["object_rotation"].to(device)
    target_translation = target["object_translation"].to(device)
    predicted_rotation = generated["object_rotation"].to(device)
    predicted_translation = generated["object_translation"].to(device)
    target_vertices = object_vertices(
        rest_vertices[category], target_rotation, target_translation,
    )
    predicted_vertices = object_vertices(
        rest_vertices[category], predicted_rotation, predicted_translation,
    )
    target_fk_distance = hand_distances(
        target["fk_joints"].to(device), target_vertices, FK_PALM_INDICES,
    ).cpu().numpy().astype(np.float64)
    predicted_fk_distance = hand_distances(
        generated["fk_joints"].to(device), predicted_vertices, FK_PALM_INDICES,
    ).cpu().numpy().astype(np.float64)
    target_direct_distance = hand_distances(
        target["joints"].to(device), target_vertices, DIRECT_HAND_INDICES,
    ).cpu().numpy().astype(np.float64)
    predicted_direct_distance = hand_distances(
        generated["joints"].to(device), predicted_vertices, DIRECT_HAND_INDICES,
    ).cpu().numpy().astype(np.float64)
    target_contact = target["contact"].cpu().numpy().astype(np.float64)
    predicted_contact = generated["contact"].cpu().numpy().astype(np.float64)
    return {
        "sequence": target["sequence"],
        "object_category": category,
        "positions": target["positions"],
        "pi": target["pi"],
        "semantic_vs_gt": semantic_report(predicted_contact, target_contact),
        "fk_physical_geometry_vs_gt": geometry_report(
            predicted_fk_distance, target_fk_distance,
        ),
        "direct_physical_geometry_vs_gt": geometry_report(
            predicted_direct_distance, target_direct_distance,
        ),
        "per_frame": _frame_lists(
            target_contact=target_contact,
            predicted_contact=predicted_contact,
            target_fk_distance=target_fk_distance,
            predicted_fk_distance=predicted_fk_distance,
            target_direct_distance=target_direct_distance,
            predicted_direct_distance=predicted_direct_distance,
        ),
    }


def _concatenate_frames(
    records: Sequence[Mapping[str, object]],
    key: str,
) -> np.ndarray:
    return np.concatenate([
        np.asarray(record["per_frame"][key], dtype=np.float64)
        for record in records
    ])


def _summary_for_records(
    records: Sequence[Mapping[str, object]],
    *,
    include_categories: bool,
) -> Dict[str, object]:
    target_contact = _concatenate_frames(records, "target_contact")
    predicted_contact = _concatenate_frames(records, "predicted_contact")
    target_fk = _concatenate_frames(
        records, "target_fk_hand_object_distance_m",
    )
    predicted_fk = _concatenate_frames(
        records, "predicted_fk_hand_object_distance_m",
    )
    target_direct = _concatenate_frames(
        records, "target_direct_hand_object_distance_m",
    )
    predicted_direct = _concatenate_frames(
        records, "predicted_direct_hand_object_distance_m",
    )
    result = {
        "sequences": len(records),
        "frames": len(target_contact),
        "semantic_vs_gt": semantic_report(predicted_contact, target_contact),
        "fk_physical_geometry_vs_gt": geometry_report(
            predicted_fk, target_fk,
        ),
        "direct_physical_geometry_vs_gt": geometry_report(
            predicted_direct, target_direct,
        ),
        "by_object_category": {},
    }
    if include_categories:
        categories = sorted({
            str(record["object_category"]) for record in records
        })
        for category in categories:
            result["by_object_category"][category] = _summary_for_records(
                [
                    record for record in records
                    if record["object_category"] == category
                ],
                include_categories=False,
            )
    return result


def reports_complete(value: Mapping[str, object]) -> bool:
    expected_thresholds = {
        f"{threshold:g}" for threshold in PHYSICAL_THRESHOLDS_CM
    }
    return bool(
        set(value["semantic_vs_gt"]["per_channel"]) == {"0", "1", "2", "3"}
        and set(value["fk_physical_geometry_vs_gt"]["thresholds_cm"])
        == expected_thresholds
        and set(value["direct_physical_geometry_vs_gt"]["thresholds_cm"])
        == expected_thresholds
    )


def concatenate_decoded_steps(
    chunks: Sequence[Sequence[Mapping[str, torch.Tensor]]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    return [
        {
            key: torch.cat([
                chunk[step][key] for chunk in chunks
            ]).to(device)
            for key in chunks[0][step]
        }
        for step in range(WINDOWS_PER_SEQUENCE)
    ]


def guidance_audit_summary(
    windows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    guided = bool(windows and windows[0]["guided"])
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
    )
    aggregate = {}
    for name in scalar_names:
        values = [float(value[name]) for value in per_step]
        aggregate[name] = {
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return {
        "guided": guided,
        "windows": len(windows),
        "applied_steps": sum(int(value["applied_steps"]) for value in windows),
        "expected_applied_steps": len(windows) * 499 if guided else 0,
        "step_zero_guidance_applied": any(
            bool(value["step_zero_guidance_applied"]) for value in windows
        ),
        "finite": all(bool(value["finite"]) for value in windows),
        "aggregate": aggregate,
        "per_window": list(windows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--balanced-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-sha256", default=EXPECTED_CHECKPOINT_SHA256["source"],
    )
    parser.add_argument(
        "--current-sha256", default=EXPECTED_CHECKPOINT_SHA256["current"],
    )
    parser.add_argument(
        "--balanced-sha256", default=EXPECTED_CHECKPOINT_SHA256["balanced"],
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
        raise ValueError(f"D2-Q0 run id must be {RUN_ID}")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-Q0 batch size must evenly divide 64")
    if checkpoint_hashes(args) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("D2-Q0 requested checkpoint hashes differ from preregistration")
    config = resolved_config(args)
    config_path = args.resolved_config.resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-Q0 runtime arguments do not match archived resolved config")
    if Path(sys.executable).resolve() != Path(
        os.environ.get("INFBAGEL_PYTHON", ""),
    ).resolve():
        raise ValueError("D2-Q0 requires the absolute INFBAGEL_PYTHON interpreter")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-Q0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if git_output("status", "--porcelain"):
        raise RuntimeError("D2-Q0 refuses a dirty worker checkout")
    paths = checkpoint_paths(args)
    for name in MODELS:
        actual = sha256_file(paths[name])
        if actual != EXPECTED_CHECKPOINT_SHA256[name]:
            raise ValueError(f"D2-Q0 {name} checkpoint hash mismatch: {actual}")
    asset_hashes = {
        "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
        "bps": sha256_file((REPO / "code/bps.pt").resolve()),
    }
    expected_asset_hashes = {
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "bps": BPS_SHA256,
    }
    if asset_hashes != expected_asset_hashes:
        raise ValueError(
            f"D2-Q0 asset hash mismatch: {asset_hashes} != {expected_asset_hashes}"
        )
    actual_author_hashes = author_blob_hashes()
    if actual_author_hashes != AUTHOR_BLOB_SHA256:
        raise ValueError("D2-Q0 author blob hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-Q0 is a four-GPU-worker CUDA diagnostic")
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
    selection = select_guidance_holdout(dataset)
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
    models: Dict[str, object] = {}
    comparisons: Dict[str, object] = {}
    for name in MODELS:
        model, metadata = load_trained_hoi_prior(
            str(paths[name]), device, weight_variant="online",
        )
        if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError(f"D2-Q0 {name} data-contract mismatch")
        model.eval()
        model_before = state_dict_sha256(model)
        variants: Dict[str, object] = {}
        variant_records: Dict[str, List[Dict[str, object]]] = {}
        variant_kinematics: Dict[str, object] = {}
        for variant in VARIANTS:
            guided = variant == "guided"
            records = []
            decoded_chunks = []
            noise_streams = []
            guidance_windows = []
            history_max_abs = 0.0
            variant_error = None
            try:
                for chunk_index, offset in enumerate(
                    range(0, len(triples), args.batch_size)
                ):
                    selected_triples = triples[
                        offset:offset + args.batch_size
                    ]
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
                        guided=guided,
                    )
                    decoded_chunks.append(rollout["decoded_steps"])
                    noise_streams.extend(rollout["noise_streams"])
                    guidance_windows.extend(rollout["guidance_windows"])
                    history_max_abs = max(
                        history_max_abs, float(rollout["history_max_abs"]),
                    )
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
                decoded_steps = concatenate_decoded_steps(
                    decoded_chunks, device,
                )
                kinematics = physical_summary(
                    dataset, triples, decoded_steps, device,
                )
                summary = _summary_for_records(
                    records, include_categories=True,
                )
                audit = guidance_audit_summary(guidance_windows)
                finite = bool(
                    all_finite(summary)
                    and all_finite(kinematics)
                    and all_finite(audit)
                    and torch.isfinite(torch.cat([
                        value["joints"].reshape(-1)
                        for value in decoded_steps
                    ])).all()
                )
                complete = bool(
                    len(records) == 64
                    and reports_complete(summary)
                    and all(reports_complete(record) for record in records)
                    and set(kinematics["aggregate"]) >= {
                        "object_goal_error_cm",
                        "pelvis_goal_error_cm",
                        "mpjpe_cm",
                        "object_translation_mae_cm",
                        "object_rotation_geodesic_deg",
                        "foot_sliding",
                        "physical_contact_f1",
                        "physical_contact_precision",
                        "physical_contact_recall",
                    }
                )
            except Exception as exc:  # preserve a contract-failure artifact
                variant_error = f"{type(exc).__name__}: {exc}"
                summary = {}
                kinematics = {}
                audit = guidance_audit_summary(guidance_windows)
                finite = False
                complete = False
            variants[variant] = {
                "error": variant_error,
                "history_max_abs": history_max_abs,
                "noise_streams": noise_streams,
                "guidance_audit": audit,
                "aggregate": summary,
                "kinematics": kinematics,
                "per_sequence": records,
                "finite": finite,
                "all_fields_thresholds_and_metrics_reported": complete,
            }
            variant_records[variant] = records
            variant_kinematics[variant] = kinematics
            torch.cuda.empty_cache()
        model_after = state_dict_sha256(model)
        comparison = {}
        comparison_error = None
        try:
            if all(
                variants[variant]["all_fields_thresholds_and_metrics_reported"]
                for variant in VARIANTS
            ):
                comparison = paired_guidance_comparison(
                    variant_records["unguided"],
                    variant_records["guided"],
                    variant_kinematics["unguided"],
                    variant_kinematics["guided"],
                )
        except Exception as exc:
            comparison_error = f"{type(exc).__name__}: {exc}"
        comparisons[name] = comparison
        models[name] = {
            "checkpoint": {
                "path": str(paths[name]),
                "sha256": EXPECTED_CHECKPOINT_SHA256[name],
                "metadata": metadata,
                "model_state_sha256_before": model_before,
                "model_state_sha256_after": model_after,
                "model_state_unchanged": model_before == model_after,
                "parameter_grad_buffers_clear": all(
                    parameter.grad is None for parameter in model.parameters()
                ),
            },
            "variants": variants,
            "paired_guided_minus_unguided": comparison,
            "paired_comparison_error": comparison_error,
        }
        del model
        torch.cuda.empty_cache()

    all_noise_streams = [
        models[model]["variants"][variant]["noise_streams"]
        for model in MODELS for variant in VARIANTS
    ]
    reference_noise = all_noise_streams[0]
    paired_noise_identity = bool(
        reference_noise
        and all(value == reference_noise for value in all_noise_streams[1:])
    )
    custom_source = inspect.getsource(sample_contact_counterfactual)
    production_source = inspect.getsource(GaussianDiffusion.sample)
    formula_replay_max_abs = author_formula_replay_max_abs(device)
    contract = {
        "checkpoint_hashes_exact": True,
        "asset_hashes_exact": asset_hashes == expected_asset_hashes,
        "author_blob_hashes_exact": actual_author_hashes == AUTHOR_BLOB_SHA256,
        "author_formula_replay_max_abs_le_1e-6": (
            formula_replay_max_abs <= 1e-6
        ),
        "data_contract_exact": all(
            models[name]["checkpoint"]["metadata"]["data_contract_sha256"]
            == EXPECTED_DATA_CONTRACT_SHA256
            for name in MODELS
        ),
        "selection_exact": (
            selection["sha256"] == SELECTION_SHA256
            and selection["sequences"] == 64
            and selection["windows"] == 192
            and selection["phase_offsets"] == list(PHASE_OFFSETS)
        ),
        "paired_sampler_noise_identity": paired_noise_identity,
        "history_restoration": all(
            float(models[name]["variants"][variant]["history_max_abs"])
            <= HISTORY_MAX_ABS
            for name in MODELS for variant in VARIANTS
        ),
        "all_finite": all(
            bool(models[name]["variants"][variant]["finite"])
            for name in MODELS for variant in VARIANTS
        ),
        "all_checkpoints_and_variants_reported": (
            set(models) == set(MODELS)
            and all(
                set(models[name]["variants"]) == set(VARIANTS)
                for name in MODELS
            )
        ),
        "all_fields_thresholds_and_metrics_reported": all(
            bool(models[name]["variants"][variant][
                "all_fields_thresholds_and_metrics_reported"
            ])
            for name in MODELS for variant in VARIANTS
        ),
        "guided_steps_and_step_zero_exact": all(
            (
                int(models[name]["variants"][variant]["guidance_audit"][
                    "applied_steps"
                ])
                == int(models[name]["variants"][variant]["guidance_audit"][
                    "expected_applied_steps"
                ])
                and not bool(models[name]["variants"][variant][
                    "guidance_audit"
                ]["step_zero_guidance_applied"])
            )
            for name in MODELS for variant in VARIANTS
        ),
        "author_hand_weight_exact": all(
            (
                models[name]["variants"]["guided"]["guidance_audit"][
                    "aggregate"
                ]["author_hand_weight"]["min"]
                == AUTHOR_HAND_WEIGHT
                and models[name]["variants"]["guided"]["guidance_audit"][
                    "aggregate"
                ]["author_hand_weight"]["max"]
                == AUTHOR_HAND_WEIGHT
            )
            for name in MODELS
        ),
        "paired_comparisons_complete": all(
            models[name]["paired_comparison_error"] is None
            and set(comparisons[name]) == {"contact", "kinematics"}
            for name in MODELS
        ),
        "model_state_unchanged": all(
            bool(models[name]["checkpoint"]["model_state_unchanged"])
            for name in MODELS
        ),
        "parameter_grad_buffers_clear": all(
            bool(models[name]["checkpoint"]["parameter_grad_buffers_clear"])
            for name in MODELS
        ),
        "posterior_helper_reused": "diffusion.posterior_sample(" in custom_source,
        "production_sampler_default_unchanged": (
            "guidance" not in production_source
            and "sample_contact_counterfactual" not in production_source
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
    decision = mechanism_gate(contract, comparisons)
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
        "models": models,
        "comparisons": comparisons,
        "contract": contract,
        "decision": decision,
        "sampler_contract": {
            "production_default_changed": False,
            "future_gt": False,
            "stored_per_frame_bps": False,
            "rollout_bps": "recomputed_from_current_generated_object_pose",
            "paired_noise_identity": paired_noise_identity,
            "posterior_helper_reused": contract["posterior_helper_reused"],
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
            "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    exclusive_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
