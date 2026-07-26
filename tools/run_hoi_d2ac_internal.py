#!/usr/bin/env python3
"""Run the fixed D2-AC0 internal causal interaction diagnostic."""

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
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from pytorch3d import transforms


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

import utils as author_utils  # noqa: E402
from datasets.utils import get_smpl_parents  # noqa: E402
from eval_metrics import compute_collision  # noqa: E402
from priors.contact_alignment import (  # noqa: E402
    all_finite,
    geometry_report,
    select_contact_holdout,
    semantic_report,
)
from priors.contact_guidance import decoded_fk_positions, hand_distances  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.interaction_adapter import ASSIGNMENT_SHA256, BPS_SHA256  # noqa: E402
from priors.interaction_diagnostic import (  # noqa: E402
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
    VARIANTS,
    attention_entropy,
    gt_contact_frame_distance,
    internal_mechanism_gate,
    paired_difference_fixed,
    paired_finite_difference,
    paired_ratio_fixed,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AC,
    load_trained_hoi_prior,
)
from priors.window_codec import project_to_so3  # noqa: E402
from tools.diagnose_hoi_d2o import sha256_tensor_state  # noqa: E402
from tools.diagnose_hoi_d2q import (  # noqa: E402
    _summary_for_records,
    concatenate_decoded_steps,
    prepare_targets,
    reports_complete,
)
from tools.diagnose_hoi_remediation import (  # noqa: E402
    physical_summary,
    seed_everything,
)
from tools.evaluate_hoi_remediation import (  # noqa: E402
    current_bps,
    global_goals,
    load_rest_vertices,
    stack_frames,
)


SUBPHASE = "1B-D2-AC0-internal"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ac-interaction-adapter-internal-s42-[0-9]{8}$"
)
TRAINING_RUN_ID = "p1-hoi-d2ac-interaction-adapter-s42-20260726"
EXPECTED_PYTHON = "/home/yujinlun/data/envs/infbagel/bin/python"
EXPECTED_DATA_CONTRACT_SHA256 = (
    "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
)
EXPECTED_SPLIT_SHA256 = (
    "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
)
DEFAULT_BATCH_SIZE = 8
WINDOWS_PER_SEQUENCE = 3
EXCLUDED_PENETRATION_OBJECTS = frozenset({
    "woodchair",
    "whitechair",
    "largebox",
    "largetable",
    "plasticbox",
    "trashcan",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    if chunk_index < 0 or window_index not in range(WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-AC sampler seed coordinates")
    return f"D2:d2ac-shared:chunk:{chunk_index}:window:{window_index}"


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def sequence_names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{name}\n" for name in names).encode("utf-8")
    ).hexdigest()


def checkpoint_contract(path: Path, expected_sha256: str) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_name = f"{TRAINING_RUN_ID}_windows061440000.pth"
    if actual != expected_sha256:
        raise ValueError(f"D2-AC final checkpoint hash mismatch: {actual}")
    if path.name != expected_name:
        raise ValueError("D2-AC internal requires the fixed final checkpoint basename")
    checkpoint = torch.load(path, map_location="cpu")
    initialization = checkpoint.get("weight_initialization", {})
    adapter = checkpoint.get("interaction_adapter_contract", {})
    resume = checkpoint.get("resume_contract", {})
    checks = {
        "checkpoint_type": checkpoint.get("checkpoint_type") == "hoi_prior_phase1b",
        "expert": checkpoint.get("expert") == "hoi",
        "run_id": checkpoint.get("run_id") == TRAINING_RUN_ID,
        "seed": checkpoint.get("seed") == 42,
        "processed_windows": checkpoint.get("processed_windows") == 61_440_000,
        "processed_frames": checkpoint.get("processed_frames") == 983_040_000,
        "optimizer_updates": checkpoint.get("optimizer_updates") == 30_000,
        "world_size": checkpoint.get("world_size") == 4,
        "effective_batch_size": checkpoint.get("effective_batch_size") == 2048,
        "architecture_variant": (
            checkpoint.get("architecture_variant") == HOI_ARCHITECTURE_D2AC
            and checkpoint.get("model_config", {}).get("architecture_variant")
            == HOI_ARCHITECTURE_D2AC
        ),
        "adapter_provenance": (
            adapter.get("bps_sha256") == BPS_SHA256
            and adapter.get("assignment_sha256") == ASSIGNMENT_SHA256
            and adapter.get("adapter_parameters") == 349_697
            and adapter.get("alpha_initial") == 0.0
        ),
        "data_contract": (
            checkpoint.get("data_contract_sha256")
            == EXPECTED_DATA_CONTRACT_SHA256
        ),
        "split": checkpoint.get("split_sha256") == EXPECTED_SPLIT_SHA256,
        "random_initialization": (
            checkpoint.get("initialization") == "random"
            and initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("restored_components") == []
        ),
        "no_ema": checkpoint.get("ema_models") == {},
        "online_model": isinstance(checkpoint.get("model"), dict),
        "d2x_routing": resume.get("fk_foot_temporal_routing") is True,
        "d2ab_disabled": resume.get("d2ab_predicted_support_no_slip") is False,
        "d2ac_enabled": resume.get("d2ac_interaction_adapter") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AC final checkpoint contract mismatch: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
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
        "mode": "interaction-adapter-internal-causal-diagnostic",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(Path(sys.executable).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "target_checkpoint": {
            "path": str(args.target_checkpoint.resolve()),
            "sha256": args.target_sha256,
            "run_id": TRAINING_RUN_ID,
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
            "same_text_global_goals_progress_and_window_order": True,
            "history_restoration": True,
            "rollout_bps": "recomputed_from_each_generated_object_reference",
            "cfg": False,
            "guidance": False,
            "dynamic_perception": False,
            "future_gt": False,
            "stored_per_frame_bps": False,
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
            "penetration": (
                "official SDF formulas at native internal 10Hz frames; "
                "official excluded object categories remain non-finite"
            ),
            "paired_unit": "sequence",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "attention": {
            "capture": True,
            "roles": list(ROLE_NAMES),
            "tokens": 16,
            "reported": ["entropy_nats", "entropy_normalized"],
            "selection_use": False,
        },
        "assets": {
            "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
            "bps_sha256": BPS_SHA256,
            "assignment_sha256": ASSIGNMENT_SHA256,
        },
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_writes": 0,
        "checkpoint_selection": False,
        "official_test_used": False,
        "output_dir": str(args.output_dir.resolve()),
        "metrics_path": str(args.metrics.resolve()),
    }


class AttentionCapture:
    """Accumulate per-row/role attention entropy without retaining maps."""

    def __init__(self, batch: int, device: torch.device) -> None:
        self.raw_sum = torch.zeros(
            batch, len(ROLE_NAMES), dtype=torch.float64, device=device
        )
        self.normalized_sum = torch.zeros_like(self.raw_sum)
        self.calls = 0

    def hook(self, module, inputs, output) -> None:
        del inputs, output
        weights = module.attention_snapshot()
        if weights is None:
            raise RuntimeError("D2-AC attention capture produced no weights")
        # [B,T,role,head,token] -> mean over T/head for each row/role.
        entropy = attention_entropy(weights)
        self.raw_sum += entropy["nats"].mean(dim=(1, 3)).double()
        self.normalized_sum += entropy["normalized"].mean(dim=(1, 3)).double()
        self.calls += 1

    def result(self) -> List[Dict[str, object]]:
        if self.calls != 500:
            raise ValueError(f"D2-AC expected 500 attention calls, got {self.calls}")
        raw = (self.raw_sum / self.calls).detach().cpu().numpy()
        normalized = (
            self.normalized_sum / self.calls
        ).detach().cpu().numpy()
        return [
            {
                "forward_calls": self.calls,
                "roles": {
                    role: {
                        "entropy_nats": float(raw[row, role_index]),
                        "entropy_normalized": float(normalized[row, role_index]),
                    }
                    for role_index, role in enumerate(ROLE_NAMES)
                },
            }
            for row in range(raw.shape[0])
        ]


def _sequence_name(dataset: PriorWindowDataset, position: int) -> str:
    global_index = int(dataset.indices[position])
    sequence = int(dataset.sequence_ids[global_index])
    return str(dataset.scene_names[sequence])


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
    attention_by_window: List[List[Dict[str, object]]] = []
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
        label = sampler_seed_label(chunk_index, window_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(label))
        initial_state = sha256_tensor_state(generator.get_state())
        capture = AttentionCapture(len(triples), device)
        adapter = model.network.interaction_adapter
        model.network.set_interaction_attention_capture(True)
        hook = adapter.register_forward_hook(capture.hook)
        try:
            sample = diffusion.sample(
                model, fixed, text, bps, goals, progress, generator=generator,
            )
        finally:
            hook.remove()
            model.network.set_interaction_attention_capture(False)
        final_state = sha256_tensor_state(generator.get_state())
        attention_by_window.append(capture.result())
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        history_max_abs = max(
            history_max_abs,
            float((sample[:, :2] - fixed).abs().max().detach().cpu()),
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
        value = {
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
        }
        value["fk_joints"] = torch.cat([
            fk_steps[step][row, 2:]
            for step in range(WINDOWS_PER_SEQUENCE)
        ])
        value["attention_entropy"] = {
            role: {
                statistic: float(np.mean([
                    attention_by_window[step][row]["roles"][role][statistic]
                    for step in range(WINDOWS_PER_SEQUENCE)
                ]))
                for statistic in ("entropy_nats", "entropy_normalized")
            }
            for role in ROLE_NAMES
        }
        value["attention_forward_calls"] = sum(
            int(attention_by_window[step][row]["forward_calls"])
            for step in range(WINDOWS_PER_SEQUENCE)
        )
        generated.append(value)
    return {
        "generated": generated,
        "decoded_steps": decoded_steps,
        "noise_streams": noise_streams,
        "history_max_abs": history_max_abs,
    }


def load_penetration_assets(repo: Path) -> Dict[str, object]:
    sdf_root = repo / "data/object/rest_object_sdf_256_npy_files"
    sdf = {}
    metadata = {}
    for path in sorted(sdf_root.glob("*.npy")):
        name = path.stem
        json_path = path.with_suffix(".json")
        sdf[name] = np.load(path)
        metadata[name] = json.loads(json_path.read_text(encoding="utf-8"))
    hand_ids_path = repo / "smpl_models/MANO_SMPLX_vertex_ids.pkl"
    with hand_ids_path.open("rb") as handle:
        hand_ids_value = pickle.load(handle)
    hand_indices = np.concatenate((
        hand_ids_value["left_hand"],
        hand_ids_value["right_hand"],
    ))
    return {
        "sdf": sdf,
        "metadata": metadata,
        "hand_indices": hand_indices,
        "hand_ids_sha256": sha256_file(hand_ids_path),
    }


def global_to_local_rotations(
    global_rotation: torch.Tensor,
    parents_22: torch.Tensor,
) -> torch.Tensor:
    if global_rotation.ndim != 4 or global_rotation.shape[1:] != (22, 3, 3):
        raise ValueError("D2-AC global rotations must be [frames,22,3,3]")
    local = global_rotation.clone()
    for joint in range(1, 22):
        parent = int(parents_22[joint])
        local[:, joint] = (
            global_rotation[:, parent].transpose(-1, -2)
            @ global_rotation[:, joint]
        )
    return local


def sequence_penetration(
    generated: Mapping[str, torch.Tensor],
    *,
    sequence_index: int,
    object_name: str,
    device: torch.device,
    parents_22: torch.Tensor,
    betas: np.ndarray,
    genders: Sequence[str],
    translations: np.ndarray,
    penetration_assets: Mapping[str, object],
    smpl_cache: MutableMapping[str, torch.nn.Module],
) -> Dict[str, object]:
    if object_name in EXCLUDED_PENETRATION_OBJECTS:
        return {
            "finite": False,
            "excluded_by_official_contract": True,
            "hand_pen_loss_omomo": None,
            "hand_pen_ratio": None,
            "human_pen_loss_infbagel": None,
            "human_pen_ratio": None,
        }
    if object_name not in penetration_assets["sdf"]:
        raise ValueError(f"D2-AC missing penetration SDF for {object_name}")
    gender = str(genders[sequence_index])
    if gender not in smpl_cache:
        smpl_cache[gender] = author_utils.create_smplx_model(
            gender, device, batch_size=1,
        )
    global_rotation = generated["human_rotation"].to(device)
    local_rotation = global_to_local_rotations(global_rotation, parents_22)
    pose = author_utils.yup_to_zup(
        transforms.matrix_to_axis_angle(local_rotation)
    )
    aligned_translation = torch.as_tensor(
        translations[sequence_index], device=device, dtype=torch.float32,
    )
    root_translation = author_utils.yup_to_zup(
        generated["joints"].to(device)[:, 0] + aligned_translation
    )
    beta = torch.as_tensor(
        betas[sequence_index], device=device, dtype=torch.float32,
    )
    vertices, _ = author_utils.run_smplx_model(
        pose,
        root_translation,
        beta,
        gender,
        joints_ind=None,
        smpl_model=smpl_cache[gender],
    )
    object_rotation = author_utils.yup_to_zup_rotation_matrix(
        generated["object_rotation"].to(device)
    )
    object_translation = author_utils.yup_to_zup(
        generated["object_translation"].to(device)
    )
    sdf = penetration_assets["sdf"][object_name]
    sdf_metadata = penetration_assets["metadata"][object_name]
    hand_vertices = vertices[:, penetration_assets["hand_indices"], :]
    hand_loss, hand_ratio = compute_collision(
        vertices.new_tensor(hand_vertices),
        sdf,
        sdf_metadata,
        object_rotation,
        object_translation,
    )
    human_loss, human_ratio = compute_collision(
        vertices,
        sdf,
        sdf_metadata,
        object_rotation,
        object_translation,
    )
    result = {
        "finite": True,
        "excluded_by_official_contract": False,
        "hand_pen_loss_omomo": float(hand_loss),
        "hand_pen_ratio": float(hand_ratio),
        "human_pen_loss_infbagel": float(human_loss * 10475 / 100),
        "human_pen_ratio": float(human_ratio),
    }
    if not all(math.isfinite(float(value)) for key, value in result.items()
               if key not in {"finite", "excluded_by_official_contract"}):
        raise FloatingPointError("D2-AC internal penetration is non-finite")
    return result


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
    target_direct = hand_distances(
        target["joints"].to(device), target_vertices, DIRECT_HAND_INDICES,
    ).cpu().numpy().astype(np.float64)
    predicted_direct = hand_distances(
        generated["joints"].to(device), predicted_vertices, DIRECT_HAND_INDICES,
    ).cpu().numpy().astype(np.float64)
    target_fk = hand_distances(
        target["fk_joints"].to(device), target_vertices, FK_PALM_INDICES,
    ).cpu().numpy().astype(np.float64)
    predicted_fk = hand_distances(
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
        "semantic_vs_gt": semantic_report(predicted_contact, target_contact),
        "direct_physical_geometry_vs_gt": geometry_report(
            predicted_direct, target_direct,
        ),
        "fk_physical_geometry_vs_gt": geometry_report(
            predicted_fk, target_fk,
        ),
        "gt_contact_frame_direct_distance": gt_contact_frame_distance(
            predicted_direct, target_direct,
        ),
        "penetration": dict(penetration),
        "attention_entropy": generated["attention_entropy"],
        "attention_forward_calls": generated["attention_forward_calls"],
        "per_frame": {
            "target_contact": target_contact.tolist(),
            "predicted_contact": predicted_contact.tolist(),
            "target_direct_hand_object_distance_m": target_direct.tolist(),
            "predicted_direct_hand_object_distance_m": predicted_direct.tolist(),
            "target_fk_hand_object_distance_m": target_fk.tolist(),
            "predicted_fk_hand_object_distance_m": predicted_fk.tolist(),
        },
    }


def kinematics_by_sequence(
    value: Mapping[str, object],
    sequence_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    windows = defaultdict(list)
    for row in value["per_sequence_window"]:
        windows[str(row["sequence"])].append(row)
    foot = {
        str(row["sequence"]): float(row["foot_sliding"])
        for row in value["per_sequence"]
    }
    result = {}
    for name in sequence_names:
        rows = sorted(windows[name], key=lambda row: int(row["window"]))
        if len(rows) != 3 or name not in foot:
            raise ValueError(f"D2-AC incomplete kinematics for {name}")
        result[name] = {
            "mpjpe_cm": float(np.mean([row["mpjpe_cm"] for row in rows])),
            "object_goal_error_cm": float(rows[-1]["object_goal_error_cm"]),
            "pelvis_goal_error_cm": float(np.mean([
                row["pelvis_goal_error_cm"] for row in rows
            ])),
            "object_translation_mae_cm": float(np.mean([
                row["object_translation_mae_cm"] for row in rows
            ])),
            "object_rotation_geodesic_deg": float(np.mean([
                row["object_rotation_geodesic_deg"] for row in rows
            ])),
            "foot_sliding": foot[name],
        }
    return result


def aggregate_penetration(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    result = {
        "official_finite_sequences": 0,
        "excluded_sequences": 0,
        "metrics": {},
    }
    for metric in (
        "hand_pen_loss_omomo",
        "hand_pen_ratio",
        "human_pen_loss_infbagel",
        "human_pen_ratio",
    ):
        values = [
            float(record["penetration"][metric])
            for record in records
            if record["penetration"][metric] is not None
        ]
        result["metrics"][metric] = float(np.mean(values)) if values else None
    result["official_finite_sequences"] = sum(
        bool(record["penetration"]["finite"]) for record in records
    )
    result["excluded_sequences"] = len(records) - int(
        result["official_finite_sequences"]
    )
    return result


def aggregate_attention(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        role: {
            statistic: float(np.mean([
                record["attention_entropy"][role][statistic]
                for record in records
            ]))
            for statistic in ("entropy_nats", "entropy_normalized")
        }
        for role in ROLE_NAMES
    }


def _path(record: Mapping[str, object], keys: Sequence[str]) -> object:
    value: object = record
    for key in keys:
        value = value[key]  # type: ignore[index]
    return value


def _values(
    records: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> List[object]:
    return [_path(record, keys) for record in records]


def paired_comparisons(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    names = [str(record["sequence"]) for record in records["full"]]
    for variant in VARIANTS[1:]:
        if [str(record["sequence"]) for record in records[variant]] != names:
            raise ValueError("D2-AC paired sequence ordering differs")
    result = {}
    for other in VARIANTS[1:]:
        comparison: Dict[str, object] = {
            "full_minus_other_direct_union_5cm_f1": paired_difference_fixed(
                _values(
                    records["full"],
                    ("direct_physical_geometry_vs_gt", "thresholds_cm", "5", "union", "f1"),
                ),
                _values(
                    records[other],
                    ("direct_physical_geometry_vs_gt", "thresholds_cm", "5", "union", "f1"),
                ),
            ),
            "other_minus_full_gt_contact_distance_cm": paired_finite_difference(
                _values(
                    records[other],
                    ("gt_contact_frame_direct_distance", "union", "mean_cm"),
                ),
                _values(
                    records["full"],
                    ("gt_contact_frame_direct_distance", "union", "mean_cm"),
                ),
                names,
            ),
            "semantic_contact": {},
            "physical_contact": {},
            "kinematics": {},
            "penetration": {},
        }
        for unit in ("left_hand", "right_hand", "union"):
            comparison["semantic_contact"][unit] = {
                metric: paired_difference_fixed(
                    _values(
                        records["full"],
                        ("semantic_vs_gt", "thresholds", "0.5", unit, metric),
                    ),
                    _values(
                        records[other],
                        ("semantic_vs_gt", "thresholds", "0.5", unit, metric),
                    ),
                )
                for metric in ("precision", "recall", "f1", "prediction_percent")
            }
        for geometry in (
            "direct_physical_geometry_vs_gt",
            "fk_physical_geometry_vs_gt",
        ):
            comparison["physical_contact"][geometry] = {}
            for threshold in PHYSICAL_THRESHOLDS_CM:
                key = f"{threshold:g}"
                comparison["physical_contact"][geometry][key] = {}
                for unit in ("left_hand", "right_hand", "union"):
                    comparison["physical_contact"][geometry][key][unit] = {
                        metric: paired_difference_fixed(
                            _values(
                                records["full"],
                                (geometry, "thresholds_cm", key, unit, metric),
                            ),
                            _values(
                                records[other],
                                (geometry, "thresholds_cm", key, unit, metric),
                            ),
                        )
                        for metric in (
                            "precision",
                            "recall",
                            "f1",
                            "prediction_percent",
                        )
                    }
                    comparison["physical_contact"][geometry][key][unit][
                        "prediction_run_mean_frames"
                    ] = paired_difference_fixed(
                        _values(
                            records["full"],
                            (
                                geometry, "thresholds_cm", key, unit,
                                "prediction_run_lengths", "mean_frames",
                            ),
                        ),
                        _values(
                            records[other],
                            (
                                geometry, "thresholds_cm", key, unit,
                                "prediction_run_lengths", "mean_frames",
                            ),
                        ),
                    )
        for metric in (
            "mpjpe_cm",
            "object_goal_error_cm",
            "pelvis_goal_error_cm",
            "object_translation_mae_cm",
            "object_rotation_geodesic_deg",
            "foot_sliding",
        ):
            comparison["kinematics"][metric] = paired_ratio_fixed(
                _values(records["full"], ("kinematics", metric)),
                _values(records[other], ("kinematics", metric)),
            )
        for metric in (
            "hand_pen_loss_omomo",
            "human_pen_loss_infbagel",
        ):
            full = _values(records["full"], ("penetration", metric))
            alternate = _values(records[other], ("penetration", metric))
            keep = [
                index for index, (left, right) in enumerate(zip(full, alternate))
                if left is not None and right is not None
            ]
            comparison["penetration"][metric] = paired_ratio_fixed(
                [float(full[index]) for index in keep],
                [float(alternate[index]) for index in keep],
            )
            comparison["penetration"][metric]["finite_sequence_count"] = len(keep)
            comparison["penetration"][metric]["finite_sequence_names"] = [
                names[index] for index in keep
            ]
        result[f"full_vs_{other}"] = comparison
    return result


def variant_complete(
    records: Sequence[Mapping[str, object]],
    kinematics: Mapping[str, object],
) -> bool:
    return bool(
        len(records) == 64
        and reports_complete({
            "semantic_vs_gt": records[0]["semantic_vs_gt"],
            "fk_physical_geometry_vs_gt": records[0]["fk_physical_geometry_vs_gt"],
            "direct_physical_geometry_vs_gt": records[0][
                "direct_physical_geometry_vs_gt"
            ],
        })
        and all(
            reports_complete({
                "semantic_vs_gt": record["semantic_vs_gt"],
                "fk_physical_geometry_vs_gt": record["fk_physical_geometry_vs_gt"],
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
        and all(
            int(record["attention_forward_calls"]) == 1500
            for record in records
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("invalid D2-AC internal lifecycle run id")
    if not re.fullmatch(r"[0-9a-f]{64}", args.target_sha256):
        raise ValueError("D2-AC target SHA-256 must be lowercase hexadecimal")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-AC internal batch size must evenly divide 64")
    config = resolved_config(args)
    if args.resolve_only:
        exclusive_json(args.resolved_config.resolve(), config)
        return
    if Path(sys.executable).resolve() != Path(
        os.environ.get("INFBAGEL_PYTHON", ""),
    ).resolve():
        raise ValueError("D2-AC internal requires the absolute INFBAGEL_PYTHON")
    if Path(sys.executable).resolve() != Path(EXPECTED_PYTHON).resolve():
        raise ValueError(f"D2-AC internal requires {EXPECTED_PYTHON}")
    if (
        os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi"
        or socket.gethostname() != "node01"
    ):
        raise RuntimeError("D2-AC internal is restricted to the HOI worker")
    if git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-AC internal refuses a dirty worker checkout")
    if json.loads(args.resolved_config.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-AC internal runtime differs from archived config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-AC internal requires worker CUDA")
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
            args.target_checkpoint.resolve(), args.target_sha256,
        )
        asset_hashes = {
            "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
            "bps": sha256_file((REPO / "code/bps.pt").resolve()),
            "split": sha256_file(
                REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
        }
        if asset_hashes != {
            "normalization": EXPECTED_NORMALIZATION_SHA256,
            "bps": BPS_SHA256,
            "split": EXPECTED_SPLIT_SHA256,
        }:
            raise ValueError(f"D2-AC internal asset hash mismatch: {asset_hashes}")

        seed_everything(42)
        dataset = PriorWindowDataset(
            str(REPO),
            "hoi",
            partition="internal_validation",
            split_manifest=(
                "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
        )
        selection = select_contact_holdout(dataset)
        if (
            selection["sha256"] != SELECTION_SHA256
            or selection["sequences"] != 64
            or selection["windows"] != 192
            or selection["phase_offsets"] != [14, 56, 98]
        ):
            raise ValueError("D2-AC internal selection contract mismatch")
        triples = selection["triples"]
        parents_24 = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy()
        ).long().to(device)
        parents_22 = torch.from_numpy(
            get_smpl_parents(use_joints24=False).copy()
        ).long().to(device)
        targets = prepare_targets(dataset, triples, parents_24.cpu())
        for target, triple in zip(targets, triples):
            target["sequence_index"] = int(
                dataset[triple[0]]["sequence_index"].item()
            )
        rest_vertices = load_rest_vertices(dataset, triples, device)
        penetration_assets = load_penetration_assets(REPO)
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
            expected_architecture_variant=HOI_ARCHITECTURE_D2AC,
        )
        if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError("D2-AC internal checkpoint data-contract mismatch")
        model.eval()
        model_before = state_dict_sha256(model)

        variants: Dict[str, object] = {}
        records_by_variant: Dict[str, List[Dict[str, object]]] = {}
        noise_by_variant: Dict[str, object] = {}
        attention_appendix: Dict[str, object] = {}
        for variant in VARIANTS:
            model.network.set_interaction_diagnostic_variant(variant)
            records: List[Dict[str, object]] = []
            decoded_chunks = []
            noise_streams = []
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
                history_max_abs = max(
                    history_max_abs, float(rollout["history_max_abs"])
                )
                for target, generated in zip(
                    targets[offset:offset + args.batch_size],
                    rollout["generated"],
                ):
                    penetration = sequence_penetration(
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
            decoded_steps = concatenate_decoded_steps(decoded_chunks, device)
            kinematics = physical_summary(
                dataset, triples, decoded_steps, device,
            )
            sequence_names = [str(record["sequence"]) for record in records]
            mapped_kinematics = kinematics_by_sequence(
                kinematics, sequence_names,
            )
            for record in records:
                record["kinematics"] = mapped_kinematics[str(record["sequence"])]
            semantic_geometry = _summary_for_records(
                records, include_categories=True,
            )
            penetration_summary = aggregate_penetration(records)
            attention_summary = aggregate_attention(records)
            finite = bool(
                all_finite(semantic_geometry)
                and all_finite(kinematics)
                and all_finite(attention_summary)
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
                    "attention_entropy": attention_summary,
                },
                "kinematics_full": kinematics,
                "per_sequence": records,
                "noise_streams": noise_streams,
                "finite": finite,
                "all_fields_reported": complete,
            }
            variant_path = output_dir / f"{variant}.json"
            exclusive_json(variant_path, variant_value)
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
            attention_appendix[variant] = {
                "aggregate": attention_summary,
                "per_sequence": [
                    {
                        "sequence": record["sequence"],
                        "roles": record["attention_entropy"],
                        "forward_calls": record["attention_forward_calls"],
                    }
                    for record in records
                ],
            }
            model.network.interaction_adapter.clear_diagnostic_state()
            torch.cuda.empty_cache()

        model_after = state_dict_sha256(model)
        comparisons = paired_comparisons(records_by_variant)
        finite_masks = [
            comparisons[f"full_vs_{variant}"][
                "other_minus_full_gt_contact_distance_cm"
            ]["finite_sequence_names"]
            for variant in VARIANTS[1:]
        ]
        gt_contact_mask_exact = bool(
            finite_masks[0] == finite_masks[1]
            and len(finite_masks[0]) == GT_CONTACT_FINITE_SEQUENCE_COUNT
            and sequence_names_sha256(finite_masks[0])
            == GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256
        )
        paired_noise_identity = all(
            noise_by_variant[variant] == noise_by_variant["full"]
            for variant in VARIANTS[1:]
        )
        paired_noise_path = output_dir / "paired_noise.json"
        exclusive_json(paired_noise_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "shared": paired_noise_identity,
            "variants": noise_by_variant,
        })
        attention_path = output_dir / "attention_appendix.json"
        exclusive_json(attention_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "selection_use": False,
            "roles": list(ROLE_NAMES),
            "variants": attention_appendix,
        })

        sampler_source = inspect.getsource(rollout_chunk)
        contract = {
            "checkpoint_contract": all(checkpoint["checks"].values()),
            "checkpoint_architecture_variant": (
                metadata["architecture_variant"] == HOI_ARCHITECTURE_D2AC
            ),
            "asset_hashes_exact": True,
            "selection_exact": True,
            "gt_contact_finite_mask_exact": gt_contact_mask_exact,
            "paired_noise_identity": paired_noise_identity,
            "history_restoration": all(
                float(variants[variant]["history_max_abs"])
                <= HISTORY_MAX_ABS
                for variant in VARIANTS
            ),
            "all_variants_finite": all(
                bool(variants[variant]["finite"]) for variant in VARIANTS
            ),
            "all_fields_reported": all(
                bool(variants[variant]["all_fields_reported"])
                for variant in VARIANTS
            ),
            "model_state_unchanged": model_before == model_after,
            "parameter_grad_buffers_clear": all(
                parameter.grad is None for parameter in model.parameters()
            ),
            "attention_capture_descriptive_only": True,
            "sampler_future_gt_absent": "future_gt" not in sampler_source,
            "sampler_stored_per_frame_bps_absent": (
                "stored_per_frame_bps" not in sampler_source
                and 'item["object_bps"]' not in sampler_source
            ),
            "rollout_bps_recomputed": "current_bps(" in sampler_source,
            "optimizer_absent": True,
            "checkpoint_write_absent": True,
            "official_test_absent": True,
        }
        decision = internal_mechanism_gate(contract, comparisons)
        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "completed",
            "seed": 42,
            "git_commit": git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "selection": {
                key: value
                for key, value in selection.items()
                if key != "triples"
            },
            "target_checkpoint": checkpoint,
            "checkpoint_metadata": metadata,
            "learned_adapter": {
                "alpha": float(
                    model.network.interaction_adapter.alpha.detach().cpu()
                ),
                "gate": float(torch.tanh(
                    model.network.interaction_adapter.alpha.detach()
                ).cpu()),
                "contract": model.network.interaction_adapter.contract_metadata(),
            },
            "assets": {
                **asset_hashes,
                "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
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
            "attention_appendix": {
                "path": str(attention_path),
                "sha256": sha256_file(attention_path),
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
        exclusive_json(args.metrics.resolve(), result)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "failed",
            "seed": 42,
            "git_commit": git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "classification": "interaction-adapter-contract-failure-stop",
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
            exclusive_json(failure_path, failure)
        if not args.metrics.resolve().exists():
            exclusive_json(args.metrics.resolve(), failure)
        raise


if __name__ == "__main__":
    main()
