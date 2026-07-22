#!/usr/bin/env python3
"""Run the preregistered D2-W fixed-checkpoint FK foot-sliding frontier audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from pytorch3d import transforms


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from eval_metrics import (  # noqa: E402
    compute_foot_sliding_for_smpl,
    determine_floor_height_and_contacts,
)
from priors.contact_guidance import decoded_fk_positions  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import (  # noqa: E402
    GaussianDiffusion,
    normalize_progress,
    prepare_clean_x0,
)
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.optimizer_reset import (  # noqa: E402
    NATIVE_SELECTION_SHA256,
    select_native_holdout,
    stable_seed,
)
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import WindowFrame, project_to_so3, rotation_geodesic  # noqa: E402
from tools.diagnose_hoi_remediation import object_vertices, raw_window_target  # noqa: E402
from tools.evaluate_hoi_remediation import (  # noqa: E402
    current_bps,
    global_goals,
    load_rest_vertices,
    stack_frames,
)
from tools.run_hoi_d2n import paired_difference, paired_ratio  # noqa: E402


RUN_ID = "p1-hoi-d2w-checkpoint-frontier-s42-20260722"
SUBPHASE = "1B-D2-W0"
EXPECTED_PYTHON = "/home/yujinlun/data/envs/infbagel/bin/python"
EXPECTED_DATA_CONTRACT_SHA256 = (
    "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
)
EXPECTED_CHECKPOINTS = {
    "control": {
        "processed_windows": 6_144_000,
        "file_sha256": "be8233c0a4c013d973c4140ba5c1f472332f1fdd6be8efa21585deeb250506d3",
        "model_state_sha256": "cfcb5836129d177bf57c60ffd8669ee4516fad77f52b58afd037d063e9aaa0c7",
    },
    "midpoint": {
        "processed_windows": 24_576_000,
        "file_sha256": "efab7f55d6a719ac85659de0aa66c2f94235e1875ae5e6951e9c4334017ee9a3",
        "model_state_sha256": "1ee340962d158e12a31d3ad081da37886cd8bbc3eddd80b523de4eb236ba2735",
    },
    "final": {
        "processed_windows": 61_440_000,
        "file_sha256": "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4",
        "model_state_sha256": "f7d134ac98ede806abae322c77816ef21ace427e3905a4cb5e1d4a2a2b4b89fc",
    },
}
LOWER_METRICS = (
    "fk_mpjpe_cm",
    "pelvis_goal_error_cm",
    "object_goal_error_cm",
    "object_translation_mae_cm",
)
CONTROL_IMPROVEMENT_METRICS = (
    "fk_mpjpe_cm",
    "object_goal_error_cm",
    "object_translation_mae_cm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _noise_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def sample_with_noise_audit(
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    fixed_history: torch.Tensor,
    text_embedding: torch.Tensor,
    object_bps: torch.Tensor,
    goals: torch.Tensor,
    progress: torch.Tensor,
    *,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, str]:
    """Replay production sampling while hashing every explicit base noise tensor."""
    batch = fixed_history.shape[0]
    shape = (batch, REPRESENTATION.window_frames, REPRESENTATION.dimension)
    digest = hashlib.sha256()
    current = torch.randn(shape, device=fixed_history.device, generator=generator)
    digest.update(current.detach().contiguous().cpu().numpy().tobytes())
    current[:, :REPRESENTATION.history_frames] = fixed_history
    for step in reversed(range(diffusion.timesteps)):
        timesteps = torch.full((batch,), step, dtype=torch.long, device=current.device)
        clean = model(current, timesteps, text_embedding, object_bps, goals, progress)
        clean = prepare_clean_x0(clean, fixed_history, object_so3_x0=False)
        if step:
            noise = torch.randn(current.shape, device=current.device, generator=generator)
            digest.update(noise.detach().contiguous().cpu().numpy().tobytes())
        else:
            noise = torch.zeros_like(current)
        current = diffusion.posterior_sample(
            current, clean, timesteps, noise, fixed_history,
        )
    return current, digest.hexdigest()


def torch_foot_sliding(joints: torch.Tensor, floor_height: torch.Tensor) -> torch.Tensor:
    """Vectorized replay of the official ankle/toe sliding formula."""
    if joints.ndim != 4 or joints.shape[2] < 12 or joints.shape[3] != 3:
        raise ValueError(f"expected [B,T,J,3] joints, got {tuple(joints.shape)}")
    if floor_height.shape != (joints.shape[0],):
        raise ValueError("floor height must contain one value per sequence")
    shifted = joints.clone()
    shifted[..., 1] -= floor_height[:, None, None]
    terms = []
    for joint, height in ((7, 0.08), (10, 0.04), (8, 0.08), (11, 0.04)):
        displacement = torch.linalg.vector_norm(
            shifted[:, 1:, joint][:, :, (0, 2)]
            - shifted[:, :-1, joint][:, :, (0, 2)],
            dim=-1,
        )
        y = shifted[:, :-1, joint, 1]
        active = y < height
        weighted = (displacement * (2.0 - torch.pow(2.0, y / height))).abs()
        terms.append((weighted * active).sum(dim=1) / shifted.shape[1] * 100.0)
    return torch.stack(terms, dim=1).mean(dim=1)


def official_foot_sliding(joints: torch.Tensor) -> Dict[str, object]:
    """Use official floor inference and prove the torch sliding replay is equivalent."""
    arrays = joints.detach().cpu().numpy().astype(np.float64, copy=True)
    floors: List[float] = []
    official: List[float] = []
    for value in arrays:
        floor = float(determine_floor_height_and_contacts(value.copy()))
        floors.append(floor)
        official.append(float(compute_foot_sliding_for_smpl(value.copy(), floor)))
    torch_values = torch_foot_sliding(
        joints.double().cpu(), torch.tensor(floors, dtype=torch.float64),
    ).numpy()
    maximum = float(np.max(np.abs(torch_values - np.asarray(official))))
    return {
        "values": torch_values.astype(np.float64),
        "floor_height": np.asarray(floors, dtype=np.float64),
        "official_values": np.asarray(official, dtype=np.float64),
        "maximum_torch_official_abs": maximum,
        "parity_passed": bool(maximum <= 1e-9),
    }


def _precision_recall_f1(counts: Sequence[int]) -> Tuple[float, float, float]:
    tp, fp, _, fn = [int(value) for value in counts]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _checkpoint_items(
    dataset: PriorWindowDataset, triples: Sequence[Sequence[int]], step: int,
) -> List[Mapping[str, torch.Tensor]]:
    return [dataset[triple[step]] for triple in triples]


@torch.no_grad()
def rollout_checkpoint(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
    rest_vertices: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    positions_by_step = [[triple[step] for triple in triples] for step in range(3)]
    items_by_step = [_checkpoint_items(dataset, triples, step) for step in range(3)]
    names = [
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
        for position in positions_by_step[0]
    ]
    first_items = items_by_step[0]
    frame = stack_frames(first_items, device)
    fixed = torch.stack([item["x"][:REPRESENTATION.history_frames] for item in first_items]).to(device)
    rest_offsets = torch.stack([item["rest_human_offsets"] for item in first_items]).to(device)
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True), device=device, dtype=torch.long)
    decoded_steps = []
    fk_steps = []
    target_steps = []
    pelvis_goals = []
    object_goals = []
    noise_hashes = []
    for step in range(3):
        items = items_by_step[step]
        gt_frame = stack_frames(items, device)
        pelvis_global, object_global = global_goals(dataset, items, gt_frame, device)
        goals = torch.zeros(len(triples), 9, device=device)
        goals[:, :3] = dataset.codec.pelvis_goal(pelvis_global, frame)
        goals[:, 6:9] = dataset.codec.object_goal(object_global, frame)
        text = torch.stack([item["text_embedding"] for item in items]).to(device)
        bps = current_bps(dataset, frame.object_reference, names, rest_vertices)
        progress = normalize_progress(torch.stack([item["progress"] for item in items]).to(device))
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(f"D2W:paired:{step}"))
        sample, noise_hash = sample_with_noise_audit(
            diffusion, model, fixed, text, bps, goals, progress, generator=generator,
        )
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        decoded = dataset.codec.decode(sample, frame)
        decoded_steps.append(decoded)
        fk_steps.append(decoded_fk_positions(decoded, rest_offsets, parents))
        target_steps.append([
            raw_window_target(dataset, position, device)
            for position in positions_by_step[step]
        ])
        pelvis_goals.append(pelvis_global)
        object_goals.append(object_global)
        noise_hashes.append(noise_hash)
        if step < 2:
            fixed, frame = dataset.codec.encode(
                decoded["joints"][:, -2:], decoded["human_rotation"][:, -2:],
                global_object_translation=decoded["object_translation"][:, -2:],
                global_object_rotation=decoded["object_rotation"][:, -2:],
                contact=decoded["contact"][:, -2:],
            )

    active = slice(REPRESENTATION.history_frames, None)
    predicted_direct = torch.cat([value["joints"][:, active] for value in decoded_steps], dim=1)
    predicted_fk = torch.cat([value[:, active] for value in fk_steps], dim=1)
    predicted_object = torch.cat(
        [value["object_translation"][:, active] for value in decoded_steps], dim=1,
    )
    predicted_object_rotation = torch.cat(
        [value["object_rotation"][:, active] for value in decoded_steps], dim=1,
    )
    target_direct = torch.stack([
        torch.cat([target_steps[step][row]["joints"][active] for step in range(3)], dim=0)
        for row in range(len(triples))
    ])
    target_object = torch.stack([
        torch.cat([
            target_steps[step][row]["object_translation"][active] for step in range(3)
        ], dim=0)
        for row in range(len(triples))
    ])
    target_object_rotation = torch.stack([
        torch.cat([
            target_steps[step][row]["object_rotation"][active] for step in range(3)
        ], dim=0)
        for row in range(len(triples))
    ])
    direct_sliding = official_foot_sliding(predicted_direct[:, :, :24])
    fk_sliding = official_foot_sliding(predicted_fk)
    per_sequence = {}
    aggregate_counts = np.zeros(4, dtype=np.int64)
    for row, name in enumerate(names):
        relative_prediction = predicted_fk[row] - predicted_fk[row, :, :1]
        relative_target = target_direct[row, :, :24] - target_direct[row, :, :1]
        fk_mpjpe = torch.linalg.vector_norm(
            relative_prediction - relative_target, dim=-1,
        ).mean() * 100.0
        pelvis_errors = []
        for step in range(3):
            pelvis_errors.append(torch.linalg.vector_norm(
                fk_steps[step][row, -1, 0, (0, 2)] - pelvis_goals[step][row, (0, 2)],
            ) * 100.0)
        object_goal_error = torch.linalg.vector_norm(
            decoded_steps[-1]["object_translation"][row, -1] - object_goals[-1][row],
        ) * 100.0
        object_translation = torch.linalg.vector_norm(
            predicted_object[row] - target_object[row], dim=-1,
        ).mean() * 100.0
        object_rotation = rotation_geodesic(
            predicted_object_rotation[row], target_object_rotation[row],
        ).mean() * (180.0 / math.pi)
        foot_disagreement = torch.linalg.vector_norm(
            predicted_direct[row, :, (7, 8, 10, 11)]
            - predicted_fk[row, :, (7, 8, 10, 11)],
            dim=-1,
        ).mean() * 100.0
        object_name = name.split("_")[1]
        rest = rest_vertices[object_name]
        indices = np.linspace(0, len(rest) - 1, min(2048, len(rest))).round().astype(np.int64)
        surface = rest[torch.from_numpy(indices).to(device=device)]
        predicted_vertices = object_vertices(
            surface, predicted_object_rotation[row], predicted_object[row],
        )
        target_vertices = object_vertices(
            surface, target_object_rotation[row], target_object[row],
        )
        pred_distance = torch.cdist(
            predicted_fk[row, :, (22, 23)], predicted_vertices,
        ).amin(dim=(1, 2))
        target_distance = torch.cdist(
            target_direct[row, :, (22, 23)], target_vertices,
        ).amin(dim=(1, 2))
        pred_contact = pred_distance < 0.05
        target_contact = target_distance < 0.05
        counts = np.asarray((
            int((pred_contact & target_contact).sum()),
            int((pred_contact & ~target_contact).sum()),
            int((~pred_contact & ~target_contact).sum()),
            int((~pred_contact & target_contact).sum()),
        ), dtype=np.int64)
        aggregate_counts += counts
        precision, recall, f1 = _precision_recall_f1(counts)
        per_sequence[name] = {
            "object_name": object_name,
            "direct_foot_sliding": float(direct_sliding["values"][row]),
            "fk_foot_sliding": float(fk_sliding["values"][row]),
            "fk_mpjpe_cm": float(fk_mpjpe),
            "pelvis_goal_error_cm": float(torch.stack(pelvis_errors).mean()),
            "object_goal_error_cm": float(object_goal_error),
            "object_translation_mae_cm": float(object_translation),
            "object_rotation_geodesic_deg": float(object_rotation),
            "physical_contact_precision": precision,
            "physical_contact_recall": recall,
            "physical_contact_f1": f1,
            "direct_fk_foot_disagreement_cm": float(foot_disagreement),
            "contact_counts": {
                "tp": int(counts[0]), "fp": int(counts[1]),
                "tn": int(counts[2]), "fn": int(counts[3]),
            },
        }
    aggregate_precision, aggregate_recall, aggregate_f1 = _precision_recall_f1(
        aggregate_counts,
    )
    scalar_names = tuple(
        name for name in next(iter(per_sequence.values())) if name not in {"object_name", "contact_counts"}
    )
    aggregate = {
        name: float(np.mean([record[name] for record in per_sequence.values()]))
        for name in scalar_names
    }
    aggregate.update({
        "physical_contact_precision": aggregate_precision,
        "physical_contact_recall": aggregate_recall,
        "physical_contact_f1": aggregate_f1,
        "finite": bool(all(
            math.isfinite(float(record[name]))
            for record in per_sequence.values()
            for name in scalar_names
        )),
    })
    return {
        "aggregate": aggregate,
        "per_sequence": per_sequence,
        "contact_counts": {
            "tp": int(aggregate_counts[0]), "fp": int(aggregate_counts[1]),
            "tn": int(aggregate_counts[2]), "fn": int(aggregate_counts[3]),
        },
        "noise_sha256_by_step": noise_hashes,
        "foot_sliding_parity": {
            "direct_max_abs": direct_sliding["maximum_torch_official_abs"],
            "fk_max_abs": fk_sliding["maximum_torch_official_abs"],
            "passed": bool(
                direct_sliding["parity_passed"] and fk_sliding["parity_passed"]
            ),
        },
    }


def per_sequence_array(result: Mapping[str, object], metric: str) -> np.ndarray:
    records = result["per_sequence"]
    return np.asarray([
        float(records[name][metric]) for name in sorted(records)
    ], dtype=np.float64)


def compare_frontier(results: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    control, midpoint, final = (
        results["control"], results["midpoint"], results["final"],
    )
    return {
        "final_minus_midpoint_fk_foot_sliding": paired_difference(
            per_sequence_array(final, "fk_foot_sliding"),
            per_sequence_array(midpoint, "fk_foot_sliding"),
        ),
        "midpoint_over_final_lower_is_better": {
            metric: paired_ratio(
                per_sequence_array(midpoint, metric), per_sequence_array(final, metric),
            )
            for metric in LOWER_METRICS
        },
        "midpoint_minus_final_contact_f1": paired_difference(
            per_sequence_array(midpoint, "physical_contact_f1"),
            per_sequence_array(final, "physical_contact_f1"),
        ),
        "control_minus_midpoint_improvement": {
            metric: paired_difference(
                per_sequence_array(control, metric), per_sequence_array(midpoint, metric),
            )
            for metric in CONTROL_IMPROVEMENT_METRICS
        },
        "descriptive": {
            "midpoint_over_control_fk_foot_sliding": paired_ratio(
                per_sequence_array(midpoint, "fk_foot_sliding"),
                per_sequence_array(control, "fk_foot_sliding"),
            ),
            "final_over_control_fk_foot_sliding": paired_ratio(
                per_sequence_array(final, "fk_foot_sliding"),
                per_sequence_array(control, "fk_foot_sliding"),
            ),
            "midpoint_minus_control_direct_fk_disagreement_cm": paired_difference(
                per_sequence_array(midpoint, "direct_fk_foot_disagreement_cm"),
                per_sequence_array(control, "direct_fk_foot_disagreement_cm"),
            ),
        },
    }


def classify_frontier(
    comparison: Mapping[str, object], *, contract_passed: bool,
) -> Dict[str, object]:
    checks = {
        "midpoint_has_lower_fk_foot_sliding_than_final": (
            comparison["final_minus_midpoint_fk_foot_sliding"]["bootstrap_95_ci"][0] > 0.0
        ),
        "midpoint_preserves_contact_f1": (
            comparison["midpoint_minus_final_contact_f1"]["bootstrap_95_ci"][0] >= -0.02
        ),
    }
    for metric in LOWER_METRICS:
        checks[f"midpoint_preserves_{metric}"] = (
            comparison["midpoint_over_final_lower_is_better"][metric]["bootstrap_95_ci"][1]
            <= 1.10
        )
    for metric in CONTROL_IMPROVEMENT_METRICS:
        checks[f"midpoint_improves_control_{metric}"] = (
            comparison["control_minus_midpoint_improvement"][metric]["bootstrap_95_ci"][0]
            > 0.0
        )
    gate_passed = bool(contract_passed and all(checks.values()))
    if not contract_passed:
        classification = "midbudget-protection-contract-failure-stop"
    elif gate_passed:
        classification = "midbudget-protection-supported-stop"
    else:
        classification = "midbudget-protection-negative-stop"
    return {
        "classification": classification,
        "contract_passed": bool(contract_passed),
        "gate_passed": gate_passed,
        "checks": checks,
        "checkpoint_selected": False,
        "training_authorized": False,
        "consistency_authorized": False,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "seed": 42,
        "repo_root": str(REPO),
        "execution_host": "infbagel-4gpu/node01",
        "python": str(Path(args.python).resolve()),
        "device": args.device,
        "checkpoints": {
            name: {
                "path": str(Path(getattr(args, f"checkpoint_{name}")).resolve()),
                **EXPECTED_CHECKPOINTS[name],
                "weight_variant": "online",
                "load": ["model"],
            }
            for name in EXPECTED_CHECKPOINTS
        },
        "selection": {
            "partition": "internal_validation",
            "eligible_sequence_ranks": [128, 159],
            "sequences": 32,
            "windows_per_sequence": 3,
            "selection_sha256": NATIVE_SELECTION_SHA256,
        },
        "sampling": {
            "diffusion_steps": 500,
            "paired_noise_across_checkpoints": True,
            "matched_conditions": True,
            "generated_history": True,
            "current_generated_object_bps": True,
            "cfg": False,
            "dynamic_perception": False,
            "guidance": False,
        },
        "statistics": {"paired_unit": "sequence", "replicates": 10_000, "seed": 42},
        "official_test_used": False,
        "chois_used": False,
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selection": False,
        "production_change": False,
        "consistency_authorized": False,
        "output": str(Path(args.output).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-control", required=True)
    parser.add_argument("--checkpoint-midpoint", required=True)
    parser.add_argument("--checkpoint-final", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def _validate_checkpoint_raw(name: str, path: Path) -> Dict[str, object]:
    expected = EXPECTED_CHECKPOINTS[name]
    actual_hash = sha256_file(path)
    if actual_hash != expected["file_sha256"]:
        raise ValueError(f"D2-W {name} checkpoint hash mismatch: {actual_hash}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "run_id": raw.get("run_id") == "p1-hoi-d2v-balanced-long-budget-s42-20260722",
        "processed_windows": int(raw.get("processed_windows", -1)) == expected["processed_windows"],
        "seed": int(raw.get("seed", -1)) == 42,
        "expert": raw.get("expert") == "hoi",
        "initialization": raw.get("initialization") == "random",
        "primary_weight_variant": raw.get("primary_weight_variant") == "online",
        "ema_empty": not raw.get("ema_models"),
        "data_contract": raw.get("data_contract_sha256") == EXPECTED_DATA_CONTRACT_SHA256,
        "model_state": state_dict_sha256(raw["model"]) == expected["model_state_sha256"],
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-W {name} checkpoint contract mismatch: {failed}")
    del raw
    return {"file_sha256": actual_hash, "checks": checks}


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-W run id must be {RUN_ID}")
    config = resolved_config(args)
    resolved_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(resolved_path, config)
        return
    if json.loads(resolved_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments differ from archived D2-W resolved config")
    if socket.gethostname() != "node01":
        raise RuntimeError("D2-W CUDA diagnostic is restricted to infbagel-4gpu/node01")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-W requires INFBAGEL_WORKER_EXPERT=hoi")
    if os.environ.get("INFBAGEL_PYTHON") != EXPECTED_PYTHON:
        raise RuntimeError("D2-W INFBAGEL_PYTHON mismatch")
    if str(Path(sys.executable).resolve()) != EXPECTED_PYTHON:
        raise RuntimeError("D2-W interpreter mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
    ).strip():
        raise RuntimeError("D2-W refuses a dirty worker checkout")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-W requires worker CUDA")
    checkpoint_paths = {
        name: Path(getattr(args, f"checkpoint_{name}")).resolve()
        for name in EXPECTED_CHECKPOINTS
    }
    checkpoint_contracts = {
        name: _validate_checkpoint_raw(name, path)
        for name, path in checkpoint_paths.items()
    }
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    selection = select_native_holdout(dataset)
    if selection["sha256"] != NATIVE_SELECTION_SHA256 or selection["sequences"] != 32:
        raise ValueError("D2-W native selection mismatch")
    triples = selection["triples"]
    rest_vertices = load_rest_vertices(dataset, triples, device)
    diffusion = GaussianDiffusion(500).to(device)
    torch.cuda.synchronize(device)
    started = time.time()
    results = {}
    metadata = {}
    for name in ("control", "midpoint", "final"):
        model, checkpoint_metadata = load_trained_hoi_prior(
            str(checkpoint_paths[name]), device, weight_variant="online",
        )
        if checkpoint_metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError(f"D2-W {name} loaded metadata contract mismatch")
        model.eval()
        results[name] = rollout_checkpoint(
            model, diffusion, dataset, triples, device, rest_vertices,
        )
        metadata[name] = checkpoint_metadata
        del model
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    runtime_seconds = time.time() - started
    noise_identity = all(
        results[name]["noise_sha256_by_step"] == results["control"]["noise_sha256_by_step"]
        for name in ("midpoint", "final")
    )
    parity_passed = all(results[name]["foot_sliding_parity"]["passed"] for name in results)
    finite = all(bool(results[name]["aggregate"]["finite"]) for name in results)
    contract_passed = bool(noise_identity and parity_passed and finite)
    comparison = compare_frontier(results)
    decision = classify_frontier(comparison, contract_passed=contract_passed)
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "eligible_sequence_ranks": [128, 159],
            "sequences": selection["sequences"],
            "windows": selection["sequences"] * 3,
            "sha256": selection["sha256"],
            "official_test_sequences": 0,
        },
        "checkpoint_contracts": checkpoint_contracts,
        "checkpoint_metadata": metadata,
        "results": results,
        "comparison": comparison,
        "decision": decision,
        "contract": {
            "noise_identity": noise_identity,
            "foot_sliding_parity": parity_passed,
            "finite": finite,
            "passed": contract_passed,
        },
        "runtime": {
            "seconds": runtime_seconds,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_synchronized_for_timing": True,
        },
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selected": False,
        "production_change": False,
        "released_checkpoint_loaded": False,
        "author_checkpoint_loaded": False,
        "official_test_used": False,
        "chois_used": False,
        "consistency_started": False,
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
