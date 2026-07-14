#!/usr/bin/env python3
"""Evaluate D2 checkpoints on the fixed internal rollout and condition gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import trimesh


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import zup_to_yup  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.remediation import select_internal_triples, selection_sha256  # noqa: E402
from priors.window_codec import WindowFrame, project_to_so3  # noqa: E402
from tools.diagnose_hoi_remediation import physical_summary, stable_seed  # noqa: E402


WEIGHT_VARIANTS = ("online", "ema_0.999", "ema_0.9999")
CONDITION_VARIANTS = ("matched", "text_permuted", "bps_permuted", "pelvis_permuted", "object_goal_permuted")
PRIMARY_ERROR = {
    "text_permuted": "mpjpe_cm",
    "bps_permuted": "object_translation_mae_cm",
    "pelvis_permuted": "pelvis_goal_error_cm",
    "object_goal_permuted": "object_goal_error_cm",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def stack_frames(items: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> WindowFrame:
    return WindowFrame(
        torch.stack([item["window_origin"] for item in items]).to(device),
        torch.stack([item["world_to_local_rotation"] for item in items]).to(device),
        torch.stack([item["object_rotation_reference"] for item in items]).to(device),
    )


def global_goals(dataset, items, frames, device):
    pelvis, objects = [], []
    object_minimum = dataset.codec.object_minimum.to(device)
    object_maximum = dataset.codec.object_maximum.to(device)
    for item, frame in zip(items, [
        WindowFrame(frames.origin[row], frames.world_to_local[row], frames.object_reference[row])
        for row in range(len(items))
    ]):
        local_pelvis = item["goals"][:3].to(device)
        pelvis.append(dataset.codec.global_position(local_pelvis[None], frame)[0])
        normalized = item["goals"][6:9].to(device)
        local_object = dataset.codec._denormalize(normalized, object_minimum, object_maximum)
        objects.append(dataset.codec.global_position(local_object[None], frame)[0])
    return torch.stack(pelvis), torch.stack(objects)


def load_rest_vertices(dataset, triples, device):
    result = {}
    for triple in triples:
        position = triple[0]
        index = int(dataset.indices[position])
        name = str(dataset.scene_names[int(dataset.sequence_ids[index])])
        object_name = name.split("_")[1]
        if object_name in result:
            continue
        mesh = trimesh.load_mesh(REPO / "data/object/rest_object_geo" / f"{object_name}.ply", process=False)
        result[object_name] = torch.from_numpy(
            zup_to_yup(np.asarray(mesh.vertices, dtype=np.float32).copy())
        ).to(device)
    return result


def current_bps(dataset, references, names, rest_vertices, chunk_size=8):
    result = torch.empty(references.shape[0], 1024, 3, device=references.device)
    groups: Dict[str, List[int]] = {}
    for row, name in enumerate(names):
        groups.setdefault(name.split("_")[1], []).append(row)
    for object_name, rows in groups.items():
        rest = rest_vertices[object_name]
        for offset in range(0, len(rows), chunk_size):
            selected = rows[offset:offset + chunk_size]
            vertices = rest[None].expand(len(selected), -1, -1)
            result[selected] = dataset.codec.recompute_bps(vertices, references[selected])
    return result


@torch.no_grad()
def rollout(model, diffusion, dataset, triples, device, weight_variant, condition_variant, rest_vertices):
    positions_by_step = [[triple[step] for triple in triples] for step in range(3)]
    items_by_step = [[dataset[position] for position in positions] for positions in positions_by_step]
    names = [
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
        for position in positions_by_step[0]
    ]
    permutation = torch.roll(torch.arange(len(triples), device=device), shifts=1)
    first_items = items_by_step[0]
    frame = stack_frames(first_items, device)
    fixed = torch.stack([item["x"][:2] for item in first_items]).to(device)
    decoded_steps = []
    for step in range(3):
        items = items_by_step[step]
        gt_frame = stack_frames(items, device)
        pelvis_global, object_global = global_goals(dataset, items, gt_frame, device)
        if condition_variant == "pelvis_permuted":
            pelvis_global = pelvis_global[permutation]
        if condition_variant == "object_goal_permuted":
            object_global = object_global[permutation]
        goals = torch.zeros(len(triples), 9, device=device)
        goals[:, :3] = dataset.codec.pelvis_goal(pelvis_global, frame)
        goals[:, 6:9] = dataset.codec.object_goal(object_global, frame)
        text = torch.stack([item["text_embedding"] for item in items]).to(device)
        bps = current_bps(dataset, frame.object_reference, names, rest_vertices)
        if condition_variant == "text_permuted":
            text = text[permutation]
        if condition_variant == "bps_permuted":
            bps = bps[permutation]
        progress = normalize_progress(torch.stack([item["progress"] for item in items]).to(device))
        generator = torch.Generator(device=device)
        # Matched/permuted comparisons share the exact diffusion noise stream.
        generator.manual_seed(stable_seed(f"D2:{weight_variant}:paired:{step}"))
        sample = diffusion.sample(model, fixed, text, bps, goals, progress, generator=generator)
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        decoded = dataset.codec.decode(sample, frame)
        decoded_steps.append(decoded)
        if step < 2:
            fixed, frame = dataset.codec.encode(
                decoded["joints"][:, -2:], decoded["human_rotation"][:, -2:],
                global_object_translation=decoded["object_translation"][:, -2:],
                global_object_rotation=decoded["object_rotation"][:, -2:],
                contact=decoded["contact"][:, -2:],
            )
    return physical_summary(dataset, triples, decoded_steps, device)


@torch.no_grad()
def rollout_all_conditions(model, diffusion, dataset, triples, device, weight_variant, rest_vertices):
    """Evaluate all paired condition variants in one batched diffusion trajectory."""
    base_batch = len(triples)
    repeat_count = len(CONDITION_VARIANTS)
    positions_by_step = [[triple[step] for triple in triples] for step in range(3)]
    items_by_step = [[dataset[position] for position in positions] for positions in positions_by_step]
    names = [
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
        for position in positions_by_step[0]
    ]
    names_repeated = names * repeat_count
    permutation = torch.roll(torch.arange(base_batch, device=device), shifts=1)
    first_items = items_by_step[0]
    first_frame = stack_frames(first_items, device)
    frame = WindowFrame(
        first_frame.origin.repeat(repeat_count, 1),
        first_frame.world_to_local.repeat(repeat_count, 1, 1),
        first_frame.object_reference.repeat(repeat_count, 1, 1),
    )
    fixed = torch.stack([item["x"][:2] for item in first_items]).to(device).repeat(repeat_count, 1, 1)
    decoded_by_variant = {variant: [] for variant in CONDITION_VARIANTS}
    for step in range(3):
        items = items_by_step[step]
        gt_frame = stack_frames(items, device)
        pelvis_global, object_global = global_goals(dataset, items, gt_frame, device)
        text_base = torch.stack([item["text_embedding"] for item in items]).to(device)
        progress_base = normalize_progress(torch.stack([item["progress"] for item in items]).to(device))
        text = text_base.repeat(repeat_count, 1)
        progress = progress_base.repeat(repeat_count, 1)
        bps = current_bps(dataset, frame.object_reference, names_repeated, rest_vertices)
        goals = torch.zeros(base_batch * repeat_count, 9, device=device)
        for variant_index, variant in enumerate(CONDITION_VARIANTS):
            selected = slice(variant_index * base_batch, (variant_index + 1) * base_batch)
            variant_pelvis = pelvis_global[permutation] if variant == "pelvis_permuted" else pelvis_global
            variant_object = object_global[permutation] if variant == "object_goal_permuted" else object_global
            variant_frame = WindowFrame(
                frame.origin[selected], frame.world_to_local[selected], frame.object_reference[selected],
            )
            goals[selected, :3] = dataset.codec.pelvis_goal(variant_pelvis, variant_frame)
            goals[selected, 6:9] = dataset.codec.object_goal(variant_object, variant_frame)
            if variant == "text_permuted":
                text[selected] = text_base[permutation]
            if variant == "bps_permuted":
                bps[selected] = bps[selected][permutation]
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(f"D2:{weight_variant}:paired:{step}"))
        sample = diffusion.sample(
            model, fixed, text, bps, goals, progress, generator=generator,
            paired_repeats=repeat_count,
        )
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(base_batch * repeat_count, 16, 3, 3)
        ).reshape(base_batch * repeat_count, 16, 9)
        decoded = dataset.codec.decode(sample, frame)
        for variant_index, variant in enumerate(CONDITION_VARIANTS):
            selected = slice(variant_index * base_batch, (variant_index + 1) * base_batch)
            decoded_by_variant[variant].append({
                key: value[selected] for key, value in decoded.items()
            })
        if step < 2:
            fixed, frame = dataset.codec.encode(
                decoded["joints"][:, -2:], decoded["human_rotation"][:, -2:],
                global_object_translation=decoded["object_translation"][:, -2:],
                global_object_rotation=decoded["object_rotation"][:, -2:],
                contact=decoded["contact"][:, -2:],
            )
    return {
        variant: physical_summary(dataset, triples, decoded_by_variant[variant], device)
        for variant in CONDITION_VARIANTS
    }


def per_sequence_error(metrics, key):
    grouped: Dict[str, List[float]] = {}
    for value in metrics["per_sequence_window"]:
        grouped.setdefault(value["sequence"], []).append(float(value[key]))
    return np.asarray([np.mean(grouped[name]) for name in sorted(grouped)], dtype=np.float64)


def paired_bootstrap(matched, permuted, seed=42, replicates=10000):
    difference = np.asarray(permuted) - np.asarray(matched)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    samples = difference[indices].mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "paired_mean_permuted_minus_matched": float(difference.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "replicates": replicates,
        "seed": seed,
        "matched_significantly_better": bool(lower > 0.0),
    }


def eligibility(matched, baseline, sensitivity):
    current = matched["aggregate"]
    old = baseline["aggregate"]
    ratios = {
        "object_goal": current["object_goal_error_cm"] / old["object_goal_error_cm"],
        "pelvis_goal": current["pelvis_goal_error_cm"] / old["pelvis_goal_error_cm"],
        "mpjpe": current["mpjpe_cm"] / old["mpjpe_cm"],
        "foot_sliding": current["foot_sliding"] / old["foot_sliding"],
    }
    checks = {
        "object_goal_ratio_le_0.70": ratios["object_goal"] <= 0.70,
        "pelvis_goal_ratio_le_0.70": ratios["pelvis_goal"] <= 0.70,
        "contact_f1_increase_ge_0.10": (
            current["physical_contact_f1"] - old["physical_contact_f1"] >= 0.10
        ),
        "mpjpe_ratio_le_1.10": ratios["mpjpe"] <= 1.10,
        "foot_sliding_ratio_le_1.10": ratios["foot_sliding"] <= 1.10,
        "finite": bool(current["finite"]),
        "all_conditions_significant": all(
            value["matched_significantly_better"] for value in sensitivity.values()
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "ratios": ratios,
        "contact_f1_increase": current["physical_contact_f1"] - old["physical_contact_f1"],
    }


def resolved_config(args):
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B",
        "mode": "D2-internal-rollout",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "baseline_d0": str(Path(args.baseline_d0).resolve()),
        "partition": "internal_validation",
        "sequence_count": 128,
        "windows_per_sequence": 3,
        "weight_variants": list(WEIGHT_VARIANTS),
        "condition_variants": list(CONDITION_VARIANTS),
        "bootstrap_replicates": 10000,
        "official_test_used": False,
        "chois_used": False,
        "device": args.device,
        "output": str(Path(args.output).resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--baseline-d0", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match archived resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2 evaluation requires INFBAGEL_WORKER_EXPERT=hoi")
    checkpoint_path = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint_path) != args.checkpoint_sha256:
        raise ValueError("D2 checkpoint hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2 evaluation is a worker CUDA workload")
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    triples = select_internal_triples(dataset, 128)
    names = [
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[triple[0]])])])
        for triple in triples
    ]
    rest_vertices = load_rest_vertices(dataset, triples, device)
    diffusion = GaussianDiffusion(500).to(device)
    baseline_payload = json.loads(Path(args.baseline_d0).read_text(encoding="utf-8"))
    baseline = baseline_payload["weights"]["online"]["generation"]["three_window_generated_history_legacy"]
    output_weights = {}
    started = time.time()
    for weight_variant in WEIGHT_VARIANTS:
        model, metadata = load_trained_hoi_prior(
            str(checkpoint_path), device, weight_variant=weight_variant,
        )
        variants = rollout_all_conditions(
            model, diffusion, dataset, triples, device, weight_variant, rest_vertices,
        )
        matched = variants["matched"]
        sensitivity = {}
        for condition_variant, key in PRIMARY_ERROR.items():
            sensitivity[condition_variant] = paired_bootstrap(
                per_sequence_error(matched, key),
                per_sequence_error(variants[condition_variant], key),
            )
            sensitivity[condition_variant]["primary_error"] = key
        output_weights[weight_variant] = {
            "checkpoint": metadata,
            "rollout": variants,
            "condition_sensitivity": sensitivity,
            "eligibility": eligibility(matched, baseline, sensitivity),
        }
        del model
        torch.cuda.empty_cache()
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B",
        "seed": 42,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "selection": {
            "partition": "internal_validation",
            "official_test_sequence_count": 0,
            "chois_sequence_count": 0,
            "sequence_count": 128,
            "sequence_selection_sha256": selection_sha256(names),
        },
        "baseline": {
            "run_id": baseline_payload["run_id"],
            "checkpoint_sha256": baseline_payload["checkpoint"]["sha256"],
            "metrics": baseline["aggregate"],
        },
        "weights": output_weights,
        "runtime": {
            "seconds": time.time() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "throughput_not_used_for_selection": True,
        },
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
