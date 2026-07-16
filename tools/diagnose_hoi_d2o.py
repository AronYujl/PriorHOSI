#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-O0 contact-alignment diagnostic."""

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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.contact_alignment import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DECOMPOSITION_PATHS,
    EXPECTED_CHECKPOINT_SHA256,
    HISTORY_MAX_ABS,
    MODELS,
    PHASE_OFFSETS,
    PHYSICAL_THRESHOLDS_CM,
    RUN_ID,
    SELECTION_SHA256,
    SEMANTIC_THRESHOLDS,
    WINDOWS_PER_SEQUENCE,
    all_finite,
    classification_gate,
    decomposition_report,
    distance_decomposition,
    geometry_report,
    hand_object_distances,
    object_vertices,
    sampler_seed_label,
    select_contact_holdout,
    semantic_geometry_report,
    semantic_report,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.optimizer_reset import paired_difference  # noqa: E402
from priors.window_codec import BPS_SHA256, project_to_so3  # noqa: E402
from tools.diagnose_hoi_remediation import (  # noqa: E402
    raw_window_target,
    seed_everything,
    stable_seed,
)
from tools.evaluate_hoi_remediation import (  # noqa: E402
    current_bps,
    global_goals,
    load_rest_vertices,
    stack_frames,
)


SUBPHASE = "1B-D2-O0"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_NORMALIZATION_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
DEFAULT_BATCH_SIZE = 16


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
        "mode": "contact-semantic-geometry-alignment-diagnostic",
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
        "selection": {
            "partition": "internal_validation",
            "phase_offsets": list(PHASE_OFFSETS),
            "sequences": 64,
            "windows_per_sequence": WINDOWS_PER_SEQUENCE,
            "windows": 192,
            "global_window_indices_sha256": SELECTION_SHA256,
        },
        "evaluation": {
            "diffusion_steps": 500,
            "condition_variant": "matched",
            "shared_sampler_noise": True,
            "semantic_channels": [0, 1, 2, 3],
            "hand_semantic_channels": [0, 1],
            "semantic_thresholds": list(SEMANTIC_THRESHOLDS),
            "physical_thresholds_cm": list(PHYSICAL_THRESHOLDS_CM),
            "hand_joint_indices": [24, 26],
            "distance_decomposition": list(DECOMPOSITION_PATHS),
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
        },
        "sampler_contract": {
            "production_equation_changed": False,
            "future_gt": False,
            "stored_per_frame_bps": False,
            "rollout_bps": "recomputed_from_current_generated_object_pose",
            "object_so3_projection": "unchanged_production_post-sample_projection",
            "cfg": False,
        },
        "released_checkpoint_loaded": False,
        "ema_used": False,
        "checkpoint_selection": False,
        "optimizer_created": False,
        "training_updates": 0,
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
) -> List[Dict[str, object]]:
    result = []
    device = torch.device("cpu")
    for triple in triples:
        targets = [raw_window_target(dataset, position, device) for position in triple]
        name = _sequence_name(dataset, triple[0])
        result.append({
            "sequence": name,
            "object_category": name.split("_")[1],
            "positions": list(triple),
            "pi": [
                int(dataset.language["pi"][int(dataset.indices[position])])
                for position in triple
            ],
            "joints": torch.cat([value["joints"][2:] for value in targets]),
            "object_translation": torch.cat([
                value["object_translation"][2:] for value in targets
            ]),
            "object_rotation": torch.cat([
                value["object_rotation"][2:] for value in targets
            ]),
            "contact": torch.cat([value["contact"][2:] for value in targets]),
        })
    return result


def _frame_lists(
    *,
    gt_semantic: np.ndarray,
    gt_distance: np.ndarray,
    predicted_semantic: np.ndarray | None = None,
    predicted_distance: np.ndarray | None = None,
    decomposition: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    value: Dict[str, object] = {
        "gt_semantic_labels": gt_semantic.tolist(),
        "gt_hand_object_distance_m": gt_distance.tolist(),
    }
    if predicted_semantic is not None:
        value["predicted_semantic"] = predicted_semantic.tolist()
    if predicted_distance is not None:
        value["predicted_hand_object_distance_m"] = predicted_distance.tolist()
    if decomposition is not None:
        value["distance_decomposition_m"] = {
            name: decomposition[name].tolist() for name in DECOMPOSITION_PATHS
        }
    return value


def _concatenate_frames(
    records: Sequence[Mapping[str, object]],
    key: str,
) -> np.ndarray:
    return np.concatenate([
        np.asarray(record["per_frame"][key], dtype=np.float64)
        for record in records
    ])


def ground_truth_records(
    targets: Sequence[Mapping[str, object]],
    rest_vertices: Mapping[str, torch.Tensor],
    device: torch.device,
) -> List[Dict[str, object]]:
    records = []
    for target in targets:
        category = str(target["object_category"])
        joints = target["joints"].to(device)
        translation = target["object_translation"].to(device)
        rotation = target["object_rotation"].to(device)
        semantic = target["contact"].cpu().numpy().astype(np.float64)
        vertices = object_vertices(rest_vertices[category], rotation, translation)
        distance = hand_object_distances(joints, vertices).cpu().numpy().astype(np.float64)
        records.append({
            "sequence": target["sequence"],
            "object_category": category,
            "positions": target["positions"],
            "pi": target["pi"],
            "per_frame": _frame_lists(gt_semantic=semantic, gt_distance=distance),
            "semantic_geometry_alignment": semantic_geometry_report(semantic, distance),
        })
    return records


def ground_truth_summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    semantic = _concatenate_frames(records, "gt_semantic_labels")
    distance = _concatenate_frames(records, "gt_hand_object_distance_m")
    categories = sorted({str(record["object_category"]) for record in records})
    return {
        "frames": int(len(semantic)),
        "contact_label_prevalence": {
            str(channel): float((semantic[:, channel] >= 0.5).mean())
            for channel in range(4)
        },
        "semantic_geometry_alignment": semantic_geometry_report(semantic, distance),
        "by_object_category": {
            category: ground_truth_summary([
                record for record in records
                if record["object_category"] == category
            ]) | {"by_object_category": {}}
            for category in categories
        } if len(categories) > 1 else {},
    }


@torch.no_grad()
def rollout_chunk(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
    rest_vertices: Mapping[str, torch.Tensor],
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
    noise_streams = []
    history_max_abs = 0.0
    for step in range(WINDOWS_PER_SEQUENCE):
        items = items_by_step[step]
        gt_frame = stack_frames(items, device)
        pelvis_global, object_global = global_goals(dataset, items, gt_frame, device)
        goals = torch.zeros(len(triples), 9, device=device)
        goals[:, :3] = dataset.codec.pelvis_goal(pelvis_global, frame)
        goals[:, 6:9] = dataset.codec.object_goal(object_global, frame)
        text = torch.stack([item["text_embedding"] for item in items]).to(device)
        bps = current_bps(dataset, frame.object_reference, names, rest_vertices)
        progress = normalize_progress(
            torch.stack([item["progress"] for item in items]).to(device)
        )
        label = sampler_seed_label(chunk_index, step)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(label))
        initial_state = sha256_tensor_state(generator.get_state())
        sample = diffusion.sample(
            model, fixed, text, bps, goals, progress, generator=generator,
        )
        final_state = sha256_tensor_state(generator.get_state())
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        history_max_abs = max(
            history_max_abs,
            float((sample[:, :2] - fixed).abs().max().detach().cpu()),
        )
        decoded = dataset.codec.decode(sample, frame)
        decoded_steps.append({
            key: value.detach()
            for key, value in decoded.items()
        })
        noise_streams.append({
            "chunk_index": chunk_index,
            "step": step,
            "label": label,
            "seed": stable_seed(label),
            "generator_initial_state_sha256": initial_state,
            "generator_final_state_sha256": final_state,
        })
        if step < WINDOWS_PER_SEQUENCE - 1:
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
                decoded_steps[step][key][row, 2:].cpu()
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
    return {
        "generated": generated,
        "noise_streams": noise_streams,
        "history_max_abs": history_max_abs,
    }


def analyze_generated_sequence(
    target: Mapping[str, object],
    generated: Mapping[str, torch.Tensor],
    rest_vertices: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    category = str(target["object_category"])
    gt_joints = target["joints"].to(device)
    gt_translation = target["object_translation"].to(device)
    gt_rotation = target["object_rotation"].to(device)
    gt_semantic = target["contact"].cpu().numpy().astype(np.float64)
    generated_joints = generated["joints"].to(device)
    generated_translation = generated["object_translation"].to(device)
    generated_rotation = generated["object_rotation"].to(device)
    predicted_semantic = generated["contact"].cpu().numpy().astype(np.float64)
    gt_vertices = object_vertices(
        rest_vertices[category], gt_rotation, gt_translation,
    )
    generated_vertices = object_vertices(
        rest_vertices[category], generated_rotation, generated_translation,
    )
    decomposition_tensors = distance_decomposition(
        generated_joints, gt_joints, generated_vertices, gt_vertices,
    )
    decomposition = {
        name: value.cpu().numpy().astype(np.float64)
        for name, value in decomposition_tensors.items()
    }
    gt_distance = decomposition["gt_human_gt_object"]
    predicted_distance = decomposition["generated_human_generated_object"]
    semantic = semantic_report(predicted_semantic, gt_semantic)
    geometry = geometry_report(predicted_distance, gt_distance)
    return {
        "sequence": target["sequence"],
        "object_category": category,
        "positions": target["positions"],
        "pi": target["pi"],
        "per_frame": _frame_lists(
            gt_semantic=gt_semantic,
            gt_distance=gt_distance,
            predicted_semantic=predicted_semantic,
            predicted_distance=predicted_distance,
            decomposition=decomposition,
        ),
        "semantic_vs_gt": semantic,
        "physical_geometry_vs_gt": geometry,
        "predicted_semantic_vs_predicted_geometry": semantic_geometry_report(
            predicted_semantic, predicted_distance,
        ),
        "distance_decomposition_on_gt_5cm_contact": decomposition_report(
            decomposition, gt_distance,
        ),
    }


def model_summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    prediction_semantic = _concatenate_frames(records, "predicted_semantic")
    gt_semantic = _concatenate_frames(records, "gt_semantic_labels")
    prediction_distance = _concatenate_frames(
        records, "predicted_hand_object_distance_m",
    )
    gt_distance = _concatenate_frames(records, "gt_hand_object_distance_m")
    decomposition = {
        name: np.concatenate([
            np.asarray(
                record["per_frame"]["distance_decomposition_m"][name],
                dtype=np.float64,
            )
            for record in records
        ])
        for name in DECOMPOSITION_PATHS
    }
    categories = sorted({str(record["object_category"]) for record in records})
    result = {
        "frames": int(len(prediction_semantic)),
        "semantic_vs_gt": semantic_report(prediction_semantic, gt_semantic),
        "physical_geometry_vs_gt": geometry_report(
            prediction_distance, gt_distance,
        ),
        "predicted_semantic_vs_predicted_geometry": semantic_geometry_report(
            prediction_semantic, prediction_distance,
        ),
        "distance_decomposition_on_gt_5cm_contact": decomposition_report(
            decomposition, gt_distance,
        ),
        "by_object_category": {},
    }
    for category in categories:
        selected = [
            record for record in records
            if record["object_category"] == category
        ]
        category_prediction_semantic = _concatenate_frames(
            selected, "predicted_semantic",
        )
        category_gt_semantic = _concatenate_frames(
            selected, "gt_semantic_labels",
        )
        category_prediction_distance = _concatenate_frames(
            selected, "predicted_hand_object_distance_m",
        )
        category_gt_distance = _concatenate_frames(
            selected, "gt_hand_object_distance_m",
        )
        category_decomposition = {
            name: np.concatenate([
                np.asarray(
                    record["per_frame"]["distance_decomposition_m"][name],
                    dtype=np.float64,
                )
                for record in selected
            ])
            for name in DECOMPOSITION_PATHS
        }
        result["by_object_category"][category] = {
            "sequences": len(selected),
            "frames": int(len(category_prediction_semantic)),
            "semantic_vs_gt": semantic_report(
                category_prediction_semantic, category_gt_semantic,
            ),
            "physical_geometry_vs_gt": geometry_report(
                category_prediction_distance, category_gt_distance,
            ),
            "predicted_semantic_vs_predicted_geometry": semantic_geometry_report(
                category_prediction_semantic, category_prediction_distance,
            ),
            "distance_decomposition_on_gt_5cm_contact": decomposition_report(
                category_decomposition, category_gt_distance,
            ),
        }
    return result


def _per_sequence_metric(
    records: Sequence[Mapping[str, object]],
    path: Sequence[str],
) -> np.ndarray:
    values = []
    for record in records:
        current: object = record
        for key in path:
            current = current[key]  # type: ignore[index]
        values.append(float(current))
    return np.asarray(values, dtype=np.float64)


def paired_comparisons(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    result = {}
    balanced_semantic = _per_sequence_metric(
        records["balanced"], ("semantic_vs_gt", "first_two_mse"),
    )
    balanced_recall = _per_sequence_metric(
        records["balanced"],
        ("physical_geometry_vs_gt", "thresholds_cm", "5", "union", "recall"),
    )
    balanced_f1 = _per_sequence_metric(
        records["balanced"],
        ("physical_geometry_vs_gt", "thresholds_cm", "5", "union", "f1"),
    )
    balanced_percent = _per_sequence_metric(
        records["balanced"],
        (
            "physical_geometry_vs_gt", "thresholds_cm", "5", "union",
            "prediction_percent",
        ),
    )
    for comparator in ("source", "current"):
        comparator_semantic = _per_sequence_metric(
            records[comparator], ("semantic_vs_gt", "first_two_mse"),
        )
        comparator_recall = _per_sequence_metric(
            records[comparator],
            ("physical_geometry_vs_gt", "thresholds_cm", "5", "union", "recall"),
        )
        comparator_f1 = _per_sequence_metric(
            records[comparator],
            ("physical_geometry_vs_gt", "thresholds_cm", "5", "union", "f1"),
        )
        comparator_percent = _per_sequence_metric(
            records[comparator],
            (
                "physical_geometry_vs_gt", "thresholds_cm", "5", "union",
                "prediction_percent",
            ),
        )
        result[f"balanced_vs_{comparator}"] = {
            "comparator_minus_balanced_semantic_first_two_mse": paired_difference(
                comparator_semantic, balanced_semantic,
            ),
            "comparator_minus_balanced_physical_recall_5cm": paired_difference(
                comparator_recall, balanced_recall,
            ),
            "comparator_minus_balanced_physical_f1_5cm": paired_difference(
                comparator_f1, balanced_f1,
            ),
            "comparator_minus_balanced_physical_contact_percent_5cm": paired_difference(
                comparator_percent, balanced_percent,
            ),
        }
    return result


def reports_complete(value: Mapping[str, object]) -> bool:
    semantic = value["semantic_vs_gt"]
    geometry = value["physical_geometry_vs_gt"]
    alignment = value["predicted_semantic_vs_predicted_geometry"]
    return bool(
        set(semantic["per_channel"]) == {"0", "1", "2", "3"}
        and set(semantic["thresholds"])
        == {f"{threshold:g}" for threshold in SEMANTIC_THRESHOLDS}
        and set(geometry["thresholds_cm"])
        == {f"{threshold:g}" for threshold in PHYSICAL_THRESHOLDS_CM}
        and set(alignment)
        == {f"{threshold:g}" for threshold in PHYSICAL_THRESHOLDS_CM}
        and set(value["distance_decomposition_on_gt_5cm_contact"])
        == set(DECOMPOSITION_PATHS)
    )


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
        raise ValueError(f"D2-O0 run id must be {RUN_ID}")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-O0 batch size must evenly divide 64")
    if checkpoint_hashes(args) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("D2-O0 requested checkpoint hashes differ from preregistration")
    config = resolved_config(args)
    config_path = args.resolved_config.resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-O0 runtime arguments do not match archived resolved config")
    if Path(sys.executable).resolve() != Path(
        os.environ.get("INFBAGEL_PYTHON", ""),
    ).resolve():
        raise ValueError("D2-O0 requires the absolute INFBAGEL_PYTHON interpreter")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-O0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if git_output("status", "--porcelain"):
        raise RuntimeError("D2-O0 refuses a dirty worker checkout")
    paths = checkpoint_paths(args)
    for name in MODELS:
        actual = sha256_file(paths[name])
        if actual != EXPECTED_CHECKPOINT_SHA256[name]:
            raise ValueError(f"D2-O0 {name} checkpoint hash mismatch: {actual}")
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
            f"D2-O0 asset hash mismatch: {asset_hashes} != {expected_asset_hashes}"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-O0 is a four-GPU-worker CUDA diagnostic")
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
    selection = select_contact_holdout(dataset)
    triples = selection["triples"]
    targets = prepare_targets(dataset, triples)
    rest_vertices = load_rest_vertices(dataset, triples, device)
    gt_records = ground_truth_records(targets, rest_vertices, device)
    gt_summary = ground_truth_summary(gt_records)
    diffusion = GaussianDiffusion(500).to(device)
    model_records: Dict[str, List[Dict[str, object]]] = {}
    models: Dict[str, object] = {}
    for name in MODELS:
        model, metadata = load_trained_hoi_prior(
            str(paths[name]), device, weight_variant="online",
        )
        if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError(f"D2-O0 {name} data-contract mismatch")
        model.eval()
        model_before = state_dict_sha256(model)
        records = []
        noise_streams = []
        history_max_abs = 0.0
        for chunk_index, offset in enumerate(range(0, len(triples), args.batch_size)):
            selected_triples = triples[offset:offset + args.batch_size]
            rollout = rollout_chunk(
                model,
                diffusion,
                dataset,
                selected_triples,
                device,
                rest_vertices,
                chunk_index=chunk_index,
            )
            noise_streams.extend(rollout["noise_streams"])
            history_max_abs = max(
                history_max_abs, float(rollout["history_max_abs"]),
            )
            for target, generated in zip(
                targets[offset:offset + args.batch_size],
                rollout["generated"],
            ):
                records.append(analyze_generated_sequence(
                    target, generated, rest_vertices, device,
                ))
        summary = model_summary(records)
        model_after = state_dict_sha256(model)
        model_records[name] = records
        models[name] = {
            "checkpoint": {
                "path": str(paths[name]),
                "sha256": EXPECTED_CHECKPOINT_SHA256[name],
                "metadata": metadata,
                "model_state_sha256_before": model_before,
                "model_state_sha256_after": model_after,
                "model_state_unchanged": model_before == model_after,
            },
            "history_max_abs": history_max_abs,
            "noise_streams": noise_streams,
            "aggregate": summary,
            "per_sequence": records,
            "finite": all_finite(summary) and all_finite(records),
            "all_contact_fields_thresholds_and_decomposition_reported": (
                reports_complete(summary)
                and all(reports_complete(record) for record in records)
            ),
        }
        del model
        torch.cuda.empty_cache()
    comparisons = paired_comparisons(model_records)
    noise_identity = all(
        models[name]["noise_streams"] == models["source"]["noise_streams"]
        for name in MODELS[1:]
    )
    contract = {
        "checkpoint_hashes_exact": True,
        "asset_hashes_exact": asset_hashes == expected_asset_hashes,
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
        "shared_sampler_noise_identity": noise_identity,
        "history_restoration": all(
            float(models[name]["history_max_abs"]) <= HISTORY_MAX_ABS
            for name in MODELS
        ),
        "all_finite": all(bool(models[name]["finite"]) for name in MODELS),
        "all_checkpoints_reported": set(models) == set(MODELS),
        "all_fields_thresholds_decomposition_reported": all(
            bool(models[name]["all_contact_fields_thresholds_and_decomposition_reported"])
            for name in MODELS
        ),
        "model_state_unchanged": all(
            bool(models[name]["checkpoint"]["model_state_unchanged"])
            for name in MODELS
        ),
        "sampler_future_gt_absent": True,
        "sampler_stored_per_frame_bps_absent": True,
        "production_sampler_equation_unchanged": True,
    }
    decision = classification_gate(
        contract,
        gt_summary["semantic_geometry_alignment"],
        comparisons,
    )
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
        },
        "ground_truth": {
            "aggregate": gt_summary,
            "per_sequence": gt_records,
        },
        "models": models,
        "comparisons": comparisons,
        "contract": contract,
        "decision": decision,
        "sampler_contract": {
            "future_gt": False,
            "stored_per_frame_bps": False,
            "rollout_bps": "recomputed_from_current_generated_object_pose",
            "production_equation_changed": False,
            "shared_noise_identity": noise_identity,
        },
        "training_updates": 0,
        "optimizer_created": False,
        "checkpoint_write": False,
        "released_checkpoint_used": False,
        "ema_used": False,
        "checkpoint_selection": False,
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
