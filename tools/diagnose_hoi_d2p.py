#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-P internal-only mechanism audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, _extract, normalize_progress  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.remediation import (  # noqa: E402
    D0_TIMESTEPS,
    deterministic_derangement,
    select_internal_triples,
    select_teacher_windows,
    selection_sha256,
)
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import WindowFrame, project_to_so3  # noqa: E402
from tools.diagnose_hoi_remediation import stack_items, stable_seed  # noqa: E402
from tools.evaluate_hoi_remediation import (  # noqa: E402
    current_bps,
    global_goals,
    load_rest_vertices,
    paired_bootstrap,
    stack_frames,
)


EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_CHECKPOINTS = {
    "R-1024": "d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23",
    "R-3072": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
}
CONDITION_VARIANTS = (
    "matched", "text_permuted", "bps_permuted", "pelvis_permuted", "object_goal_permuted",
)
PRIMARY_FIELD = {
    "text_permuted": "joint_positions",
    "bps_permuted": "object_translation",
    "pelvis_permuted": "joint_positions",
    "object_goal_permuted": "object_translation",
}
TRACE_STEPS = (499, 250, 100, 50, 10, 1, 0)
D0_T499 = {
    "joint_positions": 0.0024204085348173976,
    "object_translation": 0.00893084186827764,
}
D2_THRESHOLDS = {
    "object_goal_error_cm": 11.441127241589129 * 0.70,
    "pelvis_goal_error_cm": 45.29992023669183 * 0.70,
    "mpjpe_cm": 33.487335205078125 * 1.10,
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


def field_error_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return one non-history MSE per sample and representation field."""
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(f"expected matching [B,16,232], got {prediction.shape}/{target.shape}")
    return {
        field.name: (
            prediction[:, REPRESENTATION.history_frames:, field.slice]
            - target[:, REPRESENTATION.history_frames:, field.slice]
        ).square().flatten(1).mean(dim=1)
        for field in REPRESENTATION.fields
    }


def field_range(value: torch.Tensor) -> Dict[str, Dict[str, float]]:
    result = {}
    for field in REPRESENTATION.fields:
        selected = value[:, REPRESENTATION.history_frames:, field.slice]
        result[field.name] = {
            "minimum": float(selected.min()),
            "maximum": float(selected.max()),
            "absolute_maximum": float(selected.abs().max()),
            "rms": float(selected.square().mean().sqrt()),
            "nonfinite": int((~torch.isfinite(selected)).sum()),
        }
    return result


def _row_frame(frames: WindowFrame, row: int) -> WindowFrame:
    return WindowFrame(frames.origin[row], frames.world_to_local[row], frames.object_reference[row])


@torch.no_grad()
def contract_replay(dataset: PriorWindowDataset, triples, device: torch.device) -> Dict[str, object]:
    """Replay every coordinate boundary used by D2 before any checkpoint is loaded."""
    positions = [triple[0] for triple in triples[:32]]
    items = [dataset[position] for position in positions]
    frames = stack_frames(items, device)
    pelvis_global, object_global = global_goals(dataset, items, frames, device)
    names = [
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
        for position in positions
    ]
    rest_vertices = load_rest_vertices(dataset, triples[:32], device)
    replay_bps = current_bps(dataset, frames.object_reference, names, rest_vertices)
    stored_bps = torch.stack([item["object_bps"] for item in items]).to(device)
    bps_error = (replay_bps - stored_bps).abs()
    tolerances = {
        "pelvis_goal_max_abs": 1e-5,
        "object_goal_max_abs": 1e-5,
        "metric_object_target_max_abs": 1e-5,
        "history_max_abs": 1e-5,
        "bps_max_abs": 1e-4,
    }
    maxima = {
        "pelvis_goal_max_abs": 0.0,
        "object_goal_max_abs": 0.0,
        "metric_object_target_max_abs": 0.0,
        "history_max_abs": 0.0,
        "bps_max_abs": float(bps_error.max()),
    }
    failures = []
    for row, (position, item) in enumerate(zip(positions, items)):
        frame = _row_frame(frames, row)
        goals = item["goals"].to(device)
        replay_pelvis = dataset.codec.pelvis_goal(pelvis_global[row], frame)
        replay_object = dataset.codec.object_goal(object_global[row], frame)
        row_errors = {
            "pelvis_goal_max_abs": float((replay_pelvis - goals[:3]).abs().max()),
            "object_goal_max_abs": float((replay_object - goals[6:9]).abs().max()),
            "bps_max_abs": float(bps_error[row].max()),
        }
        global_index = int(dataset.indices[position])
        sequence = int(dataset.sequence_ids[global_index])
        goal_frame = int(dataset.language["end_range"][global_index]) - 4
        goal_frame = min(
            max(goal_frame, int(dataset.seq_starts[sequence])), int(dataset.seq_ends[sequence]) - 1,
        )
        raw_object_goal = torch.from_numpy(
            np.array(dataset.object_trans[goal_frame], dtype=np.float32, copy=True)
        ).to(device)
        row_errors["metric_object_target_max_abs"] = float(
            (object_global[row] - raw_object_goal).abs().max()
        )
        decoded = dataset.codec.decode(item["x"].to(device), frame)
        encoded, _ = dataset.codec.encode(
            decoded["joints"], decoded["human_rotation"], frame=frame,
            global_object_translation=decoded["object_translation"],
            global_object_rotation=decoded["object_rotation"], contact=decoded["contact"],
        )
        row_errors["history_max_abs"] = float(
            (encoded[:REPRESENTATION.history_frames] - item["x"][:REPRESENTATION.history_frames].to(device)).abs().max()
        )
        for error_name, error in row_errors.items():
            maxima[error_name] = max(maxima[error_name], error)
        failed_checks = [name for name, error in row_errors.items() if error > tolerances[name]]
        if failed_checks:
            failures.append({
                "sequence": names[row],
                "dataset_position": int(position),
                "global_window_index": global_index,
                "failed_checks": failed_checks,
                "max_abs_errors": row_errors,
                "bps_rms_error": float(bps_error[row].square().mean().sqrt()),
            })
    checks = {name: math.isfinite(maxima[name]) and maxima[name] <= limit for name, limit in tolerances.items()}
    return {
        "positions": len(positions),
        "future_gt_used_for_condition": False,
        "max_abs_errors": maxima,
        "tolerances": tolerances,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": all(checks.values()),
    }


def _append(target: Dict[str, List[float]], values: Mapping[str, torch.Tensor]) -> None:
    for name, value in values.items():
        target[name].extend(value.detach().cpu().double().tolist())


@torch.no_grad()
def teacher_x0_audit(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for timestep in D0_TIMESTEPS:
        matched_values = {field.name: [] for field in REPRESENTATION.fields}
        terminal_values = {field.name: [] for field in REPRESENTATION.fields}
        nonterminal_values = {field.name: [] for field in REPRESENTATION.fields}
        variant_values = {
            variant: {field.name: [] for field in REPRESENTATION.fields}
            for variant in CONDITION_VARIANTS[1:]
        }
        response_values = {
            variant: {field.name: [] for field in REPRESENTATION.fields}
            for variant in CONDITION_VARIANTS[1:]
        }
        terminal_count = 0
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset:offset + batch_size]
            batch = stack_items(dataset, batch_positions, device)
            terminal = batch["progress"][:, 1] == batch["progress"][:, 2]
            terminal_count += int(terminal.sum())
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_seed(f"D2P:teacher:{timestep}:{offset}"))
            noise = torch.randn(batch["x"].shape, device=device, generator=generator)
            times = torch.full((len(batch_positions),), timestep, device=device, dtype=torch.long)
            noisy = diffusion.q_sample(batch["x"], times, noise)
            permutation = deterministic_derangement(len(batch_positions), device=device)
            repeats = len(CONDITION_VARIANTS)
            texts = batch["text_embedding"].repeat(repeats, 1)
            bps = batch["object_bps"].repeat(repeats, 1, 1)
            goals = batch["goals"].repeat(repeats, 1)
            progress = normalize_progress(batch["progress"]).repeat(repeats, 1)
            expanded_noisy = noisy.repeat(repeats, 1, 1)
            expanded_times = times.repeat(repeats)
            width = len(batch_positions)
            for variant_index, variant in enumerate(CONDITION_VARIANTS[1:], start=1):
                selected = slice(variant_index * width, (variant_index + 1) * width)
                if variant == "text_permuted":
                    texts[selected] = batch["text_embedding"][permutation]
                elif variant == "bps_permuted":
                    bps[selected] = batch["object_bps"][permutation]
                elif variant == "pelvis_permuted":
                    goals[selected, :3] = batch["goals"][permutation, :3]
                elif variant == "object_goal_permuted":
                    goals[selected, 6:9] = batch["goals"][permutation, 6:9]
            predictions = model(expanded_noisy, expanded_times, texts, bps, goals, progress)
            matched = predictions[:width]
            matched_error = field_error_per_sample(matched, batch["x"])
            _append(matched_values, matched_error)
            if bool(terminal.any()):
                _append(terminal_values, {name: value[terminal] for name, value in matched_error.items()})
            if bool((~terminal).any()):
                _append(nonterminal_values, {name: value[~terminal] for name, value in matched_error.items()})
            for variant_index, variant in enumerate(CONDITION_VARIANTS[1:], start=1):
                prediction = predictions[variant_index * width:(variant_index + 1) * width]
                _append(variant_values[variant], field_error_per_sample(prediction, batch["x"]))
                _append(response_values[variant], {
                    field.name: (
                        prediction[:, REPRESENTATION.history_frames:, field.slice]
                        - matched[:, REPRESENTATION.history_frames:, field.slice]
                    ).square().flatten(1).mean(dim=1).sqrt()
                    for field in REPRESENTATION.fields
                })
        matched_arrays = {name: np.asarray(values, dtype=np.float64) for name, values in matched_values.items()}
        sensitivity = {}
        for variant in CONDITION_VARIANTS[1:]:
            variant_arrays = {
                name: np.asarray(values, dtype=np.float64) for name, values in variant_values[variant].items()
            }
            primary = PRIMARY_FIELD[variant]
            bootstrap = paired_bootstrap(matched_arrays[primary], variant_arrays[primary])
            sensitivity[variant] = {
                "primary_field": primary,
                "fieldwise_permuted_minus_matched_mse": {
                    name: float(variant_arrays[name].mean() - matched_arrays[name].mean())
                    for name in matched_arrays
                },
                "fieldwise_prediction_response_rms": {
                    name: float(np.mean(response_values[variant][name])) for name in matched_arrays
                },
                "bootstrap": bootstrap,
            }
        mean_or_none = lambda values: {  # noqa: E731
            name: (float(np.mean(field_values)) if field_values else None)
            for name, field_values in values.items()
        }
        output[str(timestep)] = {
            "matched_fieldwise_mse": mean_or_none(matched_values),
            "terminal_matched_fieldwise_mse": mean_or_none(terminal_values),
            "nonterminal_matched_fieldwise_mse": mean_or_none(nonterminal_values),
            "terminal_windows": terminal_count,
            "nonterminal_windows": len(positions) - terminal_count,
            "sensitivity": sensitivity,
        }
    return output


@torch.no_grad()
def reverse_trace(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
) -> Dict[str, object]:
    items = [dataset[position] for position in positions]
    batch = stack_items(dataset, positions, device)
    frames = stack_frames(items, device)
    fixed = batch["x"][:, :REPRESENTATION.history_frames]
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed("D2P:reverse-trace:paired"))
    current = torch.randn(batch["x"].shape, device=device, generator=generator)
    current[:, :REPRESENTATION.history_frames] = fixed
    trace = {}
    for step in reversed(range(diffusion.timesteps)):
        timesteps = torch.full((len(positions),), step, device=device, dtype=torch.long)
        clean = model(
            current, timesteps, batch["text_embedding"], batch["object_bps"], batch["goals"],
            normalize_progress(batch["progress"]),
        )
        clean[:, :REPRESENTATION.history_frames] = fixed
        if step in TRACE_STEPS:
            errors = field_error_per_sample(clean, batch["x"])
            trace[str(step)] = {
                "clean_x0_fieldwise_mse": {name: float(value.mean()) for name, value in errors.items()},
                "current_range": field_range(current),
                "clean_x0_range": field_range(clean),
                "finite": bool(torch.isfinite(current).all() and torch.isfinite(clean).all()),
            }
        mean = (
            _extract(diffusion.posterior_mean_coef1, timesteps, current.shape) * clean
            + _extract(diffusion.posterior_mean_coef2, timesteps, current.shape) * current
        )
        if step:
            noise = torch.randn(current.shape, device=device, generator=generator)
            current = mean + (
                0.5 * _extract(diffusion.posterior_log_variance, timesteps, current.shape)
            ).exp() * noise
        else:
            current = mean
        current[:, :REPRESENTATION.history_frames] = fixed
    current[..., 219:228] = project_to_so3(
        current[..., 219:228].reshape(len(positions), REPRESENTATION.window_frames, 3, 3)
    ).reshape(len(positions), REPRESENTATION.window_frames, 9)
    prediction = dataset.codec.decode(current, frames)
    target = dataset.codec.decode(batch["x"], frames)
    pelvis_global, object_global = global_goals(dataset, items, frames, device)
    object_error = torch.linalg.vector_norm(
        prediction["object_translation"][:, -1] - object_global, dim=-1,
    ) * 100.0
    pelvis_error = torch.linalg.vector_norm(
        prediction["joints"][:, -1, 0][:, (0, 2)] - pelvis_global[:, (0, 2)], dim=-1,
    ) * 100.0
    relative_prediction = prediction["joints"][:, REPRESENTATION.history_frames:] - prediction["joints"][:, REPRESENTATION.history_frames:, :1]
    relative_target = target["joints"][:, REPRESENTATION.history_frames:] - target["joints"][:, REPRESENTATION.history_frames:, :1]
    mpjpe = torch.linalg.vector_norm(relative_prediction - relative_target, dim=-1).mean() * 100.0
    final_error = field_error_per_sample(current, batch["x"])
    return {
        "trace": trace,
        "final": {
            "object_goal_error_cm": float(object_error.mean()),
            "pelvis_goal_error_cm": float(pelvis_error.mean()),
            "mpjpe_cm": float(mpjpe),
            "fieldwise_mse": {name: float(value.mean()) for name, value in final_error.items()},
            "output_range": field_range(current),
            "history_max_abs_error": float((current[:, :REPRESENTATION.history_frames] - fixed).abs().max()),
            "finite": bool(torch.isfinite(current).all()),
            "d2_thresholds": dict(D2_THRESHOLDS),
            "d2_threshold_checks": {
                name: float(value) <= D2_THRESHOLDS[name]
                for name, value in (
                    ("object_goal_error_cm", object_error.mean()),
                    ("pelvis_goal_error_cm", pelvis_error.mean()),
                    ("mpjpe_cm", mpjpe),
                )
            },
        },
    }


def classify_mechanism(contract: Mapping[str, object], candidates: Mapping[str, object]) -> Dict[str, object]:
    if not bool(contract.get("passed")):
        return {"category": "coordinate-contract-defect", "candidate_evidence": {}}
    evidence = {}
    for name, result in candidates.items():
        high = result["teacher_x0"]["499"]
        ratios = {
            field: float(high["matched_fieldwise_mse"][field]) / D0_T499[field]
            for field in D0_T499
        }
        weak = {
            condition: not bool(high["sensitivity"][condition]["bootstrap"]["matched_significantly_better"])
            for condition in ("text_permuted", "bps_permuted")
        }
        trace_checks = result["reverse_trace"]["final"]["d2_threshold_checks"]
        evidence[name] = {
            "d0_t499_ratios": ratios,
            "text_bps_not_significant": weak,
            "teacher_within_1.10": all(value <= 1.10 for value in ratios.values()),
            "reverse_trace_all_d2_checks_passed": all(trace_checks.values()),
        }
    if all(
        all(value > 1.10 for value in item["d0_t499_ratios"].values())
        and all(item["text_bps_not_significant"].values())
        for item in evidence.values()
    ):
        category = "high-noise-condition-underfit"
    elif all(item["teacher_within_1.10"] for item in evidence.values()) and any(
        not item["reverse_trace_all_d2_checks_passed"] for item in evidence.values()
    ):
        category = "reverse-process-exposure-gap"
    else:
        category = "mixed-mechanism"
    return {"category": category, "candidate_evidence": evidence}


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-P",
        "mode": "internal-mechanism-diagnostic-only",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "partition": "internal_validation",
        "checkpoints": {
            "R-1024": {"path": str(Path(args.checkpoint_r1024).resolve()), "sha256": args.sha256_r1024, "weights": "online"},
            "R-3072": {"path": str(Path(args.checkpoint_r3072).resolve()), "sha256": args.sha256_r3072, "weights": "online"},
        },
        "teacher": {
            "windows": 512,
            "timesteps": list(D0_TIMESTEPS),
            "condition_variants": list(CONDITION_VARIANTS),
            "bootstrap_replicates": 10000,
            "batch_size": args.teacher_batch_size,
        },
        "reverse_trace": {"sequences": 32, "windows_per_sequence": 1, "trace_steps": list(TRACE_STEPS)},
        "official_test_used": False,
        "chois_used": False,
        "training_updates": 0,
        "checkpoint_selection": False,
        "device": args.device,
        "output": str(Path(args.output).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-r1024", required=True)
    parser.add_argument("--sha256-r1024", default=EXPECTED_CHECKPOINTS["R-1024"])
    parser.add_argument("--checkpoint-r3072", required=True)
    parser.add_argument("--sha256-r3072", default=EXPECTED_CHECKPOINTS["R-3072"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-P resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-P requires INFBAGEL_WORKER_EXPERT=hoi")
    checkpoint_paths = {
        "R-1024": Path(args.checkpoint_r1024).resolve(),
        "R-3072": Path(args.checkpoint_r3072).resolve(),
    }
    requested_hashes = {"R-1024": args.sha256_r1024, "R-3072": args.sha256_r3072}
    for name, path in checkpoint_paths.items():
        actual = sha256_file(path)
        if requested_hashes[name] != EXPECTED_CHECKPOINTS[name] or actual != requested_hashes[name]:
            raise ValueError(f"{name} checkpoint hash mismatch: {actual}")
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    triples = select_internal_triples(dataset, 128)
    teacher_positions = select_teacher_windows(dataset, 512)
    trace_positions = [triple[0] for triple in triples[:32]]
    cpu = torch.device("cpu")
    contract = contract_replay(dataset, triples, cpu)
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-P",
        "seed": 42,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "selection": {
            "partition": "internal_validation",
            "official_test_sequence_count": 0,
            "chois_sequence_count": 0,
            "teacher_windows": len(teacher_positions),
            "teacher_window_indices_sha256": selection_sha256(
                int(dataset.indices[position]) for position in teacher_positions
            ),
            "trace_sequences": len(trace_positions),
            "trace_sequence_selection_sha256": selection_sha256(
                str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
                for position in trace_positions
            ),
        },
        "contract_replay": contract,
        "candidates": {},
        "training_updates": 0,
        "checkpoint_selection": False,
    }
    if not contract["passed"]:
        output["classification"] = classify_mechanism(contract, {})
        output["runtime_seconds"] = time.time() - started
        exclusive_json(Path(args.output).resolve(), output)
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-P P1/P2 is a worker CUDA workload")
    diffusion = GaussianDiffusion(500).to(device)
    for name in ("R-1024", "R-3072"):
        model, metadata = load_trained_hoi_prior(
            str(checkpoint_paths[name]), device, weight_variant="online",
        )
        output["candidates"][name] = {
            "checkpoint": metadata,
            "teacher_x0": teacher_x0_audit(
                model, diffusion, dataset, teacher_positions, device, args.teacher_batch_size,
            ),
            "reverse_trace": reverse_trace(model, diffusion, dataset, trace_positions, device),
        }
        del model
        torch.cuda.empty_cache()
    output["classification"] = classify_mechanism(contract, output["candidates"])
    output["runtime_seconds"] = time.time() - started
    output["gpu"] = {"device": str(device), "name": torch.cuda.get_device_name(device)}
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
