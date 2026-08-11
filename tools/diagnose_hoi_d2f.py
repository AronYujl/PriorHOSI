#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-F paired reverse-manifold diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.hoi.data import PriorWindowDataset  # noqa: E402
from priors.hoi.diffusion import (  # noqa: E402
    GaussianDiffusion,
    _extract,
    normalize_progress,
    prepare_clean_x0,
)
from priors.hoi.models import load_trained_hoi_prior  # noqa: E402
from priors.hoi.remediation import select_internal_triples, selection_sha256  # noqa: E402
from priors.core.representation import REPRESENTATION  # noqa: E402
from priors.core.window_codec import project_to_so3  # noqa: E402
from tools.diagnose_hoi_d2p import D2_THRESHOLDS, TRACE_STEPS, field_error_per_sample, field_range  # noqa: E402
from tools.diagnose_hoi_remediation import stack_items, stable_seed  # noqa: E402
from tools.evaluate_hoi_remediation import global_goals, stack_frames  # noqa: E402


RUN_ID = "p1-hoi-d2f-so3-reverse-s42-20260715"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_D2P5_AGGREGATE_SHA256 = "c060736c188ae2a29335337dc49d50a662eb0e4bb2fe44b390a4a14d44b68429"
EXPECTED_TRACE_SELECTION_SHA256 = "7f16d1b8f4f3843639d10d0ecd367d1e2073b8b55bb03f4fef9895c960b85663"
EXPECTED_CHECKPOINTS = {
    "R-1024": "d7931a3221c11903a8f9856355a16a493107ed78ad7947120906eece2b22ec23",
    "R-3072": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
}
MANIFOLD_MAX = 1e-5
HISTORY_MAX = 1e-5
MECHANISM_RATIO_MAX = {
    "object_goal_error_cm": 0.50,
    "mpjpe_cm": 1.02,
    "pelvis_goal_error_cm": 1.05,
}
VARIANTS = ("control", "object_so3_x0")


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


def object_rotation_manifold_error(value: torch.Tensor) -> Dict[str, float]:
    """Measure SO(3) residuals on predicted frames only."""
    matrices = value[:, REPRESENTATION.history_frames:, 219:228].reshape(
        value.shape[0], REPRESENTATION.window_frames - REPRESENTATION.history_frames, 3, 3,
    )
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    orthogonality = torch.linalg.matrix_norm(
        matrices.transpose(-1, -2) @ matrices - identity, ord="fro", dim=(-2, -1),
    )
    determinant = (torch.linalg.det(matrices) - 1.0).abs()
    return {
        "orthogonality_frobenius_max": float(orthogonality.max()),
        "orthogonality_frobenius_mean": float(orthogonality.mean()),
        "determinant_abs_error_max": float(determinant.max()),
        "determinant_abs_error_mean": float(determinant.mean()),
    }


def _merge_maximum(target: Dict[str, float], value: Mapping[str, float]) -> None:
    for name, metric in value.items():
        if name.endswith("_max"):
            target[name] = max(target.get(name, 0.0), float(metric))


@torch.no_grad()
def reverse_trace(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    *,
    variant: str,
) -> Dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown D2-F variant: {variant}")
    items = [dataset[position] for position in positions]
    batch = stack_items(dataset, positions, device)
    frames = stack_frames(items, device)
    fixed = batch["x"][:, :REPRESENTATION.history_frames]
    generator = torch.Generator(device=device)
    paired_seed = stable_seed("D2F:reverse-manifold:paired")
    generator.manual_seed(paired_seed)
    current = torch.randn(batch["x"].shape, device=device, generator=generator)
    current[:, :REPRESENTATION.history_frames] = fixed
    trace: Dict[str, object] = {}
    raw_maximum: Dict[str, float] = {}
    applied_maximum: Dict[str, float] = {}
    finite = True
    for step in reversed(range(diffusion.timesteps)):
        timesteps = torch.full((len(positions),), step, device=device, dtype=torch.long)
        raw_clean = model(
            current,
            timesteps,
            batch["text_embedding"],
            batch["object_bps"],
            batch["goals"],
            normalize_progress(batch["progress"]),
        )
        raw_clean = prepare_clean_x0(raw_clean, fixed, object_so3_x0=False)
        clean = prepare_clean_x0(
            raw_clean, fixed, object_so3_x0=(variant == "object_so3_x0"),
        )
        raw_manifold = object_rotation_manifold_error(raw_clean)
        applied_manifold = object_rotation_manifold_error(clean)
        _merge_maximum(raw_maximum, raw_manifold)
        _merge_maximum(applied_maximum, applied_manifold)
        finite = finite and bool(
            torch.isfinite(current).all()
            and torch.isfinite(raw_clean).all()
            and torch.isfinite(clean).all()
        )
        if step in TRACE_STEPS:
            raw_error = field_error_per_sample(raw_clean, batch["x"])
            applied_error = field_error_per_sample(clean, batch["x"])
            trace[str(step)] = {
                "raw_clean_x0_fieldwise_mse": {
                    name: float(value.mean()) for name, value in raw_error.items()
                },
                "applied_clean_x0_fieldwise_mse": {
                    name: float(value.mean()) for name, value in applied_error.items()
                },
                "current_range": field_range(current),
                "raw_clean_x0_range": field_range(raw_clean),
                "applied_clean_x0_range": field_range(clean),
                "raw_object_rotation_manifold": raw_manifold,
                "applied_object_rotation_manifold": applied_manifold,
                "finite": finite,
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

    decoded_state = current.clone()
    decoded_rotation = project_to_so3(
        decoded_state[..., 219:228].reshape(
            len(positions), REPRESENTATION.window_frames, 3, 3,
        )
    )
    decoded_state[..., 219:228] = decoded_rotation.reshape(
        len(positions), REPRESENTATION.window_frames, 9,
    )
    prediction = dataset.codec.decode(decoded_state, frames)
    target = dataset.codec.decode(batch["x"], frames)
    pelvis_global, object_global = global_goals(dataset, items, frames, device)
    object_error = torch.linalg.vector_norm(
        prediction["object_translation"][:, -1] - object_global, dim=-1,
    ) * 100.0
    pelvis_error = torch.linalg.vector_norm(
        prediction["joints"][:, -1, 0][:, (0, 2)] - pelvis_global[:, (0, 2)], dim=-1,
    ) * 100.0
    relative_prediction = (
        prediction["joints"][:, REPRESENTATION.history_frames:]
        - prediction["joints"][:, REPRESENTATION.history_frames:, :1]
    )
    relative_target = (
        target["joints"][:, REPRESENTATION.history_frames:]
        - target["joints"][:, REPRESENTATION.history_frames:, :1]
    )
    mpjpe = torch.linalg.vector_norm(relative_prediction - relative_target, dim=-1).mean() * 100.0
    final_error = field_error_per_sample(decoded_state, batch["x"])
    metrics = {
        "object_goal_error_cm": float(object_error.mean()),
        "pelvis_goal_error_cm": float(pelvis_error.mean()),
        "mpjpe_cm": float(mpjpe),
    }
    history_error = float(
        (decoded_state[:, :REPRESENTATION.history_frames] - fixed).abs().max()
    )
    manifold_checks = {
        "orthogonality_frobenius": (
            applied_maximum["orthogonality_frobenius_max"] <= MANIFOLD_MAX
        ),
        "determinant_abs_error": (
            applied_maximum["determinant_abs_error_max"] <= MANIFOLD_MAX
        ),
    }
    return {
        "variant": variant,
        "paired_seed": paired_seed,
        "trace": trace,
        "all_step_raw_object_rotation_manifold_max": raw_maximum,
        "all_step_applied_object_rotation_manifold_max": applied_maximum,
        "final": {
            **metrics,
            "fieldwise_mse": {name: float(value.mean()) for name, value in final_error.items()},
            "output_range": field_range(decoded_state),
            "history_max_abs_error": history_error,
            "finite": bool(finite and torch.isfinite(decoded_state).all()),
            "d2_thresholds": dict(D2_THRESHOLDS),
            "d2_threshold_checks": {
                name: value <= D2_THRESHOLDS[name] for name, value in metrics.items()
            },
            "manifold_threshold": MANIFOLD_MAX,
            "manifold_checks": manifold_checks,
            "history_check": history_error <= HISTORY_MAX,
        },
    }


def classify(results: Mapping[str, object]) -> Dict[str, object]:
    candidate_checks: Dict[str, object] = {}
    for name, candidate in results.items():
        control = candidate["control"]["final"]
        projected = candidate["object_so3_x0"]["final"]
        absolute_pass = bool(
            projected["finite"]
            and projected["history_check"]
            and all(projected["manifold_checks"].values())
            and all(projected["d2_threshold_checks"].values())
        )
        ratios = {
            metric: float(projected[metric]) / max(float(control[metric]), 1e-12)
            for metric in MECHANISM_RATIO_MAX
        }
        candidate_checks[name] = {
            "absolute_pass": absolute_pass,
            "so3_over_control_ratios": ratios,
            "paired_seed_equal": (
                candidate["control"]["paired_seed"]
                == candidate["object_so3_x0"]["paired_seed"]
            ),
        }
    r1024 = candidate_checks["R-1024"]
    r1024_projected = results["R-1024"]["object_so3_x0"]["final"]
    mechanism_positive = bool(
        all(
            r1024["so3_over_control_ratios"][name] <= threshold
            for name, threshold in MECHANISM_RATIO_MAX.items()
        )
        and r1024["paired_seed_equal"]
        and r1024_projected["finite"]
        and r1024_projected["history_check"]
        and all(r1024_projected["manifold_checks"].values())
    )
    absolute_passes = [
        name for name, checks in candidate_checks.items() if checks["absolute_pass"]
    ]
    if absolute_passes:
        category = "absolute-single-window-gate-pass"
    elif mechanism_positive:
        category = "sampler-mechanism-positive-training-insufficient"
    else:
        category = "sampler-mechanism-negative-stop"
    return {
        "category": category,
        "candidate_checks": candidate_checks,
        "absolute_gate_passes": absolute_passes,
        "d2f1_authorized": bool(absolute_passes),
        "d2f2_authorized": bool(not absolute_passes and mechanism_positive),
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-F0",
        "mode": "paired-reverse-manifold-diagnostic-only",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "partition": "internal_validation",
        "trace_sequences": 32,
        "trace_sequence_selection_sha256": EXPECTED_TRACE_SELECTION_SHA256,
        "trace_steps": list(TRACE_STEPS),
        "paired_variants": list(VARIANTS),
        "paired_initial_and_posterior_noise": True,
        "projection_channels": [219, 228],
        "projection_before_each_posterior_mean": True,
        "other_channel_clamp": False,
        "posterior_change": False,
        "checkpoints": {
            "R-1024": {
                "path": str(Path(args.checkpoint_r1024).resolve()),
                "sha256": args.sha256_r1024,
                "weights": "online",
            },
            "R-3072": {
                "path": str(Path(args.checkpoint_r3072).resolve()),
                "sha256": args.sha256_r3072,
                "weights": "online",
            },
        },
        "absolute_gate": {
            **{f"{name}_max": value for name, value in D2_THRESHOLDS.items()},
            "history_max_abs_max": HISTORY_MAX,
            "orthogonality_frobenius_max": MANIFOLD_MAX,
            "determinant_abs_error_max": MANIFOLD_MAX,
            "finite_required": True,
        },
        "mechanism_positive_ratio_max_R-1024": dict(MECHANISM_RATIO_MAX),
        "d2p5_aggregate_sha256": EXPECTED_D2P5_AGGREGATE_SHA256,
        "checkpoint_selection": False,
        "training_updates": 0,
        "official_test_used": False,
        "chois_used": False,
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-F0 run id must be {RUN_ID}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-F0 resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-F0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("D2-F0 refuses a dirty worker checkout")
    checkpoint_paths = {
        "R-1024": Path(args.checkpoint_r1024).resolve(),
        "R-3072": Path(args.checkpoint_r3072).resolve(),
    }
    requested_hashes = {
        "R-1024": args.sha256_r1024,
        "R-3072": args.sha256_r3072,
    }
    for name, path in checkpoint_paths.items():
        actual = sha256_file(path)
        if requested_hashes[name] != EXPECTED_CHECKPOINTS[name] or actual != requested_hashes[name]:
            raise ValueError(f"{name} checkpoint hash mismatch: {actual}")
    d2p5 = REPO / "experiments/results/p1_hoi_phase1b_d2p5_mechanism_s42_20260715.json"
    if sha256_file(d2p5) != EXPECTED_D2P5_AGGREGATE_SHA256:
        raise ValueError("D2-F0 requires the hash-verified D2-P5 aggregate")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-F0 is a worker CUDA workload")
    started = time.time()
    dataset = PriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    triples = select_internal_triples(dataset, 128)
    positions = [triple[0] for triple in triples[:32]]
    selection_hash = selection_sha256(
        str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
        for position in positions
    )
    if selection_hash != EXPECTED_TRACE_SELECTION_SHA256:
        raise ValueError(f"D2-F0 trace selection mismatch: {selection_hash}")
    diffusion = GaussianDiffusion(500).to(device)
    candidates: Dict[str, object] = {}
    for name in ("R-1024", "R-3072"):
        model, metadata = load_trained_hoi_prior(
            str(checkpoint_paths[name]), device, weight_variant="online",
        )
        candidates[name] = {"checkpoint": metadata}
        for variant in VARIANTS:
            candidates[name][variant] = reverse_trace(
                model, diffusion, dataset, positions, device, variant=variant,
            )
        del model
        torch.cuda.empty_cache()
    decision = classify(candidates)
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-F0",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "trace_sequences": len(positions),
            "trace_sequence_selection_sha256": selection_hash,
            "official_test_sequence_count": 0,
            "chois_sequence_count": 0,
        },
        "candidates": candidates,
        "decision": decision,
        "checkpoint_count_loaded": len(candidates),
        "checkpoint_selection": False,
        "training_updates": 0,
        "paired_initial_and_posterior_noise": True,
        "other_channel_clamp": False,
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
        "official_test_used": False,
        "chois_used": False,
        "runtime_seconds": time.time() - started,
        "gpu": {"device": str(device), "name": torch.cuda.get_device_name(device)},
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
