"""Paired reverse-state exposure utilities for the Phase 1B D2-H0 gate."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..core.representation import REPRESENTATION


TARGET_TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100, 250, 498)
LOW_TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100)
CONDITION_VARIANTS: Tuple[str, ...] = (
    "matched",
    "text_permuted",
    "bps_permuted",
    "pelvis_permuted",
    "object_goal_permuted",
)
CHECKPOINTS: Tuple[str, ...] = ("R-1024", "R-3072")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 42
HISTORY_MAX_ABS = 1e-5
FORMULA_REPLAY_MAX_ABS = 1e-5


def deterministic_condition_variants(
    batch: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
    """Build four independent one-condition permutations plus matched inputs."""
    required = ("text_embedding", "object_bps", "goals", "progress")
    if any(name not in batch for name in required):
        raise ValueError("condition batch is missing a required tensor")
    count = batch["text_embedding"].shape[0]
    if count < 2 or any(batch[name].shape[0] != count for name in required):
        raise ValueError("condition variants require matching batches with at least two rows")
    permutation = torch.roll(torch.arange(count, device=batch["text_embedding"].device), shifts=1)
    result: Dict[str, Dict[str, torch.Tensor]] = {}
    for variant in CONDITION_VARIANTS:
        text = batch["text_embedding"]
        bps = batch["object_bps"]
        goals = batch["goals"]
        if variant == "text_permuted":
            text = text[permutation]
        elif variant == "bps_permuted":
            bps = bps[permutation]
        elif variant == "pelvis_permuted":
            goals = goals.clone()
            goals[:, :3] = batch["goals"][permutation, :3]
        elif variant == "object_goal_permuted":
            goals = goals.clone()
            goals[:, 6:9] = batch["goals"][permutation, 6:9]
        result[variant] = {
            "text_embedding": text,
            "object_bps": bps,
            "goals": goals,
            "progress": batch["progress"],
        }
    return result, permutation


def fieldwise_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return every registered non-history field MSE for every sample."""
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(f"expected matching [B,16,232], got {prediction.shape}/{target.shape}")
    return {
        field.name: (
            prediction[:, REPRESENTATION.history_frames:, field.slice]
            - target[:, REPRESENTATION.history_frames:, field.slice]
        ).square().flatten(1).mean(dim=1)
        for field in REPRESENTATION.fields
    }


def paired_bootstrap_model_minus_oracle(
    oracle: Sequence[float],
    model: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, object]:
    """Bootstrap the paired model-parent minus oracle-parent mean gap."""
    oracle_array = np.asarray(oracle, dtype=np.float64)
    model_array = np.asarray(model, dtype=np.float64)
    if oracle_array.shape != model_array.shape or oracle_array.ndim != 1 or not len(oracle_array):
        raise ValueError("paired bootstrap expects non-empty equal one-dimensional arrays")
    difference = model_array - oracle_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    means = difference[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    oracle_mean = float(oracle_array.mean())
    model_mean = float(model_array.mean())
    return {
        "oracle_parent_mean": oracle_mean,
        "model_parent_mean": model_mean,
        "model_over_oracle_mean_ratio": model_mean / max(oracle_mean, np.finfo(np.float64).tiny),
        "paired_mean_model_minus_oracle": float(difference.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "positive_lower_bound": bool(lower > 0.0),
        "per_sample": {
            "oracle_parent": oracle_array.tolist(),
            "model_parent": model_array.tolist(),
            "model_minus_oracle": difference.tolist(),
        },
    }


def compact_metric(metric: Mapping[str, object]) -> Dict[str, object]:
    """Strip per-sample arrays while retaining every gate statistic."""
    return {key: value for key, value in metric.items() if key != "per_sample"}


def _geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(LOW_TIMESTEPS),) or not np.isfinite(array).all() or bool((array <= 0).any()):
        return float("nan")
    return float(np.exp(np.log(array).mean()))


def mechanism_gate(candidates: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Apply the preregistered conjunction without selecting favorable subsets."""
    checkpoint_results: Dict[str, object] = {}
    for checkpoint in CHECKPOINTS:
        if checkpoint not in candidates:
            checkpoint_results[checkpoint] = {
                "passed": False,
                "missing": True,
                "failed_checks": ["checkpoint_present"],
            }
            continue
        candidate = candidates[checkpoint]
        timestep_records = candidate.get("timesteps", {})
        simultaneous_positive = []
        joint_ratios = []
        object_ratios = []
        for timestep in LOW_TIMESTEPS:
            matched = timestep_records.get(str(timestep), {}).get("matched", {})
            fields = matched.get("field_comparison", {})
            joint = fields.get("joint_positions", {})
            object_translation = fields.get("object_translation", {})
            joint_positive = bool(joint.get("positive_lower_bound", False))
            object_positive = bool(object_translation.get("positive_lower_bound", False))
            simultaneous_positive.append({
                "target_timestep": timestep,
                "joint_position_positive_lower_bound": joint_positive,
                "object_translation_positive_lower_bound": object_positive,
                "simultaneous": joint_positive and object_positive,
            })
            joint_ratios.append(float(joint.get("model_over_oracle_mean_ratio", float("nan"))))
            object_ratios.append(float(
                object_translation.get("model_over_oracle_mean_ratio", float("nan"))
            ))
        positive_count = sum(int(value["simultaneous"]) for value in simultaneous_positive)
        joint_geomean = _geometric_mean(joint_ratios)
        object_geomean = _geometric_mean(object_ratios)
        base_checks = {
            "finite": bool(candidate.get("finite", False)),
            "history_max_abs": (
                math.isfinite(float(candidate.get("history_max_abs", float("nan"))))
                and float(candidate["history_max_abs"]) <= HISTORY_MAX_ABS
            ),
            "posterior_formula_replay_max_abs": (
                math.isfinite(float(candidate.get("posterior_formula_replay_max_abs", float("nan"))))
                and float(candidate["posterior_formula_replay_max_abs"]) <= FORMULA_REPLAY_MAX_ABS
            ),
            "simultaneous_positive_low_timesteps": positive_count >= 4,
            "joint_position_low_timestep_geomean_ratio": (
                math.isfinite(joint_geomean) and joint_geomean >= 1.5
            ),
            "object_translation_low_timestep_geomean_ratio": (
                math.isfinite(object_geomean) and object_geomean >= 2.0
            ),
        }
        checkpoint_results[checkpoint] = {
            "passed": all(base_checks.values()),
            "checks": base_checks,
            "failed_checks": [name for name, passed in base_checks.items() if not passed],
            "simultaneous_positive_low_timestep_count": positive_count,
            "simultaneous_positive_by_timestep": simultaneous_positive,
            "low_timestep_mean_ratios": {
                "joint_positions": joint_ratios,
                "object_translation": object_ratios,
            },
            "low_timestep_geometric_mean_ratio": {
                "joint_positions": joint_geomean,
                "object_translation": object_geomean,
            },
        }
    passed = all(bool(checkpoint_results[name].get("passed")) for name in CHECKPOINTS)
    return {
        "passed": passed,
        "classification": (
            "reverse-state-exposure-positive-stop"
            if passed else "reverse-state-exposure-negative-stop"
        ),
        "required_checkpoints": list(CHECKPOINTS),
        "checkpoint_results": checkpoint_results,
        "d2h1_condition_prerequisite_met": passed,
        "d2h1_started": False,
    }
