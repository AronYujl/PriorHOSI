"""Locked paired fresh-optimizer smoke utilities for Phase 1B D2-M0."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .auxiliary_balancing import (
    BALANCED_WEIGHTS,
    CURRENT_WEIGHTS,
    WEIGHT_SOURCE_METRICS_SHA256,
    WEIGHT_SOURCE_RUN,
)
from .exposure import CONDITION_VARIANTS
from .remediation import selection_sha256, stable_digest
from .representation import REPRESENTATION


RUN_ID = "p1-hoi-d2m-reset-paired-s42-20260716"
SOURCE_CHECKPOINT_SHA256 = "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4"
SOURCE_RUN_ID = "p1-hoi-d2-r3072-s42-20260714"
SOURCE_OPTIMIZER_LR = 3e-5
CANDIDATES: Tuple[str, ...] = ("current", "balanced")
WEIGHTS = {"current": CURRENT_WEIGHTS, "balanced": BALANCED_WEIGHTS}
OPTIMIZER_UPDATES = 64
EFFECTIVE_BATCH_SIZE = 3072
PROCESSED_WINDOWS = OPTIMIZER_UPDATES * EFFECTIVE_BATCH_SIZE
PROCESSED_FRAMES = PROCESSED_WINDOWS * REPRESENTATION.window_frames
TEACHER_TIMESTEPS: Tuple[int, ...] = (250, 499)
TEACHER_FIRST_RANK = 1026
TEACHER_WINDOWS = 512
TEACHER_SELECTION_SHA256 = "836781bbcdc3a5960631c7af635eaca62bb53f8a67093312fa761eb140174259"
TEACHER_TERMINAL_WINDOWS = 5
NATIVE_FIRST_RANK = 128
NATIVE_SEQUENCES = 32
NATIVE_SELECTION_SHA256 = "30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
HISTORY_MAX_ABS = 1e-5


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def _ranked_windows(dataset):
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        ranked.append((
            stable_digest(f"42:hoi-remediation-window:{name}:{pi}"),
            name,
            pi,
            position,
            int(global_index),
        ))
    ranked.sort()
    return ranked


def select_teacher_holdout(dataset) -> Dict[str, object]:
    """Return the preregistered fresh D0 ranks 1026--1537."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-M teacher selection is internal-validation only")
    ranked = _ranked_windows(dataset)
    stop = TEACHER_FIRST_RANK + TEACHER_WINDOWS
    if len(ranked) < stop:
        raise ValueError(f"D2-M requires at least {stop} ranked internal windows")
    rows = ranked[TEACHER_FIRST_RANK:stop]
    positions = [row[-2] for row in rows]
    global_indices = [row[-1] for row in rows]
    terminal_windows = sum(
        int(dataset.ends[index])
        == int(dataset.seq_ends[int(dataset.sequence_ids[index])]) - 1
        for index in global_indices
    )
    result = {
        "positions": positions,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "terminal_windows": terminal_windows,
        "first_rank": TEACHER_FIRST_RANK,
        "last_rank": stop - 1,
    }
    if result["sha256"] != TEACHER_SELECTION_SHA256:
        raise ValueError(f"D2-M teacher selection mismatch: {result['sha256']}")
    if terminal_windows != TEACHER_TERMINAL_WINDOWS:
        raise ValueError(f"D2-M teacher terminal count mismatch: {terminal_windows}")
    return result


def select_native_holdout(dataset) -> Dict[str, object]:
    """Return eligible three-window sequence ranks 128--159."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-M native selection is internal-validation only")
    by_sequence = defaultdict(dict)
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        pi = int(dataset.language["pi"][global_index])
        by_sequence[sequence][pi] = position
    eligible = []
    for sequence, positions in by_sequence.items():
        if all(pi in positions for pi in (0, 42, 84)):
            name = str(dataset.scene_names[sequence])
            eligible.append((
                stable_digest("42:hoi-remediation:" + name),
                name,
                sequence,
                positions,
            ))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    stop = NATIVE_FIRST_RANK + NATIVE_SEQUENCES
    if len(eligible) < stop:
        raise ValueError(f"D2-M requires at least {stop} eligible internal sequences")
    rows = eligible[NATIVE_FIRST_RANK:stop]
    triples = [tuple(row[3][pi] for pi in (0, 42, 84)) for row in rows]
    global_indices = [
        int(dataset.indices[position]) for triple in triples for position in triple
    ]
    result = {
        "triples": triples,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "first_rank": NATIVE_FIRST_RANK,
        "last_rank": stop - 1,
        "sequences": len(triples),
    }
    if result["sha256"] != NATIVE_SELECTION_SHA256:
        raise ValueError(f"D2-M native selection mismatch: {result['sha256']}")
    return result


def _paired_arrays(
    first: Sequence[float], second: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("D2-M paired statistics require equal non-empty vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("D2-M paired statistics require finite vectors")
    return left, right


def paired_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, object]:
    """Bootstrap mean(first - second) with paired sampling units."""
    left, right = _paired_arrays(first, second)
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    means = difference[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "first_mean": float(left.mean()),
        "second_mean": float(right.mean()),
        "paired_mean_first_minus_second": float(difference.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "per_unit": {
            "first": left.tolist(),
            "second": right.tolist(),
            "first_minus_second": difference.tolist(),
        },
    }


def paired_mean_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, object]:
    """Bootstrap ratio(mean(numerator) / mean(denominator)) by paired units."""
    top, bottom = _paired_arrays(numerator, denominator)
    if bool((bottom < 0).any()) or float(bottom.mean()) <= 0.0:
        raise ValueError("D2-M paired ratios require a positive denominator mean")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(top), size=(replicates, len(top)))
    bootstrap_bottom = bottom[indices].mean(axis=1)
    ratios = top[indices].mean(axis=1) / np.maximum(
        bootstrap_bottom, np.finfo(np.float64).tiny,
    )
    lower, upper = np.quantile(ratios, (0.025, 0.975))
    estimate = float(top.mean() / bottom.mean())
    return {
        "numerator_mean": float(top.mean()),
        "denominator_mean": float(bottom.mean()),
        "mean_ratio": estimate,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "per_unit": {
            "numerator": top.tolist(),
            "denominator": bottom.tolist(),
        },
    }


def compact_statistic(value: Mapping[str, object]) -> Dict[str, object]:
    return {key: item for key, item in value.items() if key != "per_unit"}


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mechanism_gate(
    training: Mapping[str, object],
    teacher: Mapping[str, object],
    native: Mapping[str, object],
) -> Dict[str, object]:
    """Apply the complete preregistered D2-M0 conjunction."""
    training_checks = {
        "all_finite": bool(training.get("all_finite", False)),
        "source_checkpoint_hash_exact": bool(training.get("source_checkpoint_hash_exact", False)),
        "source_model_hash_exact": bool(training.get("source_model_hash_exact", False)),
        "asset_hashes_exact": bool(training.get("asset_hashes_exact", False)),
        "initial_model_hashes_equal": bool(training.get("initial_model_hashes_equal", False)),
        "old_state_load_counts_zero": bool(training.get("old_state_load_counts_zero", False)),
        "initial_optimizer_state_count_zero": all(
            int(training.get("candidates", {}).get(name, {}).get(
                "initial_optimizer_state_count", -1,
            )) == 0
            for name in CANDIDATES
        ),
        "terminal_optimizer_contract": all(
            int(training.get("candidates", {}).get(name, {}).get(
                "terminal_optimizer_state_count", -1,
            )) == 119
            and int(training.get("candidates", {}).get(name, {}).get(
                "terminal_optimizer_step_min", -1,
            )) == OPTIMIZER_UPDATES
            and int(training.get("candidates", {}).get(name, {}).get(
                "terminal_optimizer_step_max", -1,
            )) == OPTIMIZER_UPDATES
            and int(training.get("candidates", {}).get(name, {}).get(
                "optimizer_updates", -1,
            )) == OPTIMIZER_UPDATES
            for name in CANDIDATES
        ),
        "paired_training_rng_audit": bool(training.get("paired_training_rng_audit", False)),
    }
    teacher_checks: Dict[str, object] = {}
    joint_source_ratios = []
    object_source_ratios = []
    for timestep in TEACHER_TIMESTEPS:
        record = teacher.get("timesteps", {}).get(str(timestep), {})
        current_minus_balanced = record.get("current_minus_balanced", {})
        balanced_over_current = record.get("balanced_over_current", {})
        balanced_over_source = record.get("balanced_over_source", {})
        joint_difference = current_minus_balanced.get("joint_positions", {})
        object_control_ratio = balanced_over_current.get("object_translation", {})
        joint_source_ratio = balanced_over_source.get("joint_positions", {})
        object_source_ratio = balanced_over_source.get("object_translation", {})
        joint_ratio_value = float(joint_source_ratio.get("mean_ratio", float("nan")))
        object_ratio_value = float(object_source_ratio.get("mean_ratio", float("nan")))
        joint_source_ratios.append(joint_ratio_value)
        object_source_ratios.append(object_ratio_value)
        checks = {
            "finite": bool(record.get("finite", False)),
            "history_max_abs": (
                _finite_number(record.get("history_max_abs"))
                and float(record["history_max_abs"]) <= HISTORY_MAX_ABS
            ),
            "current_minus_balanced_joint_position": (
                _finite_number(joint_difference.get("bootstrap_95_ci", [float("nan")])[0])
                and float(joint_difference["bootstrap_95_ci"][0]) > 0.0
            ),
            "balanced_over_current_object_translation": (
                _finite_number(object_control_ratio.get("bootstrap_95_ci", [0.0, float("nan")])[1])
                and float(object_control_ratio["bootstrap_95_ci"][1]) <= 1.05
            ),
            "balanced_over_source_joint_position_each": (
                math.isfinite(joint_ratio_value) and joint_ratio_value <= 1.02
            ),
        }
        teacher_checks[str(timestep)] = {
            "passed": all(checks.values()),
            "checks": checks,
        }
    joint_geomean = (
        float(np.exp(np.log(joint_source_ratios).mean()))
        if all(math.isfinite(value) and value > 0.0 for value in joint_source_ratios)
        else float("nan")
    )
    object_geomean = (
        float(np.exp(np.log(object_source_ratios).mean()))
        if all(math.isfinite(value) and value > 0.0 for value in object_source_ratios)
        else float("nan")
    )
    teacher_aggregate_checks = {
        "balanced_over_source_joint_position_geomean": (
            math.isfinite(joint_geomean) and joint_geomean <= 0.98
        ),
        "balanced_over_source_object_translation_geomean": (
            math.isfinite(object_geomean) and object_geomean <= 1.05
        ),
        "all_fields_conditions_and_physical_reported": bool(
            teacher.get("all_fields_conditions_and_physical_reported", False)
        ),
    }
    native_current_minus_balanced = native.get("current_minus_balanced", {})
    native_balanced_over_current = native.get("balanced_over_current", {})
    native_checks = {
        "finite": bool(native.get("finite", False)),
        "current_minus_balanced_mpjpe": (
            _finite_number(native_current_minus_balanced.get(
                "mpjpe_cm", {},
            ).get("bootstrap_95_ci", [float("nan")])[0])
            and float(native_current_minus_balanced["mpjpe_cm"]["bootstrap_95_ci"][0]) > 0.0
        ),
        "current_minus_balanced_object_goal": (
            _finite_number(native_current_minus_balanced.get(
                "object_goal_error_cm", {},
            ).get("bootstrap_95_ci", [float("nan")])[0])
            and float(
                native_current_minus_balanced["object_goal_error_cm"]["bootstrap_95_ci"][0]
            ) > 0.0
        ),
        "balanced_over_current_pelvis_goal": (
            _finite_number(native_balanced_over_current.get(
                "pelvis_goal_error_cm", {},
            ).get("bootstrap_95_ci", [0.0, float("nan")])[1])
            and float(
                native_balanced_over_current["pelvis_goal_error_cm"]["bootstrap_95_ci"][1]
            ) <= 1.10
        ),
        "all_native_metrics_reported": bool(native.get("all_native_metrics_reported", False)),
    }
    passed = (
        all(training_checks.values())
        and all(record["passed"] for record in teacher_checks.values())
        and all(teacher_aggregate_checks.values())
        and all(native_checks.values())
    )
    return {
        "passed": passed,
        "classification": (
            "fresh-optimizer-balanced-smoke-positive-stop"
            if passed else "fresh-optimizer-balanced-smoke-negative-stop"
        ),
        "training_checks": training_checks,
        "teacher_checks": teacher_checks,
        "teacher_aggregate": {
            "joint_position_balanced_over_source_geometric_mean": joint_geomean,
            "object_translation_balanced_over_source_geometric_mean": object_geomean,
            "checks": teacher_aggregate_checks,
        },
        "native_checks": native_checks,
        "from_random_screen_condition_prerequisite_met": passed,
        "from_random_screen_started": False,
        "full_training_authorized": False,
        "full_training_started": False,
        "d2h1_started": False,
    }
