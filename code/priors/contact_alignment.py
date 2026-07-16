"""Locked utilities for the Phase 1B D2-O0 contact-alignment diagnostic."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .optimizer_reset import paired_difference
from .remediation import selection_sha256, stable_digest
from .window_codec import project_to_so3


RUN_ID = "p1-hoi-d2o-contact-alignment-s42-20260716"
MODELS: Tuple[str, ...] = ("source", "current", "balanced")
EXPECTED_CHECKPOINT_SHA256 = {
    "source": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
    "current": "76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f",
    "balanced": "ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8",
}
PHASE_OFFSETS: Tuple[int, ...] = (14, 56, 98)
PRIOR_ROLLOUT_OFFSETS: Tuple[int, ...] = (0, 42, 84)
SEQUENCES = 64
WINDOWS_PER_SEQUENCE = 3
SELECTION_SHA256 = "1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a"
SEMANTIC_THRESHOLDS: Tuple[float, ...] = (0.5, 0.75, 0.95)
PHYSICAL_THRESHOLDS_CM: Tuple[float, ...] = (2.0, 5.0, 7.5, 10.0)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
HISTORY_MAX_ABS = 1e-5
GT_ALIGNMENT_MIN = 0.80
HAND_INDICES: Tuple[int, int] = (24, 26)
UNITS: Tuple[str, ...] = ("left_hand", "right_hand", "union")
DECOMPOSITION_PATHS: Tuple[str, ...] = (
    "generated_human_generated_object",
    "gt_human_generated_object",
    "generated_human_gt_object",
    "gt_human_gt_object",
)


def _threshold_key(value: float) -> str:
    return f"{value:g}"


def select_contact_holdout(dataset) -> Dict[str, object]:
    """Return the locked 64-sequence phase-offset rollout selection."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-O selection is internal-validation only")
    by_sequence = defaultdict(dict)
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        pi = int(dataset.language["pi"][global_index])
        by_sequence[sequence][pi] = position
    eligible = []
    suffix = ",".join(str(value) for value in PHASE_OFFSETS)
    for sequence, positions in by_sequence.items():
        if all(pi in positions for pi in PHASE_OFFSETS):
            name = str(dataset.scene_names[sequence])
            eligible.append((
                stable_digest(f"42:d2o-contact-alignment:{name}:{suffix}"),
                name,
                sequence,
                positions,
            ))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(eligible) < SEQUENCES:
        raise ValueError(f"D2-O requires {SEQUENCES} eligible sequences")
    rows = eligible[:SEQUENCES]
    triples = [
        tuple(row[3][pi] for pi in PHASE_OFFSETS)
        for row in rows
    ]
    global_indices = [
        int(dataset.indices[position])
        for triple in triples
        for position in triple
    ]
    result = {
        "triples": triples,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "phase_offsets": list(PHASE_OFFSETS),
        "prior_rollout_offsets": list(PRIOR_ROLLOUT_OFFSETS),
        "sequences": len(triples),
        "windows": len(global_indices),
        "eligible_sequences": len(eligible),
        "sequence_names": [row[1] for row in rows],
    }
    if result["sha256"] != SELECTION_SHA256:
        raise ValueError(f"D2-O selection mismatch: {result['sha256']}")
    return result


def sampler_seed_label(chunk_index: int, step: int) -> str:
    if chunk_index < 0 or step not in range(WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-O sampler seed coordinates")
    return f"D2:d2o-shared:chunk:{chunk_index}:step:{step}"


def binary_counts(prediction, target) -> Dict[str, int]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 1 or not len(prediction):
        raise ValueError("binary contact metrics require equal non-empty vectors")
    return {
        "tp": int(np.logical_and(prediction, target).sum()),
        "fp": int(np.logical_and(prediction, ~target).sum()),
        "tn": int(np.logical_and(~prediction, ~target).sum()),
        "fn": int(np.logical_and(~prediction, target).sum()),
    }


def metrics_from_counts(counts: Mapping[str, int]) -> Dict[str, object]:
    tp, fp, tn, fn = (int(counts[name]) for name in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    if total <= 0:
        raise ValueError("contact counts are empty")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction_percent": (tp + fp) / total,
        "target_percent": (tp + fn) / total,
        "frames": total,
    }


def unit_binary_report(prediction, target) -> Dict[str, object]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 2
        or not len(prediction)
    ):
        raise ValueError("hand contact metrics require matching [frames,2] arrays")
    values = {
        "left_hand": (prediction[:, 0], target[:, 0]),
        "right_hand": (prediction[:, 1], target[:, 1]),
        "union": (prediction.any(axis=1), target.any(axis=1)),
    }
    return {
        unit: metrics_from_counts(binary_counts(*pair))
        for unit, pair in values.items()
    }


def _smooth_l1(error: np.ndarray) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(absolute < 1.0, 0.5 * error ** 2, absolute - 0.5)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("quantiles require finite non-empty values")
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p05", "p25", "median", "p75", "p95", "max"),
            np.quantile(values, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)),
        )
    }


def calibration_report(prediction, target, bins: int = 10) -> Dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if prediction.shape != target.shape or not len(prediction):
        raise ValueError("calibration requires equal non-empty vectors")
    clipped = np.clip(prediction, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (clipped >= edges[index]) & (clipped <= edges[index + 1])
        else:
            selected = (clipped >= edges[index]) & (clipped < edges[index + 1])
        count = int(selected.sum())
        confidence = float(clipped[selected].mean()) if count else None
        frequency = float(target[selected].mean()) if count else None
        if count:
            ece += count / len(clipped) * abs(confidence - frequency)
        rows.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_confidence": confidence,
            "positive_frequency": frequency,
        })
    return {
        "brier": float(np.square(clipped - target.astype(np.float64)).mean()),
        "expected_calibration_error": float(ece),
        "raw_outside_0_1_percent": float(
            np.logical_or(prediction < 0.0, prediction > 1.0).mean()
        ),
        "bins": rows,
    }


def semantic_report(prediction, target) -> Dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 4
        or not len(prediction)
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("semantic report requires finite matching [frames,4] arrays")
    error = prediction - target
    result = {
        "all_four_mse": float(np.square(error).mean()),
        "all_four_smooth_l1": float(_smooth_l1(error).mean()),
        "first_two_mse": float(np.square(error[:, :2]).mean()),
        "first_two_smooth_l1": float(_smooth_l1(error[:, :2]).mean()),
        "per_channel": {
            str(channel): {
                "mse": float(np.square(error[:, channel]).mean()),
                "smooth_l1": float(_smooth_l1(error[:, channel]).mean()),
                "prediction_quantiles": _quantiles(prediction[:, channel]),
                "target_percent": float((target[:, channel] >= 0.5).mean()),
            }
            for channel in range(4)
        },
        "first_two_calibration": calibration_report(
            prediction[:, :2], target[:, :2] >= 0.5,
        ),
        "thresholds": {},
    }
    for threshold in SEMANTIC_THRESHOLDS:
        prediction_contact = prediction[:, :2] >= threshold
        target_contact = target[:, :2] >= 0.5
        report = unit_binary_report(prediction_contact, target_contact)
        for unit, predicted, truth in (
            ("left_hand", prediction_contact[:, 0], target_contact[:, 0]),
            ("right_hand", prediction_contact[:, 1], target_contact[:, 1]),
            ("union", prediction_contact.any(axis=1), target_contact.any(axis=1)),
        ):
            report[unit]["prediction_run_lengths"] = contact_run_lengths(predicted)
            report[unit]["target_run_lengths"] = contact_run_lengths(truth)
        result["thresholds"][_threshold_key(threshold)] = report
    return result


def contact_run_lengths(values) -> Dict[str, object]:
    values = np.asarray(values, dtype=bool).reshape(-1)
    lengths = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return {
        "runs": len(lengths),
        "mean_frames": float(np.mean(lengths)) if lengths else 0.0,
        "max_frames": int(max(lengths)) if lengths else 0,
        "lengths": lengths,
    }


def geometry_report(prediction_distance_m, target_distance_m) -> Dict[str, object]:
    prediction = np.asarray(prediction_distance_m, dtype=np.float64)
    target = np.asarray(target_distance_m, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 2
        or not len(prediction)
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("geometry report requires finite matching [frames,2] distances")
    result = {
        "prediction_distance_cm": {
            "left_hand": _quantiles(prediction[:, 0] * 100.0),
            "right_hand": _quantiles(prediction[:, 1] * 100.0),
            "union": _quantiles(prediction.min(axis=1) * 100.0),
        },
        "target_distance_cm": {
            "left_hand": _quantiles(target[:, 0] * 100.0),
            "right_hand": _quantiles(target[:, 1] * 100.0),
            "union": _quantiles(target.min(axis=1) * 100.0),
        },
        "thresholds_cm": {},
    }
    for threshold_cm in PHYSICAL_THRESHOLDS_CM:
        threshold_m = threshold_cm / 100.0
        prediction_contact = prediction < threshold_m
        target_contact = target < threshold_m
        report = unit_binary_report(prediction_contact, target_contact)
        for unit, values in (
            ("left_hand", prediction_contact[:, 0]),
            ("right_hand", prediction_contact[:, 1]),
            ("union", prediction_contact.any(axis=1)),
        ):
            report[unit]["prediction_run_lengths"] = contact_run_lengths(values)
        for unit, values in (
            ("left_hand", target_contact[:, 0]),
            ("right_hand", target_contact[:, 1]),
            ("union", target_contact.any(axis=1)),
        ):
            report[unit]["target_run_lengths"] = contact_run_lengths(values)
        result["thresholds_cm"][_threshold_key(threshold_cm)] = report
    return result


def semantic_geometry_report(
    semantic,
    distance_m,
    *,
    semantic_threshold: float = 0.5,
) -> Dict[str, object]:
    semantic = np.asarray(semantic, dtype=np.float64)
    distance = np.asarray(distance_m, dtype=np.float64)
    if semantic.ndim != 2 or semantic.shape[1] < 2:
        raise ValueError("semantic/geometry alignment requires at least two channels")
    if distance.shape != (semantic.shape[0], 2):
        raise ValueError("semantic/geometry frame counts differ")
    return {
        _threshold_key(threshold_cm): unit_binary_report(
            semantic[:, :2] >= semantic_threshold,
            distance < threshold_cm / 100.0,
        )
        for threshold_cm in PHYSICAL_THRESHOLDS_CM
    }


def object_vertices(
    rest_vertices: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        raise ValueError("rest vertices must be [vertices,3]")
    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError("object rotations must be [frames,3,3]")
    if translation.shape != (rotation.shape[0], 3):
        raise ValueError("object translations must be [frames,3]")
    return (
        rest_vertices.to(rotation)[None]
        @ project_to_so3(rotation).transpose(-1, -2)
        + translation[:, None]
    )


def hand_object_distances(
    joints: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:
    if joints.ndim != 3 or joints.shape[1] <= max(HAND_INDICES) or joints.shape[2] != 3:
        raise ValueError("joints must contain the locked left/right hand indices")
    if vertices.ndim != 3 or vertices.shape[0] != joints.shape[0] or vertices.shape[2] != 3:
        raise ValueError("object vertices must be [frames,vertices,3]")
    return torch.cdist(joints[:, HAND_INDICES], vertices).amin(dim=-1)


def distance_decomposition(
    generated_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    generated_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if generated_joints.shape != gt_joints.shape:
        raise ValueError("generated and GT joints differ")
    if generated_vertices.shape != gt_vertices.shape:
        raise ValueError("generated and GT object vertices differ")
    return {
        "generated_human_generated_object": hand_object_distances(
            generated_joints, generated_vertices,
        ),
        "gt_human_generated_object": hand_object_distances(
            gt_joints, generated_vertices,
        ),
        "generated_human_gt_object": hand_object_distances(
            generated_joints, gt_vertices,
        ),
        "gt_human_gt_object": hand_object_distances(
            gt_joints, gt_vertices,
        ),
    }


def _distance_distribution(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"frames": 0, "mean_cm": None, "quantiles_cm": None}
    return {
        "frames": len(values),
        "mean_cm": float(values.mean() * 100.0),
        "quantiles_cm": _quantiles(values * 100.0),
        "recall_by_threshold_cm": {
            _threshold_key(threshold): float((values < threshold / 100.0).mean())
            for threshold in PHYSICAL_THRESHOLDS_CM
        },
    }


def decomposition_report(
    decomposition: Mapping[str, np.ndarray],
    gt_distance_m,
) -> Dict[str, object]:
    gt_distance = np.asarray(gt_distance_m, dtype=np.float64)
    if gt_distance.ndim != 2 or gt_distance.shape[1] != 2:
        raise ValueError("GT distance must be [frames,2]")
    masks = {
        "left_hand": gt_distance[:, 0] < 0.05,
        "right_hand": gt_distance[:, 1] < 0.05,
        "union": (gt_distance < 0.05).any(axis=1),
    }
    result = {}
    for path in DECOMPOSITION_PATHS:
        values = np.asarray(decomposition[path], dtype=np.float64)
        if values.shape != gt_distance.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid decomposition path {path}")
        result[path] = {
            "left_hand": _distance_distribution(values[masks["left_hand"], 0]),
            "right_hand": _distance_distribution(values[masks["right_hand"], 1]),
            "union": _distance_distribution(values.min(axis=1)[masks["union"]]),
        }
    return result


def classification_gate(
    contract: Mapping[str, bool],
    gt_alignment: Mapping[str, object],
    comparisons: Mapping[str, object],
) -> Dict[str, object]:
    contract_passed = bool(contract) and all(bool(value) for value in contract.values())
    gt_union = gt_alignment[_threshold_key(5.0)]["union"]
    gt_contract_passed = bool(
        float(gt_union["f1"]) >= GT_ALIGNMENT_MIN
        and float(gt_union["recall"]) >= GT_ALIGNMENT_MIN
    )
    comparator_checks = {}
    for comparator in ("source", "current"):
        record = comparisons[f"balanced_vs_{comparator}"]
        semantic_ci = record["comparator_minus_balanced_semantic_first_two_mse"][
            "bootstrap_95_ci"
        ]
        recall_ci = record["comparator_minus_balanced_physical_recall_5cm"][
            "bootstrap_95_ci"
        ]
        comparator_checks[comparator] = {
            "semantic_mse_improvement_ci_lower_gt_zero": float(semantic_ci[0]) > 0.0,
            "physical_recall_deficit_ci_lower_gt_zero": float(recall_ci[0]) > 0.0,
        }
    decoupling = all(
        all(checks.values()) for checks in comparator_checks.values()
    )
    if not contract_passed:
        classification = "contact-alignment-contract-failure-stop"
    elif not gt_contract_passed:
        classification = "label-evaluator-contract-mismatch-stop"
    elif decoupling:
        classification = "semantic-geometry-decoupling-positive-stop"
    else:
        classification = "mixed-contact-deficit-stop"
    return {
        "classification": classification,
        "contract_passed": contract_passed,
        "gt_label_evaluator_contract_passed": gt_contract_passed,
        "semantic_geometry_decoupling_proved": bool(
            contract_passed and gt_contract_passed and decoupling
        ),
        "contract_checks": dict(contract),
        "gt_union_5cm": gt_union,
        "comparator_checks": comparator_checks,
        "checkpoint_selected": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "d2g_started": False,
    }


def all_finite(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True
