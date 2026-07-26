"""Locked metrics and gates for the Phase 1B D2-AC0 diagnostics.

The functions in this module are evaluator-independent.  They contain only the
pre-registered paired-statistics and classification logic; rollout execution,
checkpoint I/O, and the official evaluator remain in worker tools.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .optimizer_reset import paired_difference, paired_mean_ratio


VARIANTS: Tuple[str, ...] = (
    "full",
    "gate_ablated",
    "local_correspondence_permuted",
)
ROLE_NAMES: Tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "object_motion",
)
DIRECT_HAND_INDICES: Tuple[int, int] = (24, 26)
FK_PALM_INDICES: Tuple[int, int] = (22, 23)
PHYSICAL_THRESHOLDS_CM: Tuple[float, ...] = (2.0, 5.0, 7.5, 10.0)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
SELECTION_SHA256 = (
    "1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a"
)
GT_CONTACT_FINITE_SEQUENCE_COUNT = 57
GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256 = (
    "2fa79d30ab6dd6a915098344c4aa7267cb6c3323c6d2a762b4b704f8757cebaa"
)
HISTORY_MAX_ABS = 1.0e-5

PROTECTION_METRICS: Tuple[str, ...] = (
    "end_obj_trans_err",
    "xy_points_err",
    "foot_sliding",
    "human_pen_loss_infbagel",
    "hand_pen_loss_omomo",
    "mpjpe",
    "trans_dist",
    "obj_trans_dist",
    "obj_rot_dist",
)
PENETRATION_METRICS: Tuple[str, ...] = (
    "hand_pen_loss_omomo",
    "human_pen_loss_infbagel",
)
RELEASED_LOWER_IS_BETTER: Tuple[str, ...] = (
    "end_obj_trans_err",
    "xy_points_err",
    "foot_sliding",
    "human_pen_loss_infbagel",
    "mpjpe",
    "trans_dist",
    "obj_trans_dist",
    "obj_rot_dist",
)
RELEASED_HIGHER_IS_BETTER: Tuple[str, ...] = (
    "contact_precision",
    "contact_recall",
    "contact_f1",
)


def attention_entropy(weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return raw and normalized entropy for ``[..., object_token]`` weights."""
    if weights.ndim < 2 or weights.shape[-1] != 16:
        raise ValueError("D2-AC attention weights must end in 16 object tokens")
    if not torch.is_floating_point(weights) or not torch.isfinite(weights).all():
        raise ValueError("D2-AC attention weights must be finite floating point")
    if bool((weights < 0).any()):
        raise ValueError("D2-AC attention weights must be non-negative")
    denominator = weights.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("D2-AC attention weights have zero mass")
    probability = weights / denominator
    entropy = -(
        probability * torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))
    ).sum(dim=-1)
    return {
        "nats": entropy,
        "normalized": entropy / math.log(weights.shape[-1]),
    }


def gt_contact_frame_distance(
    predicted_distance_m: np.ndarray,
    target_distance_m: np.ndarray,
    *,
    threshold_m: float = 0.05,
) -> Dict[str, object]:
    """Describe generated distance on frames with GT physical hand contact.

    A unit with no GT-contact frames is explicitly marked non-finite rather
    than assigned a fabricated zero.  Paired sequence inference therefore uses
    the fixed target-derived finite mask, shared by every D2-AC variant.
    """
    predicted = np.asarray(predicted_distance_m, dtype=np.float64)
    target = np.asarray(target_distance_m, dtype=np.float64)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 2
        or not len(predicted)
        or not np.isfinite(predicted).all()
        or not np.isfinite(target).all()
        or threshold_m <= 0.0
    ):
        raise ValueError("GT-contact distance requires finite matching [frames,2] arrays")

    units = {
        "left_hand": (target[:, 0] < threshold_m, predicted[:, 0]),
        "right_hand": (target[:, 1] < threshold_m, predicted[:, 1]),
        "union": (
            (target < threshold_m).any(axis=1),
            predicted.min(axis=1),
        ),
    }
    result: Dict[str, object] = {}
    for name, (mask, values) in units.items():
        selected = values[mask]
        result[name] = {
            "frames": int(mask.sum()),
            "mean_m": float(selected.mean()) if len(selected) else None,
            "mean_cm": float(selected.mean() * 100.0) if len(selected) else None,
            "finite": bool(len(selected)),
        }
    return result


def paired_difference_fixed(
    first: Sequence[float],
    second: Sequence[float],
) -> Dict[str, object]:
    return paired_difference(
        first,
        second,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )


def paired_ratio_fixed(
    numerator: Sequence[float],
    denominator: Sequence[float],
) -> Dict[str, object]:
    return paired_mean_ratio(
        numerator,
        denominator,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )


def paired_finite_difference(
    first: Sequence[object],
    second: Sequence[object],
    sequence_names: Sequence[str],
) -> Dict[str, object]:
    """Bootstrap a target-derived common finite subset without imputation."""
    if len(first) != len(second) or len(first) != len(sequence_names):
        raise ValueError("D2-AC finite-mask vectors differ")
    keep = [
        index
        for index, (left, right) in enumerate(zip(first, second))
        if left is not None
        and right is not None
        and math.isfinite(float(left))
        and math.isfinite(float(right))
    ]
    if not keep:
        raise ValueError("D2-AC finite-mask comparison is empty")
    value = paired_difference_fixed(
        [float(first[index]) for index in keep],
        [float(second[index]) for index in keep],
    )
    value["finite_sequence_count"] = len(keep)
    value["finite_sequence_names"] = [str(sequence_names[index]) for index in keep]
    return value


def _positive_lower(value: Mapping[str, object]) -> bool:
    interval = value.get("bootstrap_95_ci", ())
    return bool(
        isinstance(interval, (list, tuple))
        and len(interval) == 2
        and math.isfinite(float(interval[0]))
        and float(interval[0]) > 0.0
    )


def internal_mechanism_gate(
    contract: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Apply the four fixed causal gates in their pre-registered order."""
    contract_passed = bool(contract) and all(bool(value) for value in contract.values())
    ablated = comparisons.get("full_vs_gate_ablated", {})
    permuted = comparisons.get("full_vs_local_correspondence_permuted", {})
    checks = {
        "full_minus_gate_ablated_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(ablated.get("full_minus_other_direct_union_5cm_f1", {}))
        ),
        "full_minus_permuted_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(permuted.get("full_minus_other_direct_union_5cm_f1", {}))
        ),
        "gate_ablated_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(ablated.get("other_minus_full_gt_contact_distance_cm", {}))
        ),
        "permuted_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(permuted.get("other_minus_full_gt_contact_distance_cm", {}))
        ),
    }
    adapter_used = bool(
        checks[
            "full_minus_gate_ablated_direct_union_5cm_f1_ci_lower_gt_zero"
        ]
        and checks[
            "gate_ablated_minus_full_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    locality_passed = bool(
        checks[
            "full_minus_permuted_direct_union_5cm_f1_ci_lower_gt_zero"
        ]
        and checks[
            "permuted_minus_full_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    mechanism_passed = contract_passed and adapter_used and locality_passed
    if not contract_passed:
        classification = "interaction-adapter-contract-failure-stop"
    elif not adapter_used:
        classification = "interaction-adapter-unused-optimization-negative-stop"
    elif not locality_passed:
        classification = "interaction-adapter-locality-negative-stop"
    else:
        classification = "interaction-adapter-internal-positive-continue"
    return {
        "classification": classification,
        "contract_passed": contract_passed,
        "adapter_used": adapter_used,
        "locality_passed": locality_passed,
        "mechanism_passed": mechanism_passed,
        "checks": checks,
        "checkpoint_selected": False,
        "consistency_authorized": False,
    }


def native_gate(
    *,
    contract_passed: bool,
    internal: Mapping[str, object],
    comparison: Mapping[str, object],
    target_metrics: Mapping[str, object],
    baseline_ratios: Mapping[str, float],
) -> Dict[str, object]:
    """Apply transfer, protection, and released-effectiveness gates."""
    mask_passed = bool(
        comparison.get("penetration_mask_contract", {}).get("passed", False)
    )
    f1 = comparison.get("target_minus_control_contact_f1", {})
    recall = comparison.get("target_minus_control_contact_recall", {})
    gap_closure = float(comparison.get("contact_f1_released_gap_closure", float("nan")))
    transfer_checks = {
        "contact_f1_ci_lower_gt_zero": _positive_lower(f1),
        "contact_recall_ci_lower_gt_zero": _positive_lower(recall),
        "contact_f1_released_gap_closure_ge_0.25": (
            math.isfinite(gap_closure) and gap_closure >= 0.25
        ),
    }
    transfer_passed = all(transfer_checks.values())

    ratios = comparison.get("target_over_control_protection", {})
    protection_checks = {
        f"{metric}_ratio_ci_upper_le_1.10": bool(
            metric in ratios
            and len(ratios[metric].get("bootstrap_95_ci", ())) == 2
            and math.isfinite(float(ratios[metric]["bootstrap_95_ci"][1]))
            and float(ratios[metric]["bootstrap_95_ci"][1]) <= 1.10
        )
        for metric in PROTECTION_METRICS
    }
    precision = comparison.get("target_minus_control_contact_precision", {})
    precision_interval = precision.get("bootstrap_95_ci", ())
    protection_checks["contact_precision_ci_lower_ge_minus_0.02"] = bool(
        len(precision_interval) == 2
        and math.isfinite(float(precision_interval[0]))
        and float(precision_interval[0]) >= -0.02
    )
    protection_checks["penetration_finite_mask_contract"] = mask_passed
    protection_passed = all(protection_checks.values())

    released_checks = {
        f"{metric}_target_over_released_le_1_over_0.95": bool(
            metric in baseline_ratios
            and math.isfinite(float(baseline_ratios[metric]))
            and float(baseline_ratios[metric]) <= 1.0 / 0.95
        )
        for metric in RELEASED_LOWER_IS_BETTER
    }
    released_checks.update({
        f"{metric}_target_over_released_ge_0.95": bool(
            metric in baseline_ratios
            and math.isfinite(float(baseline_ratios[metric]))
            and float(baseline_ratios[metric]) >= 0.95
        )
        for metric in RELEASED_HIGHER_IS_BETTER
    })
    released_effective = all(released_checks.values())

    internal_contract = bool(internal.get("contract_passed", False))
    adapter_used = bool(internal.get("adapter_used", False))
    locality_passed = bool(internal.get("locality_passed", False))
    full_contract = bool(contract_passed and mask_passed and internal_contract)
    if not full_contract:
        classification = "interaction-adapter-contract-failure-stop"
    elif not adapter_used:
        classification = "interaction-adapter-unused-optimization-negative-stop"
    elif not locality_passed:
        classification = "interaction-adapter-locality-negative-stop"
    elif not transfer_passed:
        classification = "interaction-adapter-transfer-negative-stop"
    elif not protection_passed:
        classification = "interaction-adapter-conflict-negative-stop"
    elif not released_effective:
        classification = "interaction-adapter-positive-but-not-effective-stop"
    else:
        classification = "interaction-adapter-positive-candidate-stop"
    selectable = classification == "interaction-adapter-positive-candidate-stop"
    return {
        "classification": classification,
        "contract_passed": full_contract,
        "internal_mechanism_passed": bool(internal.get("mechanism_passed", False)),
        "adapter_used": adapter_used,
        "locality_passed": locality_passed,
        "native_transfer_passed": transfer_passed,
        "native_transfer_checks": transfer_checks,
        "protection_passed": protection_passed,
        "protection_checks": protection_checks,
        "released_95_percent_effectiveness_passed": released_effective,
        "released_95_percent_checks": released_checks,
        "contact_f1_released_gap_closure": gap_closure,
        "checkpoint_selected": selectable,
        "selectable_autonomous_diffusion_candidate": selectable,
        "consistency_authorized": False,
        "consistency_started": False,
        "d2ac1_authorized": False,
        "target_metrics": {
            key: target_metrics[key]
            for key in sorted(target_metrics)
            if key in set(RELEASED_LOWER_IS_BETTER) | set(RELEASED_HIGHER_IS_BETTER)
        },
    }
