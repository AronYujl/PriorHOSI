"""Locked statistics and gates for the Phase 1B D2-AE0 diagnostics.

This module deliberately contains no rollout, checkpoint, dataset, or official
evaluator code.  D2-AE reuses the sealed D2-AC/D2-AD paired-statistics helpers
and adds only the preregistered sparse-relation temporal/role comparisons and
classification precedence.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from .interaction_diagnostic import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIRECT_HAND_INDICES,
    FK_PALM_INDICES,
    GT_CONTACT_FINITE_SEQUENCE_COUNT,
    GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256,
    HISTORY_MAX_ABS,
    PHYSICAL_THRESHOLDS_CM,
    PROTECTION_METRICS,
    RELEASED_HIGHER_IS_BETTER,
    RELEASED_LOWER_IS_BETTER,
    SELECTION_SHA256,
    native_gate as _shared_native_gate,
    paired_difference_fixed,
    paired_finite_difference,
    paired_nonnegative_ratio_fixed,
    paired_ratio_fixed,
)


VARIANTS: Tuple[str, ...] = (
    "full",
    "relation_gate_ablated",
    "temporal_correspondence_permuted",
    "left_right_role_swapped",
)
ROLE_NAMES: Tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "pelvis",
)
TEMPORAL_ANCHORS: Tuple[int, ...] = (0, 5, 10, 15)
CONTACT_UNITS: Tuple[str, ...] = ("left_hand", "right_hand", "union")
CONTACT_METRICS: Tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "prediction_percent",
)
KINEMATIC_METRICS: Tuple[str, ...] = (
    "mpjpe_cm",
    "object_goal_error_cm",
    "pelvis_goal_error_cm",
    "object_translation_mae_cm",
    "object_rotation_geodesic_deg",
    "foot_sliding",
)
CONTACT_F1_POINT_MINIMUM = 0.6598838781


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


def _direct_left_right_macro_f1(record: Mapping[str, object]) -> float:
    thresholds = _path(
        record,
        ("direct_physical_geometry_vs_gt", "thresholds_cm", "5"),
    )
    if not isinstance(thresholds, Mapping):
        raise ValueError("D2-AE direct-hand 5-cm report is malformed")
    left = float(_path(thresholds, ("left_hand", "f1")))
    right = float(_path(thresholds, ("right_hand", "f1")))
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("D2-AE direct-hand macro-F1 requires finite values")
    return 0.5 * (left + right)


def paired_comparisons(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    """Build all fixed paired sequence comparisons for the four D2-AE paths."""
    if set(records) != set(VARIANTS):
        raise ValueError("D2-AE diagnostic variants differ from the locked four paths")
    names = [str(record["sequence"]) for record in records["full"]]
    if not names:
        raise ValueError("D2-AE paired comparison is empty")
    for variant in VARIANTS[1:]:
        if [str(record["sequence"]) for record in records[variant]] != names:
            raise ValueError("D2-AE paired sequence ordering differs")

    result: Dict[str, object] = {}
    for other in VARIANTS[1:]:
        comparison: Dict[str, object] = {
            "full_minus_other_direct_union_5cm_f1": paired_difference_fixed(
                _values(
                    records["full"],
                    (
                        "direct_physical_geometry_vs_gt",
                        "thresholds_cm",
                        "5",
                        "union",
                        "f1",
                    ),
                ),
                _values(
                    records[other],
                    (
                        "direct_physical_geometry_vs_gt",
                        "thresholds_cm",
                        "5",
                        "union",
                        "f1",
                    ),
                ),
            ),
            "full_minus_other_direct_left_right_macro_5cm_f1": (
                paired_difference_fixed(
                    [_direct_left_right_macro_f1(row) for row in records["full"]],
                    [_direct_left_right_macro_f1(row) for row in records[other]],
                )
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
        semantic = comparison["semantic_contact"]
        assert isinstance(semantic, dict)
        for unit in CONTACT_UNITS:
            semantic[unit] = {
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
                for metric in CONTACT_METRICS
            }

        physical = comparison["physical_contact"]
        assert isinstance(physical, dict)
        for geometry in (
            "direct_physical_geometry_vs_gt",
            "fk_physical_geometry_vs_gt",
        ):
            physical[geometry] = {}
            for threshold in PHYSICAL_THRESHOLDS_CM:
                threshold_key = f"{threshold:g}"
                physical[geometry][threshold_key] = {}
                for unit in CONTACT_UNITS:
                    unit_value = {
                        metric: paired_difference_fixed(
                            _values(
                                records["full"],
                                (
                                    geometry,
                                    "thresholds_cm",
                                    threshold_key,
                                    unit,
                                    metric,
                                ),
                            ),
                            _values(
                                records[other],
                                (
                                    geometry,
                                    "thresholds_cm",
                                    threshold_key,
                                    unit,
                                    metric,
                                ),
                            ),
                        )
                        for metric in CONTACT_METRICS
                    }
                    unit_value["prediction_run_mean_frames"] = (
                        paired_difference_fixed(
                            _values(
                                records["full"],
                                (
                                    geometry,
                                    "thresholds_cm",
                                    threshold_key,
                                    unit,
                                    "prediction_run_lengths",
                                    "mean_frames",
                                ),
                            ),
                            _values(
                                records[other],
                                (
                                    geometry,
                                    "thresholds_cm",
                                    threshold_key,
                                    unit,
                                    "prediction_run_lengths",
                                    "mean_frames",
                                ),
                            ),
                        )
                    )
                    physical[geometry][threshold_key][unit] = unit_value

        kinematics = comparison["kinematics"]
        assert isinstance(kinematics, dict)
        for metric in KINEMATIC_METRICS:
            kinematics[metric] = paired_ratio_fixed(
                _values(records["full"], ("kinematics", metric)),
                _values(records[other], ("kinematics", metric)),
            )

        penetration = comparison["penetration"]
        assert isinstance(penetration, dict)
        for metric in ("hand_pen_loss_omomo", "human_pen_loss_infbagel"):
            full = _values(records["full"], ("penetration", metric))
            alternate = _values(records[other], ("penetration", metric))
            keep = [
                index
                for index, (left, right) in enumerate(zip(full, alternate))
                if left is not None and right is not None
            ]
            value = paired_nonnegative_ratio_fixed(
                [float(full[index]) for index in keep],
                [float(alternate[index]) for index in keep],
            )
            value["finite_sequence_count"] = len(keep)
            value["finite_sequence_names"] = [names[index] for index in keep]
            penetration[metric] = value
        result[f"full_vs_{other}"] = comparison
    return result


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
    """Apply the five preregistered D2-AE causal gates in fixed precedence."""
    contract_passed = bool(contract) and all(bool(value) for value in contract.values())
    ablated = comparisons.get("full_vs_relation_gate_ablated", {})
    temporal = comparisons.get("full_vs_temporal_correspondence_permuted", {})
    role = comparisons.get("full_vs_left_right_role_swapped", {})
    checks = {
        "full_minus_gate_ablated_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(ablated.get("full_minus_other_direct_union_5cm_f1", {}))
        ),
        "full_minus_temporal_permuted_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(temporal.get("full_minus_other_direct_union_5cm_f1", {}))
        ),
        "full_minus_role_swapped_direct_left_right_macro_f1_ci_lower_gt_zero": (
            _positive_lower(
                role.get("full_minus_other_direct_left_right_macro_5cm_f1", {})
            )
        ),
        "gate_ablated_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(ablated.get("other_minus_full_gt_contact_distance_cm", {}))
        ),
        "temporal_permuted_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(temporal.get("other_minus_full_gt_contact_distance_cm", {}))
        ),
    }
    relation_path_used = bool(
        checks["full_minus_gate_ablated_direct_union_5cm_f1_ci_lower_gt_zero"]
        and checks["gate_ablated_minus_full_gt_contact_distance_ci_lower_gt_zero"]
    )
    temporal_routing_passed = bool(
        checks["full_minus_temporal_permuted_direct_union_5cm_f1_ci_lower_gt_zero"]
        and checks["temporal_permuted_minus_full_gt_contact_distance_ci_lower_gt_zero"]
    )
    role_binding_passed = bool(
        checks[
            "full_minus_role_swapped_direct_left_right_macro_f1_ci_lower_gt_zero"
        ]
    )
    mechanism_passed = bool(
        contract_passed
        and relation_path_used
        and temporal_routing_passed
        and role_binding_passed
    )
    if not contract_passed:
        classification = "sparse-relation-field-contract-failure-stop"
    elif not relation_path_used:
        classification = "sparse-relation-field-unused-optimization-negative-stop"
    elif not temporal_routing_passed:
        classification = "sparse-relation-field-temporal-routing-negative-stop"
    elif not role_binding_passed:
        classification = "sparse-relation-field-role-binding-negative-stop"
    else:
        classification = "sparse-relation-field-internal-positive-continue"
    return {
        "classification": classification,
        "contract_passed": contract_passed,
        "relation_path_used": relation_path_used,
        "temporal_routing_passed": temporal_routing_passed,
        "role_binding_passed": role_binding_passed,
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
    """Reuse locked native math, adding D2-AE's point and role gates."""
    relation_path_used = bool(internal.get("relation_path_used", False))
    temporal_routing_passed = bool(
        internal.get("temporal_routing_passed", False)
    )
    role_binding_passed = bool(internal.get("role_binding_passed", False))
    shared = _shared_native_gate(
        contract_passed=contract_passed,
        internal={
            "contract_passed": bool(internal.get("contract_passed", False)),
            "adapter_used": relation_path_used,
            "locality_passed": bool(
                temporal_routing_passed and role_binding_passed
            ),
            "mechanism_passed": bool(internal.get("mechanism_passed", False)),
        },
        comparison=comparison,
        target_metrics=target_metrics,
        baseline_ratios=baseline_ratios,
    )
    target_f1 = float(target_metrics.get("contact_f1", float("nan")))
    transfer_checks = dict(shared["native_transfer_checks"])
    transfer_checks[
        "contact_f1_point_estimate_ge_0.6598838781"
    ] = bool(math.isfinite(target_f1) and target_f1 >= CONTACT_F1_POINT_MINIMUM)
    transfer_passed = all(bool(value) for value in transfer_checks.values())
    full_contract = bool(shared["contract_passed"])
    if not full_contract:
        classification = "sparse-relation-field-contract-failure-stop"
    elif not relation_path_used:
        classification = "sparse-relation-field-unused-optimization-negative-stop"
    elif not temporal_routing_passed:
        classification = "sparse-relation-field-temporal-routing-negative-stop"
    elif not role_binding_passed:
        classification = "sparse-relation-field-role-binding-negative-stop"
    elif not transfer_passed:
        classification = "sparse-relation-field-transfer-negative-stop"
    elif not bool(shared["protection_passed"]):
        classification = "sparse-relation-field-conflict-negative-stop"
    elif not bool(shared["released_95_percent_effectiveness_passed"]):
        classification = "sparse-relation-field-positive-but-not-effective-stop"
    else:
        classification = "sparse-relation-field-positive-candidate-stop"
    selectable = classification == "sparse-relation-field-positive-candidate-stop"

    value = dict(shared)
    value.pop("adapter_used", None)
    value.pop("locality_passed", None)
    value.pop("d2ac1_authorized", None)
    value.update({
        "classification": classification,
        "relation_path_used": relation_path_used,
        "temporal_routing_passed": temporal_routing_passed,
        "role_binding_passed": role_binding_passed,
        "native_transfer_passed": transfer_passed,
        "native_transfer_checks": transfer_checks,
        "contact_f1_point_estimate": target_f1,
        "contact_f1_point_minimum": CONTACT_F1_POINT_MINIMUM,
        "checkpoint_selected": selectable,
        "selectable_autonomous_diffusion_candidate": selectable,
        "consistency_authorized": False,
        "consistency_started": False,
        "d2ae1_authorized": False,
    })
    return value


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONTACT_F1_POINT_MINIMUM",
    "DIRECT_HAND_INDICES",
    "FK_PALM_INDICES",
    "GT_CONTACT_FINITE_SEQUENCE_COUNT",
    "GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256",
    "HISTORY_MAX_ABS",
    "PHYSICAL_THRESHOLDS_CM",
    "PROTECTION_METRICS",
    "RELEASED_HIGHER_IS_BETTER",
    "RELEASED_LOWER_IS_BETTER",
    "ROLE_NAMES",
    "SELECTION_SHA256",
    "TEMPORAL_ANCHORS",
    "VARIANTS",
    "internal_mechanism_gate",
    "native_gate",
    "paired_comparisons",
]
