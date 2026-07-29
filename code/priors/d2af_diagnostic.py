"""Locked statistics and gates for Phase 1B D2-AF0.

The module is deliberately independent of rollout, checkpoint, dataset, and
official-evaluator code.  It adds the registered sqrt(alpha_bar) schedule
counterfactual and the sealed D2-AE repair comparison while retaining the
previous sparse-relation path, temporal, role, protection, and effectiveness
statistics.
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
    "full_rho",
    "unit_rho",
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
        raise ValueError("D2-AF direct-hand 5-cm report is malformed")
    left = float(_path(thresholds, ("left_hand", "f1")))
    right = float(_path(thresholds, ("right_hand", "f1")))
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("D2-AF direct-hand macro-F1 requires finite values")
    return 0.5 * (left + right)


def paired_comparisons(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    """Build the fixed paired sequence comparisons for all five paths."""
    if set(records) != set(VARIANTS):
        raise ValueError("D2-AF diagnostic variants differ from the locked five paths")
    names = [str(record["sequence"]) for record in records["full_rho"]]
    if not names:
        raise ValueError("D2-AF paired comparison is empty")
    for variant in VARIANTS[1:]:
        if [str(record["sequence"]) for record in records[variant]] != names:
            raise ValueError("D2-AF paired sequence ordering differs")

    result: Dict[str, object] = {}
    for other in VARIANTS[1:]:
        comparison: Dict[str, object] = {
            "full_rho_minus_other_direct_union_5cm_f1": paired_difference_fixed(
                _values(
                    records["full_rho"],
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
            "full_rho_minus_other_direct_left_right_macro_5cm_f1": (
                paired_difference_fixed(
                    [
                        _direct_left_right_macro_f1(row)
                        for row in records["full_rho"]
                    ],
                    [_direct_left_right_macro_f1(row) for row in records[other]],
                )
            ),
            "other_minus_full_rho_gt_contact_distance_cm": (
                paired_finite_difference(
                    _values(
                        records[other],
                        ("gt_contact_frame_direct_distance", "union", "mean_cm"),
                    ),
                    _values(
                        records["full_rho"],
                        ("gt_contact_frame_direct_distance", "union", "mean_cm"),
                    ),
                    names,
                )
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
                        records["full_rho"],
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
                                records["full_rho"],
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
                                records["full_rho"],
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
                _values(records["full_rho"], ("kinematics", metric)),
                _values(records[other], ("kinematics", metric)),
            )

        penetration = comparison["penetration"]
        assert isinstance(penetration, dict)
        for metric in ("hand_pen_loss_omomo", "human_pen_loss_infbagel"):
            full = _values(records["full_rho"], ("penetration", metric))
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
        result[f"full_rho_vs_{other}"] = comparison
    return result


def _positive_lower(value: Mapping[str, object]) -> bool:
    interval = value.get("bootstrap_95_ci", ())
    return bool(
        isinstance(interval, (list, tuple))
        and len(interval) == 2
        and math.isfinite(float(interval[0]))
        and float(interval[0]) > 0.0
    )


def _ratio_upper(
    value: Mapping[str, object],
    maximum: float,
    *,
    strict: bool,
) -> bool:
    interval = value.get("bootstrap_95_ci", ())
    if not (
        isinstance(interval, (list, tuple))
        and len(interval) == 2
        and math.isfinite(float(interval[1]))
    ):
        return False
    upper = float(interval[1])
    return upper < maximum if strict else upper <= maximum


def internal_mechanism_gate(
    contract: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Apply the seven registered D2-AF internal gates."""
    contract_passed = bool(contract) and all(bool(value) for value in contract.values())
    unit = comparisons.get("full_rho_vs_unit_rho", {})
    ablated = comparisons.get("full_rho_vs_relation_gate_ablated", {})
    temporal = comparisons.get(
        "full_rho_vs_temporal_correspondence_permuted", {}
    )
    role = comparisons.get("full_rho_vs_left_right_role_swapped", {})
    checks = {
        "full_rho_minus_unit_rho_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                unit.get("full_rho_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "unit_rho_minus_full_rho_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                unit.get("other_minus_full_rho_gt_contact_distance_cm", {})
            )
        ),
        "full_rho_minus_gate_ablated_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                ablated.get("full_rho_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "full_rho_minus_temporal_permuted_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                temporal.get("full_rho_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "full_rho_minus_role_swapped_direct_left_right_macro_f1_ci_lower_gt_zero": (
            _positive_lower(
                role.get(
                    "full_rho_minus_other_direct_left_right_macro_5cm_f1", {}
                )
            )
        ),
        "gate_ablated_minus_full_rho_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                ablated.get("other_minus_full_rho_gt_contact_distance_cm", {})
            )
        ),
        "temporal_permuted_minus_full_rho_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                temporal.get("other_minus_full_rho_gt_contact_distance_cm", {})
            )
        ),
    }
    relation_path_used = bool(
        checks["full_rho_minus_gate_ablated_direct_union_5cm_f1_ci_lower_gt_zero"]
        and checks[
            "gate_ablated_minus_full_rho_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    schedule_reliability_passed = bool(
        checks["full_rho_minus_unit_rho_direct_union_5cm_f1_ci_lower_gt_zero"]
        and checks[
            "unit_rho_minus_full_rho_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    temporal_routing_passed = bool(
        checks[
            "full_rho_minus_temporal_permuted_direct_union_5cm_f1_ci_lower_gt_zero"
        ]
        and checks[
            "temporal_permuted_minus_full_rho_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    role_binding_passed = bool(
        checks[
            "full_rho_minus_role_swapped_direct_left_right_macro_f1_ci_lower_gt_zero"
        ]
    )
    mechanism_passed = bool(
        contract_passed
        and relation_path_used
        and schedule_reliability_passed
        and temporal_routing_passed
        and role_binding_passed
    )
    if not contract_passed:
        internal_status = "unused"
        classification = "diffusion-reliability-contract-failure-stop"
    elif not relation_path_used:
        internal_status = "unused"
        classification = "diffusion-reliability-internal-unused-continue-native"
    elif not schedule_reliability_passed:
        internal_status = "schedule-negative"
        classification = (
            "diffusion-reliability-internal-schedule-negative-continue-native"
        )
    elif not temporal_routing_passed:
        internal_status = "temporal-negative"
        classification = (
            "diffusion-reliability-internal-temporal-negative-continue-native"
        )
    elif not role_binding_passed:
        internal_status = "role-negative"
        classification = (
            "diffusion-reliability-internal-role-negative-continue-native"
        )
    else:
        internal_status = "passed"
        classification = "diffusion-reliability-internal-positive-continue-native"
    return {
        "classification": classification,
        "internal_status": internal_status,
        "contract_passed": contract_passed,
        "relation_path_used": relation_path_used,
        "schedule_reliability_passed": schedule_reliability_passed,
        "temporal_routing_passed": temporal_routing_passed,
        "role_binding_passed": role_binding_passed,
        "mechanism_passed": mechanism_passed,
        "checks": checks,
        "native_evaluation_authorized": contract_passed,
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
    """Apply D2-AE repair, D2-X candidate, protection, and release gates."""
    shared = _shared_native_gate(
        contract_passed=contract_passed,
        internal={
            "contract_passed": bool(internal.get("contract_passed", False)),
            # Internal mechanism outcomes are reported independently and do not
            # block the one registered native evaluation.
            "adapter_used": True,
            "locality_passed": True,
            "mechanism_passed": bool(internal.get("mechanism_passed", False)),
        },
        comparison=comparison,
        target_metrics=target_metrics,
        baseline_ratios=baseline_ratios,
    )

    target_f1 = float(target_metrics.get("contact_f1", float("nan")))
    candidate_checks = dict(shared["native_transfer_checks"])
    candidate_checks["contact_f1_point_estimate_ge_0.6598838781"] = bool(
        math.isfinite(target_f1) and target_f1 >= CONTACT_F1_POINT_MINIMUM
    )
    candidate_passed = all(bool(value) for value in candidate_checks.values())

    repair = comparison.get("target_vs_sealed_d2ae_repair", {})
    repair_ratios = (
        repair.get("target_over_control_protection", {})
        if isinstance(repair, Mapping)
        else {}
    )
    repair_checks = {
        "af_minus_ae_contact_f1_ci_lower_gt_zero": _positive_lower(
            repair.get("target_minus_control_contact_f1", {})
            if isinstance(repair, Mapping) else {}
        ),
        "af_minus_ae_contact_recall_ci_lower_gt_zero": _positive_lower(
            repair.get("target_minus_control_contact_recall", {})
            if isinstance(repair, Mapping) else {}
        ),
        "af_over_ae_end_object_ratio_ci_upper_lt_1.0": _ratio_upper(
            repair_ratios.get("end_obj_trans_err", {})
            if isinstance(repair_ratios, Mapping) else {},
            1.0,
            strict=True,
        ),
        "af_over_ae_foot_sliding_ratio_ci_upper_lt_1.0": _ratio_upper(
            repair_ratios.get("foot_sliding", {})
            if isinstance(repair_ratios, Mapping) else {},
            1.0,
            strict=True,
        ),
    }
    repair_passed = all(repair_checks.values())
    full_contract = bool(shared["contract_passed"])
    internal_mechanism_passed = bool(internal.get("mechanism_passed", False))
    if not full_contract:
        classification = "diffusion-reliability-contract-failure-stop"
        native_status = "contract-failure"
    elif not repair_passed:
        classification = "diffusion-reliability-ae-repair-negative-stop"
        native_status = "ae-repair-negative"
    elif not candidate_passed:
        classification = "diffusion-reliability-d2x-transfer-negative-stop"
        native_status = "d2x-transfer-negative"
    elif not bool(shared["protection_passed"]):
        classification = "diffusion-reliability-conflict-negative-stop"
        native_status = "conflict-negative"
    elif not bool(shared["released_95_percent_effectiveness_passed"]):
        classification = "diffusion-reliability-positive-but-not-effective-stop"
        native_status = "positive-but-not-effective"
    elif not internal_mechanism_passed:
        classification = (
            "diffusion-reliability-native-positive-mechanism-unverified-stop"
        )
        native_status = "native-positive-mechanism-unverified"
    else:
        classification = "diffusion-reliability-positive-candidate-stop"
        native_status = "positive-candidate"
    selectable = classification == "diffusion-reliability-positive-candidate-stop"

    value = dict(shared)
    value.pop("adapter_used", None)
    value.pop("locality_passed", None)
    value.pop("d2ac1_authorized", None)
    value.update({
        "classification": classification,
        "native_status": native_status,
        "internal_status": internal.get("internal_status"),
        "internal_mechanism_passed": internal_mechanism_passed,
        "relation_path_used": bool(internal.get("relation_path_used", False)),
        "schedule_reliability_passed": bool(
            internal.get("schedule_reliability_passed", False)
        ),
        "temporal_routing_passed": bool(
            internal.get("temporal_routing_passed", False)
        ),
        "role_binding_passed": bool(
            internal.get("role_binding_passed", False)
        ),
        "d2ae_repair_passed": repair_passed,
        "d2ae_repair_checks": repair_checks,
        "d2x_candidate_transfer_passed": candidate_passed,
        "native_transfer_passed": candidate_passed,
        "native_transfer_checks": candidate_checks,
        "contact_f1_point_estimate": target_f1,
        "contact_f1_point_minimum": CONTACT_F1_POINT_MINIMUM,
        "checkpoint_selected": selectable,
        "selectable_autonomous_diffusion_candidate": selectable,
        "consistency_authorized": False,
        "consistency_started": False,
        "d2af1_authorized": False,
        "hoiprior_search_closed": True,
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
