"""Locked statistics and gates for Phase 1B D2-AG0.

The module is deliberately independent of rollout, checkpoint, dataset, and
official-evaluator code.  It registers the six-path self-conditioned relation
source diagnostic, the five internal causal gates, the standard D2-X native
gates, and the two report-only quantities that must never reach a decision.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

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


REFERENCE_VARIANT = "full_self_conditioned"
VARIANTS: Tuple[str, ...] = (
    REFERENCE_VARIANT,
    "source_substituted_xt",
    "high_t_restricted",
    "object_displaced_counterfactual",
    "temporal_correspondence_permuted",
    "left_right_role_swapped",
)
ROLE_NAMES: Tuple[str, ...] = ("left_hand", "right_hand", "pelvis")
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
CONTACT_THRESHOLD_KEY = "0.5"
FIELD_ANCHOR_JOINTS: Tuple[int, int] = DIRECT_HAND_INDICES
OFFICIAL_FK_PALM_JOINTS: Tuple[int, int] = FK_PALM_INDICES
# Registered per-variant count of reverse steps whose relation source is
# bitwise equal to the current state.  The three counts named in the plan are
# ``full_self_conditioned`` = 1 (only t=499 lacks ``prev_x0``),
# ``source_substituted_xt`` = 500 and ``high_t_restricted`` = 250.
#
# ``object_displaced_counterfactual`` is 0, not 1: the registered delta is
# applied to the source at every reverse step, including the first one where the
# source starts as x_t, so no step is bitwise equal to the current state.  The
# temporal and role gates leave the source untouched and therefore match the
# reference count.
SOURCE_IS_CURRENT_STEP_COUNTS: Dict[str, int] = {
    REFERENCE_VARIANT: 1,
    "source_substituted_xt": 500,
    "high_t_restricted": 250,
    "object_displaced_counterfactual": 0,
    "temporal_correspondence_permuted": 1,
    "left_right_role_swapped": 1,
}


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
        raise ValueError("D2-AG direct-hand 5-cm report is malformed")
    left = float(_path(thresholds, ("left_hand", "f1")))
    right = float(_path(thresholds, ("right_hand", "f1")))
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("D2-AG direct-hand macro-F1 requires finite values")
    return 0.5 * (left + right)


def paired_comparisons(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    """Build the fixed paired sequence comparisons for all six paths."""
    if set(records) != set(VARIANTS):
        raise ValueError("D2-AG diagnostic variants differ from the locked six paths")
    names = [str(record["sequence"]) for record in records[REFERENCE_VARIANT]]
    if not names:
        raise ValueError("D2-AG paired comparison is empty")
    for variant in VARIANTS[1:]:
        if [str(record["sequence"]) for record in records[variant]] != names:
            raise ValueError("D2-AG paired sequence ordering differs")

    result: Dict[str, object] = {}
    for other in VARIANTS[1:]:
        comparison: Dict[str, object] = {
            "full_minus_other_direct_union_5cm_f1": paired_difference_fixed(
                _values(
                    records[REFERENCE_VARIANT],
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
                    [
                        _direct_left_right_macro_f1(row)
                        for row in records[REFERENCE_VARIANT]
                    ],
                    [_direct_left_right_macro_f1(row) for row in records[other]],
                )
            ),
            "other_minus_full_gt_contact_distance_cm": (
                paired_finite_difference(
                    _values(
                        records[other],
                        ("gt_contact_frame_direct_distance", "union", "mean_cm"),
                    ),
                    _values(
                        records[REFERENCE_VARIANT],
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
                        records[REFERENCE_VARIANT],
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
                                records[REFERENCE_VARIANT],
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
                                records[REFERENCE_VARIANT],
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
                _values(records[REFERENCE_VARIANT], ("kinematics", metric)),
                _values(records[other], ("kinematics", metric)),
            )

        penetration = comparison["penetration"]
        assert isinstance(penetration, dict)
        for metric in ("hand_pen_loss_omomo", "human_pen_loss_infbagel"):
            full = _values(records[REFERENCE_VARIANT], ("penetration", metric))
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


def _finite_mean(values: Sequence[object]) -> Optional[float]:
    kept = [
        float(value)
        for value in values
        if value is not None
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    return float(np.mean(kept)) if kept else None


def _unit_mask(values: np.ndarray, unit: str) -> np.ndarray:
    """Boolean per-frame contact for one unit, within a single sequence."""
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("D2-AG contact structure requires [frames,>=2] values")
    if unit == "union":
        return (values[:, :2] >= 0.5).any(axis=1)
    if unit not in CONTACT_UNITS:
        raise ValueError(f"unknown D2-AG contact unit: {unit}")
    return values[:, CONTACT_UNITS.index(unit)] >= 0.5


def per_sequence_contact_structure(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Report-only within-sequence contact run structure and coverage.

    One row per sequence and never a cross-sequence concatenation: gluing
    per-frame vectors across sequences merges a run that ends at a sequence
    boundary with the next sequence's opening run and inflates coverage toward
    the longest sequences.  Aggregates are therefore means over sequences.
    """
    rows: List[Dict[str, object]] = []
    for record in records:
        report = _path(
            record, ("semantic_vs_gt", "thresholds", CONTACT_THRESHOLD_KEY),
        )
        if not isinstance(report, Mapping):
            raise ValueError("D2-AG contact structure requires a semantic report")
        predicted = np.asarray(
            _path(record, ("per_frame", "predicted_contact")), dtype=np.float64,
        )
        target = np.asarray(
            _path(record, ("per_frame", "target_contact")), dtype=np.float64,
        )
        if predicted.shape != target.shape:
            raise ValueError("D2-AG contact structure shapes differ")
        row: Dict[str, object] = {
            "sequence": str(record["sequence"]),
            "frames": int(predicted.shape[0]),
            "units": {},
        }
        units = row["units"]
        assert isinstance(units, dict)
        for unit in CONTACT_UNITS:
            unit_report = report[unit]
            if not isinstance(unit_report, Mapping):
                raise ValueError(f"D2-AG contact unit report missing: {unit}")
            predicted_mask = _unit_mask(predicted, unit)
            target_mask = _unit_mask(target, unit)
            target_frames = int(target_mask.sum())
            units[unit] = {
                "prediction_run_lengths": unit_report["prediction_run_lengths"],
                "target_run_lengths": unit_report["target_run_lengths"],
                "prediction_coverage": float(predicted_mask.mean()),
                "target_coverage": float(target_mask.mean()),
                "coverage_ratio": (
                    float(predicted_mask.mean() / target_mask.mean())
                    if target_frames else None
                ),
                "covered_target_frames_fraction": (
                    float((predicted_mask & target_mask).sum() / target_frames)
                    if target_frames else None
                ),
            }
        rows.append(row)
    return {
        "reported_only": True,
        "paired_unit": "sequence",
        "concatenated_across_sequences": False,
        "threshold": CONTACT_THRESHOLD_KEY,
        "sequences": len(rows),
        "per_sequence": rows,
        "mean_over_sequences": {
            unit: {
                key: _finite_mean([row["units"][unit][key] for row in rows])
                for key in (
                    "prediction_coverage",
                    "target_coverage",
                    "coverage_ratio",
                    "covered_target_frames_fraction",
                )
            }
            for unit in CONTACT_UNITS
        },
        "run_structure_mean_over_sequences": {
            unit: {
                f"{side}_{key}": _finite_mean([
                    row["units"][unit][f"{side}_run_lengths"][key]
                    for row in rows
                ])
                for side in ("prediction", "target")
                for key in ("runs", "mean_frames", "max_frames")
            }
            for unit in CONTACT_UNITS
        },
    }


def anchor_vs_fk_palm_correspondence(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Report-only paired gap between field anchors 24/26 and FK palms 22/23."""
    direct: List[float] = []
    fk: List[float] = []
    names: List[str] = []
    for record in records:
        direct_report = _path(
            record, ("direct_physical_geometry_vs_gt", "thresholds_cm", "5"),
        )
        fk_report = _path(
            record, ("fk_physical_geometry_vs_gt", "thresholds_cm", "5"),
        )
        if not isinstance(direct_report, Mapping) or not isinstance(
            fk_report, Mapping
        ):
            raise ValueError("D2-AG anchor/palm correspondence report is malformed")
        direct.append(
            0.5 * (
                float(_path(direct_report, ("left_hand", "f1")))
                + float(_path(direct_report, ("right_hand", "f1")))
            )
        )
        fk.append(
            0.5 * (
                float(_path(fk_report, ("left_hand", "f1")))
                + float(_path(fk_report, ("right_hand", "f1")))
            )
        )
        names.append(str(record["sequence"]))
    return {
        "reported_only": True,
        "field_anchor_joints": list(FIELD_ANCHOR_JOINTS),
        "official_fk_palm_joints": list(OFFICIAL_FK_PALM_JOINTS),
        "paired_unit": "sequence",
        "sequences": len(names),
        "sequence_names": names,
        "direct_minus_fk_macro_5cm_f1": paired_difference_fixed(direct, fk),
        "note": (
            "reported only; the relation field anchors on direct joints 24/26 "
            "while the official evaluator scores FK palms 22/23, so an "
            "internally causal path can stay invisible to the native metric"
        ),
    }


def reported_only_quantities(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Bundle both report-only quantities for one variant."""
    return {
        "reported_only": True,
        "enters_internal_mechanism_gate": False,
        "enters_native_gate": False,
        "per_sequence_contact_structure": per_sequence_contact_structure(records),
        "anchor_vs_fk_palm_correspondence": anchor_vs_fk_palm_correspondence(
            records
        ),
    }


def internal_mechanism_gate(
    contract: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Apply the five registered D2-AG internal gates (EP:7264-7273).

    The two report-only quantities are deliberately not referenced here.
    """
    contract_passed = bool(contract) and all(bool(value) for value in contract.values())
    substituted = comparisons.get("full_vs_source_substituted_xt", {})
    high_t = comparisons.get("full_vs_high_t_restricted", {})
    displaced = comparisons.get("full_vs_object_displaced_counterfactual", {})
    temporal = comparisons.get("full_vs_temporal_correspondence_permuted", {})
    role = comparisons.get("full_vs_left_right_role_swapped", {})
    checks = {
        "full_minus_source_substituted_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                substituted.get("full_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "source_substituted_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                substituted.get("other_minus_full_gt_contact_distance_cm", {})
            )
        ),
        "full_minus_high_t_restricted_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                high_t.get("full_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "high_t_restricted_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                high_t.get("other_minus_full_gt_contact_distance_cm", {})
            )
        ),
        "object_displaced_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                displaced.get("other_minus_full_gt_contact_distance_cm", {})
            )
        ),
        "full_minus_object_displaced_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                displaced.get("full_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "full_minus_temporal_permuted_direct_union_5cm_f1_ci_lower_gt_zero": (
            _positive_lower(
                temporal.get("full_minus_other_direct_union_5cm_f1", {})
            )
        ),
        "temporal_permuted_minus_full_gt_contact_distance_ci_lower_gt_zero": (
            _positive_lower(
                temporal.get("other_minus_full_gt_contact_distance_cm", {})
            )
        ),
        "full_minus_role_swapped_direct_left_right_macro_f1_ci_lower_gt_zero": (
            _positive_lower(
                role.get("full_minus_other_direct_left_right_macro_5cm_f1", {})
            )
        ),
    }
    source_provenance_passed = bool(
        checks[
            "full_minus_source_substituted_direct_union_5cm_f1_ci_lower_gt_zero"
        ]
        and checks[
            "source_substituted_minus_full_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    high_t_provenance_passed = bool(
        checks[
            "full_minus_high_t_restricted_direct_union_5cm_f1_ci_lower_gt_zero"
        ]
        and checks[
            "high_t_restricted_minus_full_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    object_following_passed = bool(
        checks["object_displaced_minus_full_gt_contact_distance_ci_lower_gt_zero"]
        and checks[
            "full_minus_object_displaced_direct_union_5cm_f1_ci_lower_gt_zero"
        ]
    )
    temporal_routing_passed = bool(
        checks["full_minus_temporal_permuted_direct_union_5cm_f1_ci_lower_gt_zero"]
        and checks[
            "temporal_permuted_minus_full_gt_contact_distance_ci_lower_gt_zero"
        ]
    )
    role_binding_passed = bool(
        checks[
            "full_minus_role_swapped_direct_left_right_macro_f1_ci_lower_gt_zero"
        ]
    )
    mechanism_passed = bool(
        contract_passed
        and source_provenance_passed
        and high_t_provenance_passed
        and object_following_passed
        and temporal_routing_passed
        and role_binding_passed
    )
    if not contract_passed:
        internal_status = "unused"
        classification = "selfcond-relation-source-contract-failure-stop"
    elif not source_provenance_passed:
        internal_status = "source-provenance-negative"
        classification = (
            "selfcond-relation-source-internal-source-negative-continue-native"
        )
    elif not high_t_provenance_passed:
        internal_status = "high-t-negative"
        classification = (
            "selfcond-relation-source-internal-high-t-negative-continue-native"
        )
    elif not object_following_passed:
        internal_status = "object-following-negative"
        classification = (
            "selfcond-relation-source-internal-object-negative-continue-native"
        )
    elif not temporal_routing_passed:
        internal_status = "temporal-negative"
        classification = (
            "selfcond-relation-source-internal-temporal-negative-continue-native"
        )
    elif not role_binding_passed:
        internal_status = "role-negative"
        classification = (
            "selfcond-relation-source-internal-role-negative-continue-native"
        )
    else:
        internal_status = "passed"
        classification = (
            "selfcond-relation-source-internal-positive-continue-native"
        )
    return {
        "classification": classification,
        "internal_status": internal_status,
        "contract_passed": contract_passed,
        "source_provenance_passed": source_provenance_passed,
        "high_t_provenance_passed": high_t_provenance_passed,
        "object_following_passed": object_following_passed,
        "temporal_routing_passed": temporal_routing_passed,
        "role_binding_passed": role_binding_passed,
        "mechanism_passed": mechanism_passed,
        "checks": checks,
        # The one registered native evaluation runs regardless of the internal
        # outcome; only a contract failure blocks it.
        "native_evaluation_authorized": contract_passed,
        "checkpoint_selected": False,
        "consistency_authorized": False,
        "reported_only_quantities_used": False,
    }


def native_gate(
    *,
    contract_passed: bool,
    internal: Mapping[str, object],
    comparison: Mapping[str, object],
    target_metrics: Mapping[str, object],
    baseline_ratios: Mapping[str, float],
) -> Dict[str, object]:
    """Apply the standard D2-X transfer, protection, and release gates.

    D2-AG registers no predecessor-specific repair gate (EP:7169-7182), so the
    only addition to the shared gate is the registered contact-F1 point floor.
    """
    shared = _shared_native_gate(
        contract_passed=contract_passed,
        internal={
            "contract_passed": bool(internal.get("contract_passed", False)),
            # Internal mechanism outcomes are reported independently and never
            # block or replace the one registered native evaluation.
            "adapter_used": True,
            "locality_passed": True,
            "mechanism_passed": bool(internal.get("mechanism_passed", False)),
        },
        comparison=comparison,
        target_metrics=target_metrics,
        baseline_ratios=baseline_ratios,
    )

    target_f1 = float(target_metrics.get("contact_f1", float("nan")))
    transfer_checks = dict(shared["native_transfer_checks"])
    transfer_checks["contact_f1_point_estimate_ge_0.6598838781"] = bool(
        math.isfinite(target_f1) and target_f1 >= CONTACT_F1_POINT_MINIMUM
    )
    transfer_passed = all(bool(value) for value in transfer_checks.values())
    full_contract = bool(shared["contract_passed"])
    internal_mechanism_passed = bool(internal.get("mechanism_passed", False))
    protection_passed = bool(shared["protection_passed"])
    released_effective = bool(shared["released_95_percent_effectiveness_passed"])
    if not full_contract:
        classification = "selfcond-relation-source-contract-failure-stop"
        native_status = "contract-failure"
    elif not transfer_passed:
        classification = "selfcond-relation-source-transfer-negative-stop"
        native_status = "transfer-negative"
    elif not protection_passed or not released_effective:
        classification = "selfcond-relation-source-conflict-negative-stop"
        native_status = "conflict-negative"
    elif not internal_mechanism_passed:
        classification = (
            "selfcond-relation-source-native-positive-mechanism-unverified-stop"
        )
        native_status = "native-positive-mechanism-unverified"
    else:
        classification = "selfcond-relation-source-positive-candidate-stop"
        native_status = "positive-candidate"
    # The internal label is recorded beside the native headline; a negative
    # internal outcome never rescues or replaces a negative native result.
    mechanism_negative_label = (
        "selfcond-relation-source-mechanism-negative-stop"
        if not internal_mechanism_passed
        and native_status in {"contract-failure", "transfer-negative", "conflict-negative"}
        else None
    )
    selectable = (
        classification == "selfcond-relation-source-positive-candidate-stop"
    )

    value = dict(shared)
    value.pop("adapter_used", None)
    value.pop("locality_passed", None)
    value.pop("d2ac1_authorized", None)
    value.update({
        "classification": classification,
        "native_status": native_status,
        "internal_status": internal.get("internal_status"),
        "internal_mechanism_passed": internal_mechanism_passed,
        "internal_mechanism_negative_classification": mechanism_negative_label,
        "source_provenance_passed": bool(
            internal.get("source_provenance_passed", False)
        ),
        "high_t_provenance_passed": bool(
            internal.get("high_t_provenance_passed", False)
        ),
        "object_following_passed": bool(
            internal.get("object_following_passed", False)
        ),
        "temporal_routing_passed": bool(
            internal.get("temporal_routing_passed", False)
        ),
        "role_binding_passed": bool(internal.get("role_binding_passed", False)),
        "native_transfer_passed": transfer_passed,
        "native_transfer_checks": transfer_checks,
        "contact_f1_point_estimate": target_f1,
        "contact_f1_point_minimum": CONTACT_F1_POINT_MINIMUM,
        "predecessor_specific_repair_gate": False,
        "reported_only_quantities_used": False,
        "checkpoint_selected": selectable,
        "selectable_autonomous_diffusion_candidate": selectable,
        "consistency_authorized": False,
        "consistency_started": False,
        "d2ag1_authorized": False,
        "hoiprior_search_closed": True,
    })
    return value


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONTACT_F1_POINT_MINIMUM",
    "CONTACT_THRESHOLD_KEY",
    "CONTACT_UNITS",
    "DIRECT_HAND_INDICES",
    "FIELD_ANCHOR_JOINTS",
    "FK_PALM_INDICES",
    "GT_CONTACT_FINITE_SEQUENCE_COUNT",
    "GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256",
    "HISTORY_MAX_ABS",
    "OFFICIAL_FK_PALM_JOINTS",
    "PHYSICAL_THRESHOLDS_CM",
    "PROTECTION_METRICS",
    "REFERENCE_VARIANT",
    "RELEASED_HIGHER_IS_BETTER",
    "RELEASED_LOWER_IS_BETTER",
    "ROLE_NAMES",
    "SELECTION_SHA256",
    "SOURCE_IS_CURRENT_STEP_COUNTS",
    "TEMPORAL_ANCHORS",
    "VARIANTS",
    "anchor_vs_fk_palm_correspondence",
    "internal_mechanism_gate",
    "native_gate",
    "paired_comparisons",
    "per_sequence_contact_structure",
    "reported_only_quantities",
]
