from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.contact_alignment import contact_run_lengths  # noqa: E402
from priors import d2af_diagnostic as d2af  # noqa: E402
from priors import d2ag_diagnostic as d2ag  # noqa: E402
from priors.interaction_diagnostic import (  # noqa: E402
    PROTECTION_METRICS,
    RELEASED_HIGHER_IS_BETTER,
    RELEASED_LOWER_IS_BETTER,
)
from priors.sparse_relation import (  # noqa: E402
    D2AG_DIAGNOSTIC_VARIANTS,
    D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
)
from tools import run_hoi_d2ag_internal as internal_runner  # noqa: E402
from tools import run_hoi_d2ag_native_evaluation as native  # noqa: E402


def make_record(
    name: str,
    *,
    union_f1: float,
    left_f1: float,
    right_f1: float,
    gt_distance_cm: float | None,
    kinematic: float,
    penetration: float | None,
    fk_left_f1: float = 0.1,
    fk_right_f1: float = 0.2,
    frames: int = 6,
) -> dict:
    predicted = np.zeros((frames, 4), dtype=np.float64)
    target = np.zeros((frames, 4), dtype=np.float64)
    predicted[: frames // 2, :2] = 1.0
    target[1: frames // 2 + 1, :2] = 1.0
    semantic_units = {}
    for unit in d2ag.CONTACT_UNITS:
        predicted_mask = d2ag._unit_mask(predicted, unit)
        target_mask = d2ag._unit_mask(target, unit)
        semantic_units[unit] = {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "prediction_percent": float(predicted_mask.mean()),
            "prediction_run_lengths": contact_run_lengths(predicted_mask),
            "target_run_lengths": contact_run_lengths(target_mask),
        }

    def geometry(left: float, right: float, union: float) -> dict:
        thresholds = {}
        for threshold in d2ag.PHYSICAL_THRESHOLDS_CM:
            key = f"{threshold:g}"
            thresholds[key] = {
                unit: {
                    "precision": value,
                    "recall": value,
                    "f1": value,
                    "prediction_percent": value,
                    "prediction_run_lengths": {
                        "runs": 1,
                        "mean_frames": 2.0,
                        "max_frames": 2,
                        "lengths": [2],
                    },
                }
                for unit, value in (
                    ("left_hand", left),
                    ("right_hand", right),
                    ("union", union),
                )
            }
        return {"thresholds_cm": thresholds}

    return {
        "sequence": name,
        "sequence_index": 0,
        "object_category": "box",
        "positions": [0, 42, 84],
        "pi": [14, 56, 98],
        "semantic_vs_gt": {"thresholds": {"0.5": semantic_units}},
        "direct_physical_geometry_vs_gt": geometry(left_f1, right_f1, union_f1),
        "fk_physical_geometry_vs_gt": geometry(fk_left_f1, fk_right_f1, 0.15),
        "gt_contact_frame_direct_distance": {
            "union": {"mean_cm": gt_distance_cm, "finite": gt_distance_cm is not None},
        },
        "kinematics": {metric: kinematic for metric in d2ag.KINEMATIC_METRICS},
        "penetration": {
            "hand_pen_loss_omomo": penetration,
            "human_pen_loss_infbagel": penetration,
        },
        "per_frame": {
            "predicted_contact": predicted.tolist(),
            "target_contact": target.tolist(),
        },
    }


def variant_records(*, better: bool) -> list:
    """Six sequences per variant; ``better`` shifts the reference path up."""
    offset = 0.2 if better else 0.0
    return [
        make_record(
            f"seq{index:02d}",
            union_f1=0.4 + offset + 0.01 * index,
            left_f1=0.3 + offset + 0.01 * index,
            right_f1=0.5 + offset + 0.01 * index,
            gt_distance_cm=(6.0 - offset * 10.0 + 0.1 * index),
            kinematic=1.0 - offset + 0.01 * index,
            penetration=0.5 + 0.01 * index,
        )
        for index in range(6)
    ]


def all_variant_records() -> dict:
    return {
        variant: variant_records(better=variant == d2ag.REFERENCE_VARIANT)
        for variant in d2ag.VARIANTS
    }


class D2AGRegisteredVariantTests(unittest.TestCase):
    def test_six_variants_in_registered_order_without_gate_ablation(self):
        self.assertEqual(len(d2ag.VARIANTS), 6)
        self.assertEqual(d2ag.REFERENCE_VARIANT, "full_self_conditioned")
        self.assertEqual(d2ag.VARIANTS[0], d2ag.REFERENCE_VARIANT)
        self.assertEqual(
            d2ag.VARIANTS[1:],
            (
                "source_substituted_xt",
                "high_t_restricted",
                "object_displaced_counterfactual",
                "temporal_correspondence_permuted",
                "left_right_role_swapped",
            ),
        )
        self.assertNotIn("relation_gate_ablated", d2ag.VARIANTS)
        self.assertNotIn("unit_rho", d2ag.VARIANTS)

    def test_every_diagnostic_variant_is_supported_by_the_field(self):
        for variant in d2ag.VARIANTS:
            expected = (
                "full" if variant == d2ag.REFERENCE_VARIANT else variant
            )
            self.assertIn(expected, D2AG_DIAGNOSTIC_VARIANTS)

    def test_registered_source_is_x_t_counts_are_exact(self):
        self.assertEqual(
            d2ag.SOURCE_IS_CURRENT_STEP_COUNTS,
            {
                "full_self_conditioned": 1,
                "source_substituted_xt": 500,
                "high_t_restricted": 250,
                "object_displaced_counterfactual": 0,
                "temporal_correspondence_permuted": 1,
                "left_right_role_swapped": 1,
            },
        )
        self.assertEqual(D2AG_HIGH_T_SELF_CONDITION_CUTOFF, 250)
        self.assertEqual(
            d2ag.SOURCE_IS_CURRENT_STEP_COUNTS["high_t_restricted"],
            D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
        )

    def test_registered_first_step_source_labels(self):
        for variant in d2ag.VARIANTS:
            expected = (
                internal_runner.SOURCE_LABEL_PREVIOUS
                if variant == "object_displaced_counterfactual"
                else internal_runner.SOURCE_LABEL_CURRENT
            )
            self.assertEqual(
                internal_runner.expected_first_step_source(variant), expected,
            )

    def test_anchor_and_probability_constants_are_registered(self):
        self.assertEqual(d2ag.FIELD_ANCHOR_JOINTS, (24, 26))
        self.assertEqual(d2ag.OFFICIAL_FK_PALM_JOINTS, (22, 23))
        self.assertNotEqual(d2ag.FIELD_ANCHOR_JOINTS, d2ag.OFFICIAL_FK_PALM_JOINTS)
        self.assertEqual(D2AG_SELF_CONDITION_PROBABILITY, 0.5)
        self.assertEqual(D2AG_VARIABLE_ANCHORS, (5, 10, 15))


class D2AGPairedComparisonTests(unittest.TestCase):
    def test_comparison_keys_cover_all_five_alternates(self):
        comparisons = d2ag.paired_comparisons(all_variant_records())
        self.assertEqual(
            sorted(comparisons),
            sorted(f"full_vs_{variant}" for variant in d2ag.VARIANTS[1:]),
        )
        for value in comparisons.values():
            self.assertEqual(
                sorted(value),
                sorted((
                    "full_minus_other_direct_union_5cm_f1",
                    "full_minus_other_direct_left_right_macro_5cm_f1",
                    "other_minus_full_gt_contact_distance_cm",
                    "semantic_contact",
                    "physical_contact",
                    "kinematics",
                    "penetration",
                )),
            )

    def test_missing_or_extra_variant_is_rejected(self):
        records = all_variant_records()
        del records["high_t_restricted"]
        with self.assertRaises(ValueError):
            d2ag.paired_comparisons(records)
        records = all_variant_records()
        records["relation_gate_ablated"] = variant_records(better=False)
        with self.assertRaises(ValueError):
            d2ag.paired_comparisons(records)

    def test_sequence_order_mismatch_is_rejected(self):
        records = all_variant_records()
        records["source_substituted_xt"] = list(
            reversed(records["source_substituted_xt"])
        )
        with self.assertRaises(ValueError):
            d2ag.paired_comparisons(records)

    def test_paired_unit_is_sequence_and_bootstrap_is_registered(self):
        comparisons = d2ag.paired_comparisons(all_variant_records())
        value = comparisons["full_vs_source_substituted_xt"][
            "full_minus_other_direct_union_5cm_f1"
        ]
        self.assertEqual(value["bootstrap_replicates"], 10_000)
        self.assertEqual(value["bootstrap_seed"], 42)
        self.assertEqual(len(value["per_unit"]["first"]), 6)

    def test_gt_contact_distance_uses_the_finite_mask_without_imputation(self):
        records = all_variant_records()
        for variant in d2ag.VARIANTS:
            records[variant][2]["gt_contact_frame_direct_distance"]["union"] = {
                "mean_cm": None, "finite": False,
            }
        comparisons = d2ag.paired_comparisons(records)
        value = comparisons["full_vs_high_t_restricted"][
            "other_minus_full_gt_contact_distance_cm"
        ]
        self.assertEqual(value["finite_sequence_count"], 5)
        self.assertNotIn("seq02", value["finite_sequence_names"])

    def test_penetration_zero_denominator_is_explicit(self):
        records = all_variant_records()
        for variant in d2ag.VARIANTS:
            for record in records[variant]:
                record["penetration"] = {
                    "hand_pen_loss_omomo": 0.0,
                    "human_pen_loss_infbagel": 0.0,
                }
        comparisons = d2ag.paired_comparisons(records)
        value = comparisons["full_vs_left_right_role_swapped"]["penetration"][
            "hand_pen_loss_omomo"
        ]
        self.assertFalse(value["ratio_defined"])
        self.assertEqual(value["undefined_reason"], "zero_denominator_mean")
        self.assertIn("paired_difference", value)


class D2AGInternalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.comparisons = d2ag.paired_comparisons(all_variant_records())
        self.contract = {"everything": True}

    def test_nine_checks_and_five_named_outcomes(self):
        decision = d2ag.internal_mechanism_gate(self.contract, self.comparisons)
        self.assertEqual(len(decision["checks"]), 9)
        for name in (
            "source_provenance_passed",
            "high_t_provenance_passed",
            "object_following_passed",
            "temporal_routing_passed",
            "role_binding_passed",
        ):
            self.assertIn(name, decision)

    def test_all_five_gates_pass_on_a_uniformly_better_reference(self):
        decision = d2ag.internal_mechanism_gate(self.contract, self.comparisons)
        self.assertTrue(decision["mechanism_passed"], decision["checks"])
        self.assertEqual(decision["internal_status"], "passed")
        self.assertEqual(
            decision["classification"],
            "selfcond-relation-source-internal-positive-continue-native",
        )

    def test_contract_failure_stops_and_blocks_native(self):
        decision = d2ag.internal_mechanism_gate(
            {"ok": True, "broken": False}, self.comparisons,
        )
        self.assertFalse(decision["contract_passed"])
        self.assertFalse(decision["native_evaluation_authorized"])
        self.assertEqual(
            decision["classification"],
            "selfcond-relation-source-contract-failure-stop",
        )

    def test_each_negative_gate_produces_its_registered_classification(self):
        cases = {
            "source_substituted_xt": (
                "source-provenance-negative",
                "selfcond-relation-source-internal-source-negative-"
                "continue-native",
            ),
            "high_t_restricted": (
                "high-t-negative",
                "selfcond-relation-source-internal-high-t-negative-"
                "continue-native",
            ),
            "object_displaced_counterfactual": (
                "object-following-negative",
                "selfcond-relation-source-internal-object-negative-"
                "continue-native",
            ),
            "temporal_correspondence_permuted": (
                "temporal-negative",
                "selfcond-relation-source-internal-temporal-negative-"
                "continue-native",
            ),
            "left_right_role_swapped": (
                "role-negative",
                "selfcond-relation-source-internal-role-negative-"
                "continue-native",
            ),
        }
        for variant, (status, classification) in cases.items():
            records = all_variant_records()
            # Make one alternate as good as the reference so only its gate fails.
            records[variant] = variant_records(better=True)
            comparisons = d2ag.paired_comparisons(records)
            decision = d2ag.internal_mechanism_gate(self.contract, comparisons)
            self.assertFalse(decision["mechanism_passed"], variant)
            self.assertEqual(decision["internal_status"], status, variant)
            self.assertEqual(decision["classification"], classification, variant)
            # A negative internal outcome never blocks the one native run.
            self.assertTrue(decision["native_evaluation_authorized"], variant)

    def test_gate_never_consults_reported_only_quantities(self):
        decision = d2ag.internal_mechanism_gate(self.contract, self.comparisons)
        self.assertFalse(decision["reported_only_quantities_used"])
        records = all_variant_records()
        for variant in d2ag.VARIANTS:
            for record in records[variant]:
                # Destroy the report-only inputs only.
                record["fk_physical_geometry_vs_gt"]["thresholds_cm"]["5"][
                    "left_hand"
                ]["f1"] = 0.99
                record["per_frame"]["predicted_contact"] = [
                    [1.0, 1.0, 0.0, 0.0]
                ] * 6
        mutated = d2ag.internal_mechanism_gate(
            self.contract, d2ag.paired_comparisons(records),
        )
        self.assertEqual(mutated["checks"], decision["checks"])
        self.assertEqual(
            mutated["classification"], decision["classification"],
        )


def positive_statistic(mean: float) -> dict:
    return {
        "first_mean": 0.70,
        "second_mean": 0.70 - mean,
        "paired_mean_first_minus_second": mean,
        "bootstrap_95_ci": [mean * 0.5, mean * 1.5],
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
    }


def ratio_statistic(upper: float) -> dict:
    return {
        "numerator_mean": 1.0,
        "denominator_mean": 1.0,
        "mean_ratio": upper * 0.9,
        "bootstrap_95_ci": [upper * 0.5, upper],
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
    }


def native_inputs(
    *,
    contact_f1: float = 0.70,
    gap: float = 0.40,
    protection_upper: float = 1.05,
    precision_lower: float = 0.0,
    released_ratio: float = 1.0,
) -> dict:
    comparison = {
        "penetration_mask_contract": {"passed": True},
        "target_minus_control_contact_f1": positive_statistic(0.05),
        "target_minus_control_contact_recall": positive_statistic(0.05),
        "target_minus_control_contact_precision": {
            "bootstrap_95_ci": [precision_lower, precision_lower + 0.05],
        },
        "contact_f1_released_gap_closure": gap,
        "target_over_control_protection": {
            metric: ratio_statistic(protection_upper)
            for metric in PROTECTION_METRICS
        },
    }
    target_metrics = {
        metric: 1.0
        for metric in set(RELEASED_LOWER_IS_BETTER) | set(RELEASED_HIGHER_IS_BETTER)
    }
    target_metrics["contact_f1"] = contact_f1
    baseline_ratios = {
        metric: released_ratio
        for metric in set(RELEASED_LOWER_IS_BETTER) | set(RELEASED_HIGHER_IS_BETTER)
    }
    return {
        "comparison": comparison,
        "target_metrics": target_metrics,
        "baseline_ratios": baseline_ratios,
    }


def internal_state(*, mechanism_passed: bool, contract_passed: bool = True) -> dict:
    return {
        "contract_passed": contract_passed,
        "mechanism_passed": mechanism_passed,
        "internal_status": "passed" if mechanism_passed else "temporal-negative",
        "source_provenance_passed": mechanism_passed,
        "high_t_provenance_passed": mechanism_passed,
        "object_following_passed": mechanism_passed,
        "temporal_routing_passed": mechanism_passed,
        "role_binding_passed": mechanism_passed,
    }


class D2AGNativeGateTests(unittest.TestCase):
    def test_positive_candidate_requires_native_and_internal(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(),
        )
        self.assertEqual(
            value["classification"],
            "selfcond-relation-source-positive-candidate-stop",
        )
        self.assertEqual(value["native_status"], "positive-candidate")
        self.assertTrue(value["checkpoint_selected"])
        self.assertTrue(value["selectable_autonomous_diffusion_candidate"])

    def test_native_positive_with_internal_negative_is_not_selectable(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=False),
            **native_inputs(),
        )
        self.assertEqual(
            value["classification"],
            "selfcond-relation-source-native-positive-mechanism-unverified-stop",
        )
        self.assertFalse(value["checkpoint_selected"])
        self.assertFalse(value["internal_mechanism_passed"])

    def test_registered_contact_f1_point_floor_is_binding(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(contact_f1=0.65),
        )
        self.assertEqual(d2ag.CONTACT_F1_POINT_MINIMUM, 0.6598838781)
        self.assertFalse(
            value["native_transfer_checks"][
                "contact_f1_point_estimate_ge_0.6598838781"
            ]
        )
        self.assertEqual(
            value["classification"],
            "selfcond-relation-source-transfer-negative-stop",
        )

    def test_gap_closure_floor_is_binding(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(gap=0.10),
        )
        self.assertFalse(
            value["native_transfer_checks"][
                "contact_f1_released_gap_closure_ge_0.25"
            ]
        )
        self.assertEqual(value["native_status"], "transfer-negative")

    def test_protection_ceiling_and_precision_floor_are_binding(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(protection_upper=1.20),
        )
        self.assertEqual(
            value["classification"],
            "selfcond-relation-source-conflict-negative-stop",
        )
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(precision_lower=-0.10),
        )
        self.assertFalse(
            value["protection_checks"]["contact_precision_ci_lower_ge_minus_0.02"]
        )
        self.assertEqual(value["native_status"], "conflict-negative")

    def test_released_effectiveness_floor_is_binding(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(released_ratio=0.80),
        )
        self.assertFalse(value["released_95_percent_effectiveness_passed"])
        self.assertEqual(value["native_status"], "conflict-negative")

    def test_contract_failure_takes_precedence(self):
        value = d2ag.native_gate(
            contract_passed=False,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(),
        )
        self.assertEqual(
            value["classification"],
            "selfcond-relation-source-contract-failure-stop",
        )
        self.assertFalse(value["checkpoint_selected"])

    def test_internal_negative_never_rescues_a_negative_native_result(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=False),
            **native_inputs(contact_f1=0.10),
        )
        self.assertEqual(value["native_status"], "transfer-negative")
        self.assertEqual(
            value["classification"],
            "selfcond-relation-source-transfer-negative-stop",
        )
        # The internal label is recorded beside the native headline, not instead.
        self.assertEqual(
            value["internal_mechanism_negative_classification"],
            "selfcond-relation-source-mechanism-negative-stop",
        )
        self.assertFalse(value["checkpoint_selected"])

    def test_no_predecessor_repair_gate_is_registered(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(),
        )
        self.assertFalse(value["predecessor_specific_repair_gate"])
        self.assertNotIn("d2ae_repair_passed", value)
        self.assertNotIn("d2ae_repair_checks", value)
        self.assertNotIn("target_vs_sealed_d2ae_repair", value)
        # The D2-AF module does register one; D2-AG deliberately does not.
        self.assertIn("af_minus_ae_contact_f1_ci_lower_gt_zero", set(
            d2af.native_gate(
                contract_passed=True,
                internal=internal_state(mechanism_passed=True),
                **native_inputs(),
            )["d2ae_repair_checks"]
        ))

    def test_downstream_stages_stay_closed(self):
        value = d2ag.native_gate(
            contract_passed=True,
            internal=internal_state(mechanism_passed=True),
            **native_inputs(),
        )
        self.assertFalse(value["consistency_authorized"])
        self.assertFalse(value["consistency_started"])
        self.assertFalse(value["d2ag1_authorized"])
        self.assertTrue(value["hoiprior_search_closed"])
        self.assertFalse(value["reported_only_quantities_used"])


class D2AGReportedOnlyQuantityTests(unittest.TestCase):
    def test_contact_structure_is_per_sequence_and_never_concatenated(self):
        # Sequence A ends in contact and sequence B opens in contact, so a naive
        # cross-sequence concatenation would merge two runs into one.
        first = make_record(
            "a", union_f1=0.4, left_f1=0.3, right_f1=0.5,
            gt_distance_cm=5.0, kinematic=1.0, penetration=0.1, frames=4,
        )
        second = make_record(
            "b", union_f1=0.4, left_f1=0.3, right_f1=0.5,
            gt_distance_cm=5.0, kinematic=1.0, penetration=0.1, frames=8,
        )
        value = d2ag.per_sequence_contact_structure([first, second])
        self.assertTrue(value["reported_only"])
        self.assertFalse(value["concatenated_across_sequences"])
        self.assertEqual(value["paired_unit"], "sequence")
        self.assertEqual(value["sequences"], 2)
        self.assertEqual(len(value["per_sequence"]), 2)
        runs = [
            row["units"]["union"]["prediction_run_lengths"]["runs"]
            for row in value["per_sequence"]
        ]
        self.assertEqual(runs, [1, 1])
        self.assertEqual(
            value["run_structure_mean_over_sequences"]["union"][
                "prediction_runs"
            ],
            1.0,
        )

    def test_coverage_is_a_sequence_mean_not_a_frame_weighted_mean(self):
        short = make_record(
            "short", union_f1=0.4, left_f1=0.3, right_f1=0.5,
            gt_distance_cm=5.0, kinematic=1.0, penetration=0.1, frames=4,
        )
        long = make_record(
            "long", union_f1=0.4, left_f1=0.3, right_f1=0.5,
            gt_distance_cm=5.0, kinematic=1.0, penetration=0.1, frames=10,
        )
        # Asymmetric coverage is required for the two means to differ at all:
        # with equal per-sequence coverage they coincide and prove nothing.
        short["per_frame"]["predicted_contact"] = [
            [1.0, 1.0, 0.0, 0.0]] * 3 + [[0.0] * 4]
        long["per_frame"]["predicted_contact"] = [
            [1.0, 1.0, 0.0, 0.0]] * 2 + [[0.0] * 4] * 8
        value = d2ag.per_sequence_contact_structure([short, long])
        per_sequence = [
            row["units"]["union"]["prediction_coverage"]
            for row in value["per_sequence"]
        ]
        observed = value["mean_over_sequences"]["union"]["prediction_coverage"]
        self.assertAlmostEqual(observed, float(np.mean(per_sequence)))
        frame_weighted = (
            sum(
                row["units"]["union"]["prediction_coverage"] * row["frames"]
                for row in value["per_sequence"]
            )
            / sum(row["frames"] for row in value["per_sequence"])
        )
        self.assertNotAlmostEqual(observed, frame_weighted)

    def test_absent_target_contact_yields_none_not_a_fabricated_ratio(self):
        record = make_record(
            "empty", union_f1=0.4, left_f1=0.3, right_f1=0.5,
            gt_distance_cm=5.0, kinematic=1.0, penetration=0.1, frames=6,
        )
        record["per_frame"]["target_contact"] = [[0.0] * 4] * 6
        value = d2ag.per_sequence_contact_structure([record])
        unit = value["per_sequence"][0]["units"]["union"]
        self.assertEqual(unit["target_coverage"], 0.0)
        self.assertIsNone(unit["coverage_ratio"])
        self.assertIsNone(unit["covered_target_frames_fraction"])

    def test_anchor_versus_fk_palm_comparison_is_paired_and_reported_only(self):
        records = variant_records(better=False)
        value = d2ag.anchor_vs_fk_palm_correspondence(records)
        self.assertTrue(value["reported_only"])
        self.assertEqual(value["field_anchor_joints"], [24, 26])
        self.assertEqual(value["official_fk_palm_joints"], [22, 23])
        self.assertEqual(value["paired_unit"], "sequence")
        statistic = value["direct_minus_fk_macro_5cm_f1"]
        self.assertEqual(statistic["bootstrap_replicates"], 10_000)
        self.assertEqual(statistic["bootstrap_seed"], 42)
        self.assertEqual(len(statistic["per_unit"]["first"]), len(records))

    def test_bundle_declares_that_it_enters_no_gate(self):
        value = d2ag.reported_only_quantities(variant_records(better=False))
        self.assertTrue(value["reported_only"])
        self.assertFalse(value["enters_internal_mechanism_gate"])
        self.assertFalse(value["enters_native_gate"])
        self.assertEqual(
            sorted(
                key for key in value
                if key.endswith(("structure", "correspondence"))
            ),
            ["anchor_vs_fk_palm_correspondence", "per_sequence_contact_structure"],
        )


def relation_window(variant: str, *, chunk: int, window: int) -> dict:
    """Structurally valid per-window capture record for one variant."""
    expected_count = d2ag.SOURCE_IS_CURRENT_STEP_COUNTS[variant]
    first = internal_runner.expected_first_step_source(variant)
    if variant == "high_t_restricted":
        sources = [
            internal_runner.SOURCE_LABEL_CURRENT
            if timestep >= D2AG_HIGH_T_SELF_CONDITION_CUTOFF
            else internal_runner.SOURCE_LABEL_PREVIOUS
            for timestep in reversed(range(500))
        ]
    else:
        sources = [internal_runner.SOURCE_LABEL_PREVIOUS] * 500
        for index in range(expected_count):
            sources[index] = internal_runner.SOURCE_LABEL_CURRENT
    if sources[0] != first:
        sources[0] = first
    axis = {
        "temporal_anchors": [0, 5, 10, 15],
        "roles": ["left_hand", "right_hand", "pelvis"],
    }

    def filled(shape, value):
        if not shape:
            return value
        return [filled(shape[1:], value) for _ in range(shape[0])]

    trace_values = {}
    for key, shape in native.RELATION_TRACE_SHAPES.items():
        if key == "source_history_max_abs":
            trace_values[key] = filled(shape, 0.0)
        elif key == "source_minus_current_l2_by_anchor":
            trace_values[key] = [
                [0.0] * 4
                if label == internal_runner.SOURCE_LABEL_CURRENT
                else [0.25] * 4
                for label in sources
            ]
        else:
            trace_values[key] = filled(shape, 0.5)
    by_timestep = {
        "timesteps": list(reversed(range(500))),
        "axis": axis,
        "sources": sources,
        "values": {key: trace_values[key] for key in sorted(trace_values)},
    }
    return {
        "chunk_index": chunk,
        "window_index": window,
        "forward_calls": 500,
        "sources": sources,
        "source_is_x_t_count": sum(
            1 for item in sources
            if item == internal_runner.SOURCE_LABEL_CURRENT
        ),
        "first_step_source": sources[0],
        "source_history_max_abs": 0.0,
        "estimate_pass_observed": False,
        "axis": axis,
        "values": {
            key: filled(shape, 0.5)
            for key, shape in sorted(native.RELATION_MEAN_SHAPES.items())
        },
        "by_timestep": by_timestep,
        "by_timestep_sha256": internal_runner.sha256_json(by_timestep),
        "metadata": {
            "rest_object_points_shape": [8, 100, 3],
            "world_to_local_rotation_shape": [8, 3, 3],
            "object_rotation_reference_shape": [8, 3, 3],
            "device": "cuda:0",
            "dtype": "torch.float32",
            "finite": True,
        },
    }


def relation_windows(variant: str) -> list:
    return [
        relation_window(variant, chunk=chunk, window=window)
        for chunk, window in native.EXPECTED_STREAM_COORDINATES
    ]


def conditioning_rows() -> list:
    rows = []
    for chunk, window in native.EXPECTED_STREAM_COORDINATES:
        rows.append({
            "chunk_index": chunk,
            "window_index": window,
            "path_local_provenance": native._expected_provenance(window),
            "exogenous": {
                "shapes": {
                    key: list(shape)
                    for key, shape in native.EXOGENOUS_SHAPES.items()
                },
                "sha256": {
                    key: f"{index:064x}"
                    for index, key in enumerate(sorted(native.EXOGENOUS_SHAPES))
                },
            },
            "path_local_model_inputs": {
                "shapes": {
                    key: list(shape)
                    for key, shape in native.MODEL_INPUT_SHAPES.items()
                },
                "sha256": {
                    key: f"{index:064x}"
                    for index, key in enumerate(sorted(native.MODEL_INPUT_SHAPES))
                },
            },
        })
    for row in rows:
        # The shared rest-object-point hash must agree across both bundles.
        row["path_local_model_inputs"]["sha256"]["rest_object_points"] = (
            row["exogenous"]["sha256"]["rest_object_points"]
        )
    return rows


class D2AGNativeStructuralValidatorTests(unittest.TestCase):
    def test_relation_window_protocol_accepts_every_registered_variant(self):
        for variant in d2ag.VARIANTS:
            self.assertTrue(
                native._relation_window_protocol(
                    variant, relation_windows(variant),
                ),
                variant,
            )

    def test_wrong_source_count_is_rejected(self):
        windows = relation_windows(d2ag.REFERENCE_VARIANT)
        windows[0]["source_is_x_t_count"] = 2
        self.assertFalse(
            native._relation_window_protocol(d2ag.REFERENCE_VARIANT, windows)
        )

    def test_source_label_must_agree_with_the_recorded_l2(self):
        windows = relation_windows(d2ag.REFERENCE_VARIANT)
        window = windows[0]
        # Claim an x_t step while the recorded per-anchor L2 is nonzero.
        window["by_timestep"]["values"][
            "source_minus_current_l2_by_anchor"
        ][0] = [0.5] * 4
        window["by_timestep_sha256"] = internal_runner.sha256_json(
            window["by_timestep"]
        )
        self.assertFalse(
            native._relation_window_protocol(d2ag.REFERENCE_VARIANT, windows)
        )

    def test_nonzero_history_pin_is_rejected(self):
        windows = relation_windows(d2ag.REFERENCE_VARIANT)
        windows[0]["source_history_max_abs"] = 1e-9
        self.assertFalse(
            native._relation_window_protocol(d2ag.REFERENCE_VARIANT, windows)
        )

    def test_estimate_pass_during_sampling_is_rejected(self):
        windows = relation_windows(d2ag.REFERENCE_VARIANT)
        windows[3]["estimate_pass_observed"] = True
        self.assertFalse(
            native._relation_window_protocol(d2ag.REFERENCE_VARIANT, windows)
        )

    def test_high_t_variant_requires_the_registered_split(self):
        windows = relation_windows("high_t_restricted")
        self.assertTrue(
            native._relation_window_protocol("high_t_restricted", windows)
        )
        self.assertEqual(windows[0]["source_is_x_t_count"], 250)
        wrong = relation_windows(d2ag.REFERENCE_VARIANT)
        self.assertFalse(
            native._relation_window_protocol("high_t_restricted", wrong)
        )

    def test_conditioning_protocol_rejects_a_hashed_relation_source(self):
        rows = conditioning_rows()
        self.assertTrue(native._conditioning_protocol(rows))
        rows[0]["path_local_model_inputs"]["sha256"]["relation_source"] = (
            "f" * 64
        )
        rows[0]["path_local_model_inputs"]["shapes"]["relation_source"] = [
            8, 16, 232,
        ]
        self.assertFalse(native._conditioning_protocol(rows))

    def test_conditioning_protocol_requires_the_selfcond_provenance_field(self):
        rows = conditioning_rows()
        del rows[0]["path_local_provenance"]["self_conditioning_source"]
        self.assertFalse(native._conditioning_protocol(rows))

    def test_noise_protocol_binds_the_d2ag_sampler_namespace(self):
        rows = []
        for chunk, window in native.EXPECTED_STREAM_COORDINATES:
            label = internal_runner.sampler_seed_label(chunk, window)
            rows.append({
                "chunk_index": chunk,
                "window_index": window,
                "label": label,
                "seed": internal_runner.base.stable_seed(label),
                "generator_initial_state_sha256": "a" * 64,
                "generator_final_state_sha256": "b" * 64,
                "draw_contract": {
                    "initial_latent_draws": 1,
                    "posterior_noise_draws": 499,
                    "total_generator_draws": 500,
                    "draw_shape": [8, 16, 232],
                    "timestep_zero_noise": "zeros_without_generator_draw",
                },
            })
        self.assertTrue(native._noise_protocol(rows))
        rows[0]["label"] = rows[0]["label"].replace("d2ag", "d2af")
        self.assertFalse(native._noise_protocol(rows))

    def test_native_module_binds_d2ag_floors_and_no_repair_reference(self):
        from train_hoi_prior import (  # noqa: PLC0415
            D2AF_MAXIMUM_ETA_HOURS,
            D2AF_MINIMUM_THROUGHPUT,
            D2AG_MAXIMUM_ETA_HOURS,
            D2AG_MINIMUM_THROUGHPUT,
        )

        self.assertEqual(native.D2AG_MINIMUM_THROUGHPUT, D2AG_MINIMUM_THROUGHPUT)
        self.assertEqual(native.D2AG_MAXIMUM_ETA_HOURS, D2AG_MAXIMUM_ETA_HOURS)
        self.assertNotEqual(
            native.D2AG_MINIMUM_THROUGHPUT, D2AF_MINIMUM_THROUGHPUT,
        )
        self.assertNotEqual(
            native.D2AG_MAXIMUM_ETA_HOURS, D2AF_MAXIMUM_ETA_HOURS,
        )
        self.assertFalse(hasattr(native, "SEALED_D2AE_AGGREGATE_SHA256"))
        self.assertFalse(hasattr(native, "SEALED_D2AE_CHECKPOINT_SHA256"))

    def test_internal_cohort_hashes_match_the_sealed_predecessor_values(self):
        from tools import run_hoi_d2af_native_evaluation as d2af_native  # noqa: PLC0415

        self.assertEqual(
            native.INTERNAL_SELECTION_SHA256,
            d2af_native.INTERNAL_SELECTION_SHA256,
        )
        self.assertEqual(
            native.INTERNAL_SEQUENCE_NAMES_SHA256,
            d2af_native.INTERNAL_SEQUENCE_NAMES_SHA256,
        )
        self.assertEqual(
            native.EXPECTED_INITIAL_MODEL_STATE_SHA256,
            d2af_native.EXPECTED_INITIAL_MODEL_STATE_SHA256,
        )
        self.assertEqual(
            native.CONTROL_CHECKPOINT_SHA256,
            d2af_native.CONTROL_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            native.RELEASED_BASELINE_SHA256,
            d2af_native.RELEASED_BASELINE_SHA256,
        )

    def test_artifact_filenames_include_both_new_appendices(self):
        self.assertEqual(
            set(native.INTERNAL_ARTIFACT_FILENAMES),
            set(d2ag.VARIANTS) | set(native.INTERNAL_SUPPORT_ARTIFACTS),
        )
        self.assertEqual(
            native.INTERNAL_ARTIFACT_FILENAMES["self_conditioning_appendix"],
            "self_conditioning_appendix.json",
        )
        self.assertEqual(
            native.INTERNAL_ARTIFACT_FILENAMES["reported_only_quantities"],
            "reported_only_quantities.json",
        )


if __name__ == "__main__":
    unittest.main()
