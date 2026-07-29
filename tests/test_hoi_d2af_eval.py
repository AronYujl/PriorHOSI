from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.d2af_diagnostic import (  # noqa: E402
    CONTACT_F1_POINT_MINIMUM,
    KINEMATIC_METRICS,
    PROTECTION_METRICS,
    RELEASED_HIGHER_IS_BETTER,
    RELEASED_LOWER_IS_BETTER,
    VARIANTS,
    internal_mechanism_gate,
    native_gate,
    paired_comparisons,
)
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SENTINELS,
    SQRT_ALPHA_BAR_SHA256,
    canonical_diffusion_schedule,
)
from priors.models import HOI_ARCHITECTURE_D2AF  # noqa: E402
from tools import run_hoi_d2af_internal as internal_runner  # noqa: E402
from tools import run_hoi_d2af_native_evaluation as native_runner  # noqa: E402


def _contact_unit(value: float):
    return {
        "precision": value,
        "recall": value,
        "f1": value,
        "prediction_percent": value,
        "prediction_run_lengths": {"mean_frames": 2.0 + value},
    }


def _record(
    sequence: str,
    *,
    f1: float,
    distance: float,
    left_f1: float | None = None,
    right_f1: float | None = None,
):
    direct = {"thresholds_cm": {}}
    fk = {"thresholds_cm": {}}
    for threshold in (2.0, 5.0, 7.5, 10.0):
        key = f"{threshold:g}"
        direct["thresholds_cm"][key] = {
            "left_hand": _contact_unit(
                f1 if left_f1 is None or key != "5" else left_f1
            ),
            "right_hand": _contact_unit(
                f1 if right_f1 is None or key != "5" else right_f1
            ),
            "union": _contact_unit(f1),
        }
        fk["thresholds_cm"][key] = {
            unit: _contact_unit(f1)
            for unit in ("left_hand", "right_hand", "union")
        }
    return {
        "sequence": sequence,
        "semantic_vs_gt": {
            "thresholds": {
                "0.5": {
                    unit: _contact_unit(f1)
                    for unit in ("left_hand", "right_hand", "union")
                }
            }
        },
        "direct_physical_geometry_vs_gt": direct,
        "fk_physical_geometry_vs_gt": fk,
        "gt_contact_frame_direct_distance": {
            "union": {"mean_cm": distance},
        },
        "kinematics": {
            metric: 1.0 + index * 0.1
            for index, metric in enumerate(KINEMATIC_METRICS)
        },
        "penetration": {
            "hand_pen_loss_omomo": 0.2,
            "human_pen_loss_infbagel": 0.3,
        },
    }


def _internal_records():
    settings = {
        "full_rho": (0.82, 2.0, 0.82, 0.82),
        "unit_rho": (0.74, 2.8, 0.74, 0.74),
        "relation_gate_ablated": (0.62, 3.2, 0.62, 0.62),
        "temporal_correspondence_permuted": (0.68, 3.0, 0.68, 0.68),
        "left_right_role_swapped": (0.78, 2.2, 0.60, 0.60),
    }
    return {
        variant: [
            _record(
                f"sequence-{index:02d}",
                f1=f1,
                distance=distance,
                left_f1=left,
                right_f1=right,
            )
            for index in range(8)
        ]
        for variant, (f1, distance, left, right) in settings.items()
    }


def _difference(lower: float = 0.01, upper: float = 0.02):
    return {
        "bootstrap_95_ci": [lower, upper],
        "first_mean": 0.70,
        "second_mean": 0.68,
    }


def _ratio(upper: float = 1.0):
    return {
        "bootstrap_95_ci": [0.90, upper],
        "mean_ratio": 0.95,
    }


def _native_inputs():
    comparison = {
        "penetration_mask_contract": {"passed": True},
        "target_minus_control_contact_f1": _difference(),
        "target_minus_control_contact_recall": _difference(),
        "target_minus_control_contact_precision": _difference(-0.01, 0.01),
        "target_over_control_protection": {
            metric: _ratio(1.05) for metric in PROTECTION_METRICS
        },
        "contact_f1_released_gap_closure": 0.30,
        "target_vs_sealed_d2ae_repair": {
            "target_minus_control_contact_f1": _difference(),
            "target_minus_control_contact_recall": _difference(),
            "target_over_control_protection": {
                "end_obj_trans_err": _ratio(0.98),
                "foot_sliding": _ratio(0.97),
            },
        },
    }
    internal = {
        "contract_passed": True,
        "internal_status": "passed",
        "relation_path_used": True,
        "schedule_reliability_passed": True,
        "temporal_routing_passed": True,
        "role_binding_passed": True,
        "mechanism_passed": True,
    }
    target_metrics = {
        metric: 1.0
        for metric in set(RELEASED_LOWER_IS_BETTER)
        | set(RELEASED_HIGHER_IS_BETTER)
    }
    target_metrics["contact_f1"] = CONTACT_F1_POINT_MINIMUM + 0.01
    baseline_ratios = {
        metric: 1.0
        for metric in set(RELEASED_LOWER_IS_BETTER)
        | set(RELEASED_HIGHER_IS_BETTER)
    }
    return internal, comparison, target_metrics, baseline_ratios


class D2AFDiagnosticGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparisons = paired_comparisons(_internal_records())

    def test_five_paths_and_all_seven_internal_gates_pass(self):
        self.assertEqual(VARIANTS, (
            "full_rho",
            "unit_rho",
            "relation_gate_ablated",
            "temporal_correspondence_permuted",
            "left_right_role_swapped",
        ))
        self.assertEqual(set(self.comparisons), {
            f"full_rho_vs_{variant}" for variant in VARIANTS[1:]
        })
        decision = internal_mechanism_gate(
            {"all_contracts": True},
            self.comparisons,
        )
        self.assertEqual(decision["internal_status"], "passed")
        self.assertTrue(decision["schedule_reliability_passed"])
        self.assertTrue(decision["relation_path_used"])
        self.assertTrue(decision["temporal_routing_passed"])
        self.assertTrue(decision["role_binding_passed"])
        self.assertTrue(decision["mechanism_passed"])
        self.assertEqual(len(decision["checks"]), 7)
        self.assertTrue(decision["native_evaluation_authorized"])

    def test_schedule_failure_is_distinct_and_still_continues_native(self):
        comparisons = dict(self.comparisons)
        unit = dict(comparisons["full_rho_vs_unit_rho"])
        unit["full_rho_minus_other_direct_union_5cm_f1"] = _difference(
            -0.01, 0.01,
        )
        comparisons["full_rho_vs_unit_rho"] = unit
        decision = internal_mechanism_gate({"contract": True}, comparisons)
        self.assertEqual(decision["internal_status"], "schedule-negative")
        self.assertFalse(decision["schedule_reliability_passed"])
        self.assertFalse(decision["mechanism_passed"])
        self.assertTrue(decision["native_evaluation_authorized"])

    def test_native_classification_precedence_and_mechanism_unverified(self):
        internal, comparison, target, baseline = _native_inputs()
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-positive-candidate-stop",
        )
        self.assertTrue(decision["d2ae_repair_passed"])
        self.assertTrue(decision["d2x_candidate_transfer_passed"])
        self.assertTrue(decision["checkpoint_selected"])

        internal = dict(internal)
        internal["mechanism_passed"] = False
        internal["internal_status"] = "schedule-negative"
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-native-positive-mechanism-unverified-stop",
        )
        self.assertFalse(decision["checkpoint_selected"])

        comparison = dict(comparison)
        repair = dict(comparison["target_vs_sealed_d2ae_repair"])
        repair["target_minus_control_contact_f1"] = _difference(-0.01, 0.01)
        comparison["target_vs_sealed_d2ae_repair"] = repair
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-ae-repair-negative-stop",
        )

    def test_repair_ratio_threshold_is_strict(self):
        internal, comparison, target, baseline = _native_inputs()
        repair = comparison["target_vs_sealed_d2ae_repair"]
        repair["target_over_control_protection"]["end_obj_trans_err"] = _ratio(1.0)
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertFalse(
            decision["d2ae_repair_checks"][
                "af_over_ae_end_object_ratio_ci_upper_lt_1.0"
            ]
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-ae-repair-negative-stop",
        )


class _SnapshotModule:
    def __init__(self, rho: torch.Tensor):
        self.value = {
            "pooled_block_norm": torch.ones(4, 3),
            "pooled_block_variance": torch.full((4, 3), 0.5),
            "relation_norm": torch.arange(1, 5, dtype=torch.float32),
            "temporal_permutation_sensitivity": torch.ones(4),
            "role_swap_sensitivity": torch.ones(4),
            "gate": torch.tensor([0.1]),
            "rho": rho,
            "raw_writeback_norm": torch.ones(16),
            "attenuated_writeback_norm": torch.ones(16) * rho.mean(),
        }

    def snapshot(self):
        return self.value


class D2AFRelationCaptureTests(unittest.TestCase):
    def _capture(self, unit: bool):
        capture = internal_runner.RelationCapture()
        motion = torch.zeros(1, 16, 512)
        schedule = canonical_diffusion_schedule()["sqrt_alpha_bar"]
        for timestep in reversed(range(500)):
            rho = torch.ones(1) if unit else schedule[timestep].reshape(1)
            module = _SnapshotModule(rho)
            output = motion + rho[:, None, None] * 0.01
            capture.hook(module, (motion,), output)
        return capture.result()

    def test_canonical_trace_is_exact_and_separates_raw_attenuated(self):
        result = self._capture(unit=False)
        self.assertEqual(result["rho_mode"], "canonical")
        self.assertLessEqual(result["rho_canonical_max_abs"], 1.0e-7)
        self.assertEqual(result["sqrt_alpha_bar_sha256"], SQRT_ALPHA_BAR_SHA256)
        self.assertEqual(
            result["by_timestep"]["timesteps"],
            list(reversed(range(500))),
        )
        for timestep, expected in SQRT_ALPHA_BAR_SENTINELS.items():
            self.assertEqual(result["rho_sentinels"][str(timestep)], expected)
        self.assertIn(
            "raw_writeback_variance_by_anchor",
            result["by_timestep"]["values"],
        )
        self.assertIn(
            "attenuated_writeback_variance_by_anchor",
            result["by_timestep"]["values"],
        )

    def test_unit_rho_trace_is_independent_counterfactual(self):
        result = self._capture(unit=True)
        self.assertEqual(result["rho_mode"], "unit")
        self.assertEqual(result["rho_unit_max_abs"], 0.0)
        self.assertTrue(all(
            value == 1.0 for value in result["rho_sentinels"].values()
        ))


class D2AFEvaluationRunnerContractTests(unittest.TestCase):
    def test_run_id_architecture_and_resolved_internal_identity(self):
        actual_date = datetime.now().astimezone().strftime("%Y%m%d")
        internal_id = (
            "p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-"
            f"s42-{actual_date}"
        )
        native_id = f"p1-hoi-d2af-native-eval-s42-{actual_date}"
        training_id = (
            f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{actual_date}"
        )
        self.assertIsNotNone(internal_runner.RUN_ID_RE.fullmatch(internal_id))
        self.assertIsNotNone(native_runner.RUN_ID_RE.fullmatch(native_id))
        self.assertIsNotNone(
            native_runner.TRAINING_RUN_ID_RE.fullmatch(training_id)
        )
        args = argparse.Namespace(
            run_id=internal_id,
            target_checkpoint=ROOT / "final.pth",
            target_sha256="a" * 64,
            training_run_id=training_id,
            output_dir=ROOT / "results/d2af-internal",
            metrics=ROOT / "results/d2af-internal.json",
            resolved_config=ROOT / "results/d2af-internal-resolved.json",
            batch_size=8,
            device="cuda:0",
        )
        resolved = internal_runner.resolved_config(args)
        self.assertEqual(resolved["subphase"], "1B-D2-AF0-internal")
        self.assertEqual(resolved["variants"], list(VARIANTS))
        self.assertEqual(
            resolved["target_checkpoint"]["architecture_variant"],
            HOI_ARCHITECTURE_D2AF,
        )
        self.assertEqual(
            resolved["assets"]["sqrt_alpha_bar_sha256"],
            SQRT_ALPHA_BAR_SHA256,
        )

    def test_native_accepts_contract_valid_internal_even_when_mechanism_negative(self):
        actual_date = datetime.now().astimezone().strftime("%Y%m%d")
        run_id = (
            "p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-"
            f"s42-{actual_date}"
        )
        training_id = (
            f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{actual_date}"
        )
        target_sha = "b" * 64
        value = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "selection": {
                "sha256": native_runner.INTERNAL_SELECTION_SHA256,
                "sequences": 64,
                "windows": 192,
            },
            "target_checkpoint": {
                "sha256": target_sha,
                "run_id": training_id,
            },
            "decision": {
                "contract_passed": True,
                "internal_status": "schedule-negative",
                "relation_path_used": True,
                "schedule_reliability_passed": False,
                "temporal_routing_passed": True,
                "role_binding_passed": True,
                "mechanism_passed": False,
                "native_evaluation_authorized": True,
                "classification": (
                    "diffusion-reliability-internal-schedule-negative-"
                    "continue-native"
                ),
            },
            "contract": {
                "paired_noise_identity": True,
                "paired_exogenous_condition_identity": True,
                "paired_initial_history_identity": True,
                "causal_window_overlap_exact": True,
                "generator_draw_contract_exact": True,
                "path_local_condition_provenance": True,
                "current_state_relation_metadata_forwarded": True,
                "current_timestep_forwarded": True,
                "rho_variant_identity": True,
                "canonical_schedule_hash": True,
            },
            "diffusion_reliability_appendix": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "internal.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            args = argparse.Namespace(
                internal_diagnostic=path,
                target_sha256=target_sha,
                training_run_id=training_id,
            )
            validated = native_runner._validate_internal(args)
        self.assertEqual(validated["internal_status"], "schedule-negative")
        self.assertFalse(validated["mechanism_passed"])
        self.assertTrue(validated["contract_passed"])

    def test_training_and_checkpoint_contracts_are_d2af_specific(self):
        internal_source = inspect.getsource(internal_runner.checkpoint_contract)
        native_source = inspect.getsource(native_runner.validate_training_result)
        for source in (internal_source, native_source):
            self.assertIn("HOI_ARCHITECTURE_D2AF", source)
            self.assertIn("diffusion_reliability", source)
        self.assertNotIn("HOI_ARCHITECTURE_D2AE", internal_source)
        self.assertNotIn("HOI_ARCHITECTURE_D2AE", native_source)
        self.assertEqual(
            native_runner.SEALED_D2AE_AGGREGATE_SHA256,
            "157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1",
        )
        self.assertEqual(
            native_runner.SEALED_D2AE_PER_SEQUENCE_SHA256,
            "8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c",
        )


if __name__ == "__main__":
    unittest.main()
