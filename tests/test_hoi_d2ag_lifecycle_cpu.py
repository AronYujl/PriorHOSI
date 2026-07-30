from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.diffusion_schedule import SQRT_ALPHA_BAR_SHA256  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    SparseCurrentStateRelationField,
    selfcond_relation_source_contract_metadata,
)
from tools import diagnose_hoi_d2ag as diagnose  # noqa: E402
from tools import run_hoi_d2ag_internal as internal_runner  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AF_MAXIMUM_ETA_HOURS,
    D2AF_MINIMUM_THROUGHPUT,
    D2AG_FORMAL_SOURCE_SCOPES,
    D2AG_MAXIMUM_ETA_HOURS,
    D2AG_MINIMUM_THROUGHPUT,
    _d2ag_formal_source_contract,
    _validate_d2ag_contract,
)


ACTUAL_DATE = datetime.now().astimezone().strftime("%Y%m%d")


class D2AGCpuContractRunIdTests(unittest.TestCase):
    def test_locked_stems_and_actual_date(self):
        run_id = f"p1-hoi-d2ag-cpu-contract-s42-{ACTUAL_DATE}"
        self.assertEqual(diagnose.validate_actual_run_id(run_id), ACTUAL_DATE)
        retry = f"p1-hoi-d2ag-cpu-contract-r2-s42-{ACTUAL_DATE}"
        self.assertEqual(diagnose.validate_actual_run_id(retry), ACTUAL_DATE)
        for invalid in (
            "p1-hoi-d2af-cpu-contract-s42-" + ACTUAL_DATE,
            "p1-hoi-d2ag-cpu-contract-s42-20200101",
            "p1-hoi-d2ag-cpu-contract-s43-" + ACTUAL_DATE,
            "p1-hoi-d2ag-cpu-contract-r0-s42-" + ACTUAL_DATE,
        ):
            with self.assertRaises(ValueError):
                diagnose.validate_actual_run_id(invalid)

    def test_failure_classification_is_registered(self):
        self.assertEqual(
            diagnose.FAILURE_CLASSIFICATION,
            "selfcond-relation-source-contract-failure-stop",
        )
        self.assertEqual(
            internal_runner.FAILURE_CLASSIFICATION,
            "selfcond-relation-source-contract-failure-stop",
        )


class D2AGConfigContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = diagnose.resolved_config(ROOT, ACTUAL_DATE)

    def test_formal_run_id_and_single_factor_config(self):
        self.assertEqual(
            str(self.cfg.run_id),
            f"p1-hoi-d2ag-selfcond-relation-source-s42-{ACTUAL_DATE}",
        )
        value = diagnose.config_contract(ROOT, self.cfg)
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(
            value["unexpected_single_factor_differences"],
            {"d2ae": [], "d2af": []},
        )

    def test_trainer_lifecycle_binds_d2ag_only(self):
        lifecycle = _validate_d2ag_contract(
            self.cfg, 4, require_performance_gate=False,
        )
        self.assertIsNotNone(lifecycle)
        self.assertFalse(lifecycle["performance_gate_required"])
        self.assertEqual(
            lifecycle["selection_probability"], D2AG_SELF_CONDITION_PROBABILITY,
        )
        self.assertEqual(
            lifecycle["minimum_throughput_windows_per_second"],
            D2AG_MINIMUM_THROUGHPUT,
        )
        self.assertEqual(
            lifecycle["maximum_full_budget_eta_hours"], D2AG_MAXIMUM_ETA_HOURS,
        )

    def test_no_eligibility_gate_is_registered(self):
        self.assertIsNone(self.cfg.get("d2ag_clean_signal_eligibility_path"))
        self.assertIsNone(self.cfg.get("d2ag_clean_signal_eligibility_sha256"))
        self.assertNotIn(
            "tools/run_hoi_d2ag_eligibility.py", D2AG_FORMAL_SOURCE_SCOPES,
        )
        self.assertFalse((ROOT / "tools/run_hoi_d2ag_eligibility.py").exists())

    def test_throughput_floors_are_independent_of_d2af(self):
        self.assertNotEqual(D2AG_MINIMUM_THROUGHPUT, D2AF_MINIMUM_THROUGHPUT)
        self.assertNotEqual(D2AG_MAXIMUM_ETA_HOURS, D2AF_MAXIMUM_ETA_HOURS)

    def test_formal_source_contract_covers_the_runtime_tree(self):
        contract = _d2ag_formal_source_contract(ROOT)
        self.assertEqual(contract["scopes"], list(D2AG_FORMAL_SOURCE_SCOPES))
        self.assertGreater(contract["tracked_file_count"], 0)
        self.assertRegex(contract["sha256"], r"^[0-9a-f]{64}$")


class D2AGScheduleAbsenceTests(unittest.TestCase):
    def test_field_never_registers_the_reliability_schedule(self):
        value = diagnose.schedule_absence_contract()
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(
            value["field_schedule_registered_by_simulated_rank"],
            [False] * 4,
        )
        self.assertEqual(
            value["gaussian_diffusion_sqrt_alpha_bar_sha256"],
            SQRT_ALPHA_BAR_SHA256,
        )

    def test_contract_metadata_declares_attenuation_off(self):
        contract = selfcond_relation_source_contract_metadata()
        self.assertFalse(contract["sqrt_alpha_bar_attenuation"])
        self.assertFalse(contract["schedule_buffer_registered"])
        self.assertFalse(contract["relation_zero_branch"])
        self.assertEqual(
            contract["selection_probability"], D2AG_SELF_CONDITION_PROBABILITY,
        )

    def test_rho_override_is_unavailable_on_a_selfcond_field(self):
        field = SparseCurrentStateRelationField(
            512, selfcond_relation_source=True,
        )
        with self.assertRaises(ValueError):
            field.set_rho_override(1.0)


class D2AGRelationSourceContractTests(unittest.TestCase):
    def test_shared_builder_pins_history_and_detaches(self):
        value = diagnose.relation_source_contract()
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(value["variable_anchors"], list(D2AG_VARIABLE_ANCHORS))
        self.assertEqual(value["history_frames"], 2)
        self.assertTrue(
            all(
                item["rejected"]
                for item in value["invalid_input_rejection"].values()
            )
        )


class D2AGModelContractTests(unittest.TestCase):
    def test_zero_new_parameters_and_exact_d2ae_parity(self):
        value = diagnose.model_contract()
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(value["total_parameters"], 30_087_401)
        self.assertEqual(value["relation_parameters"], 413_953)
        self.assertEqual(value["base_parameters"], 29_673_448)
        self.assertEqual(value["base_parity_max_abs"], 0.0)
        self.assertEqual(value["d2ae_parity_max_abs"], 0.0)
        self.assertEqual(value["estimate_pass_minus_d2ae_max_abs"], 0.0)
        self.assertEqual(
            value["selfcond_minus_d2ae_max_abs_at_zero_alpha"], 0.0,
        )
        self.assertGreater(
            value["selfcond_minus_current_max_abs_activated_gate"], 0.0,
        )
        self.assertEqual(
            value["initial_model_state_sha256"],
            diagnose.EXPECTED_INITIAL_STATE_SHA256,
        )

    def test_predecessor_variants_reject_the_selfcond_arguments(self):
        value = diagnose.model_contract()
        restriction = value["predecessor_argument_restriction"]
        self.assertEqual(len(restriction), 6)
        self.assertTrue(
            all(item["rejected"] for item in restriction.values()), restriction,
        )


class D2AGDiagnosticGateContractTests(unittest.TestCase):
    def test_all_five_gates_behave_as_registered(self):
        value = diagnose.diagnostic_gate_contract()
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(value["high_t_cutoff"], D2AG_HIGH_T_SELF_CONDITION_CUTOFF)
        self.assertEqual(value["object_displacement_m"], [0.10, 0.0, 0.0])
        self.assertEqual(
            value["object_displacement_space"], "metric_after_denormalization",
        )
        self.assertEqual(value["object_displacement_channels"], [216, 219])
        self.assertEqual(
            value["object_displacement_frames"], list(D2AG_VARIABLE_ANCHORS),
        )
        self.assertFalse(value["gate_ablation_variant_used"])
        self.assertLessEqual(
            value["object_delta_minus_registered_max_abs"],
            diagnose.PARITY_TOLERANCE,
        )


class D2AGSamplerContractTests(unittest.TestCase):
    def test_first_step_is_x_t_and_later_steps_use_raw_prev_x0(self):
        value = diagnose.sampler_contract()
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(value["sampler_trace_first"], 499)
        self.assertEqual(value["sampler_trace_last"], 0)
        self.assertEqual(value["sampler_trace_length"], 500)
        self.assertEqual(value["steps_without_relation_source"], 1)
        self.assertEqual(
            value["first_reverse_step_source"], "current_noisy_state",
        )
        self.assertEqual(
            value["later_reverse_step_source"], "previous_step_raw_x0_hat",
        )
        self.assertFalse(value["prepare_clean_x0_applied_to_relation_source"])
        self.assertFalse(value["relation_source_so3_projected"])


class D2AGTrainingRngContractTests(unittest.TestCase):
    def test_mask_generator_never_perturbs_the_global_stream(self):
        cfg = diagnose.resolved_config(ROOT, ACTUAL_DATE)
        value = diagnose.training_rng_contract(cfg)
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(
            value["mask_seed_derivation"],
            "cfg.seed * 1000003 + processed_windows + rank",
        )
        self.assertEqual(
            value["selection_probability"], D2AG_SELF_CONDITION_PROBABILITY,
        )
        noise = value["global_noise_sha256_by_case"]
        self.assertEqual(len(set(noise.values())), 1)
        # Different masks, identical global noise: the streams are independent.
        self.assertGreater(
            len(set(value["estimate_selected_batch_by_case"].values())), 1,
        )


class D2AGCheckpointProvenanceTests(unittest.TestCase):
    def test_loader_fails_closed_in_both_directions(self):
        value = diagnose.checkpoint_rejection_contract()
        self.assertEqual(value["scientific_checkpoint_loads"], 0)
        self.assertTrue(
            all(item["rejected"] for item in value["d2ag_rejects"].values()),
            value["d2ag_rejects"],
        )
        self.assertTrue(
            all(
                item["rejected"]
                for item in value["predecessors_reject_d2ag"].values()
            ),
            value["predecessors_reject_d2ag"],
        )
        for label in ("d2ae", "d2af", "d2x_base", "released"):
            self.assertIn(label, value["d2ag_rejects"])
        for label in ("d2ag_missing_contract", "d2ag_altered_probability"):
            self.assertIn(label, value["d2ag_rejects"])


class D2AGStaticContractTests(unittest.TestCase):
    def test_forbidden_sources_and_single_shared_builder(self):
        value = diagnose.static_contract(ROOT)
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(value["forbidden_imports"], [])
        self.assertEqual(value["forbidden_field_tokens"], [])
        self.assertEqual(value["history_pin_sites"], 1)


class D2AGSealedLineageTests(unittest.TestCase):
    def test_sealed_lineage_is_unfilled_and_fails_closed(self):
        self.assertTrue(
            all(
                value is None
                for value in internal_runner.FORMAL_LINEAGE_SEALED.values()
            )
        )
        import argparse  # noqa: PLC0415

        args = argparse.Namespace(
            training_run_id=(
                f"p1-hoi-d2ag-selfcond-relation-source-s42-{ACTUAL_DATE}"
            ),
            formal_manifest=Path("/tmp/x.json"),
            formal_manifest_sha256="0" * 64,
            training_metrics=Path("/tmp/x.json"),
            training_metrics_sha256="0" * 64,
            training_state=Path("/tmp/x.json"),
            training_state_sha256="0" * 64,
            resume_contract=Path("/tmp/x.json"),
            resume_contract_sha256="0" * 64,
            target_checkpoint=Path("/tmp/x.pth"),
            target_sha256="0" * 64,
        )
        with self.assertRaises(ValueError) as caught:
            internal_runner.sealed_lineage_contract(args)
        self.assertIn("not registered yet", str(caught.exception))

    def test_no_predecessor_hash_leaked_into_the_sealed_table(self):
        from tools import run_hoi_d2af_internal as d2af_internal  # noqa: PLC0415

        forbidden = {
            d2af_internal.FORMAL_MANIFEST_SHA256,
            d2af_internal.FORMAL_METRICS_SHA256,
            d2af_internal.FORMAL_FINAL_CHECKPOINT_SHA256,
            d2af_internal.FORMAL_FINAL_MODEL_STATE_SHA256,
            d2af_internal.FORMAL_CHECKPOINT_SOURCE_COMMIT,
            d2af_internal.FORMAL_EXECUTION_TARGET_COMMIT,
        }
        for value in internal_runner.FORMAL_LINEAGE_SEALED.values():
            self.assertNotIn(value, forbidden)


class D2AGInternalRunIdTests(unittest.TestCase):
    def test_internal_and_training_run_id_stems(self):
        self.assertTrue(
            internal_runner.RUN_ID_RE.fullmatch(
                "p1-hoi-d2ag-selfcond-relation-source-internal-s42-"
                + ACTUAL_DATE
            )
        )
        self.assertTrue(
            internal_runner.TRAINING_RUN_ID_RE.fullmatch(
                f"p1-hoi-d2ag-selfcond-relation-source-s42-{ACTUAL_DATE}"
            )
        )
        self.assertIsNone(
            internal_runner.RUN_ID_RE.fullmatch(
                "p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-s42-"
                + ACTUAL_DATE
            )
        )

    def test_sampler_namespace_is_d2ag_specific(self):
        from tools import run_hoi_d2af_internal as d2af_internal  # noqa: PLC0415

        label = internal_runner.sampler_seed_label(2, 1)
        self.assertIn("d2ag", label)
        self.assertNotEqual(label, d2af_internal.sampler_seed_label(2, 1))

    def test_internal_batch_size_is_exactly_eight(self):
        internal_runner.validate_internal_batch_size(8)
        for invalid in (1, 4, 16, 512):
            with self.assertRaises(ValueError):
                internal_runner.validate_internal_batch_size(invalid)

    def test_resolved_config_records_the_registered_semantics(self):
        import argparse  # noqa: PLC0415

        args = argparse.Namespace(
            run_id=(
                "p1-hoi-d2ag-selfcond-relation-source-internal-s42-"
                + ACTUAL_DATE
            ),
            target_checkpoint=Path("/tmp/x.pth"),
            target_sha256="0" * 64,
            training_run_id=(
                f"p1-hoi-d2ag-selfcond-relation-source-s42-{ACTUAL_DATE}"
            ),
            formal_manifest=Path("/tmp/m.json"),
            formal_manifest_sha256="1" * 64,
            training_metrics=Path("/tmp/t.json"),
            training_metrics_sha256="2" * 64,
            training_state=Path("/tmp/s.json"),
            training_state_sha256="3" * 64,
            resume_contract=Path("/tmp/r.json"),
            resume_contract_sha256="4" * 64,
            output_dir=Path("/tmp/out"),
            metrics=Path("/tmp/out/metrics.json"),
            resolved_config=Path("/tmp/out/config.json"),
            batch_size=8,
            device="cuda:0",
        )
        value = internal_runner.resolved_config(args)
        self.assertEqual(value["subphase"], "1B-D2-AG0-internal")
        self.assertEqual(value["variants"], list(internal_runner.VARIANTS))
        self.assertEqual(
            value["sampling"]["first_reverse_step_source"],
            "current_noisy_state",
        )
        self.assertFalse(
            value["sampling"]["prepare_clean_x0_applied_to_relation_source"]
        )
        self.assertFalse(value["sampling"]["clean_target"])
        self.assertFalse(value["sampling"]["stored_per_frame_bps"])
        selfcond = value["self_conditioning"]
        self.assertEqual(selfcond["relation_exposure_fraction"], 1.0)
        self.assertFalse(selfcond["relation_zero_branch"])
        self.assertFalse(selfcond["sqrt_alpha_bar_attenuation"])
        self.assertFalse(selfcond["gate_ablation_variant_used"])
        self.assertEqual(len(selfcond["gates"]), 6)
        self.assertFalse(value["metrics"]["reported_only_enter_gate"])
        self.assertTrue(value["native_runs_regardless_of_internal_mechanism"])
        self.assertEqual(value["optimizer_created"], False)
        self.assertEqual(value["training_updates"], 0)
        self.assertEqual(value["checkpoint_writes"], 0)


if __name__ == "__main__":
    unittest.main()
