from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.d2ae_diagnostic import (  # noqa: E402
    CONTACT_F1_POINT_MINIMUM,
    TEMPORAL_ANCHORS as DIAGNOSTIC_TEMPORAL_ANCHORS,
    VARIANTS as LOCKED_DIAGNOSTIC_VARIANTS,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AE,
    HOI_ARCHITECTURE_D2AG,
    HOIPrior,
    build_expert,
    load_trained_hoi_prior,
)
from priors.sparse_relation import (  # noqa: E402
    BASE_PARAMETER_COUNT,
    DIAGNOSTIC_VARIANTS as MODEL_DIAGNOSTIC_VARIANTS,
    selfcond_relation_source_contract_metadata,
    OBJECT_NAMES,
    PARAMETER_INCREASE_FRACTION,
    ROLE_JOINTS,
    ROLE_NAMES,
    ROUTING_SLOTS,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    TEMPORAL_ANCHORS,
    TOTAL_PARAMETER_COUNT,
    SparseCurrentStateRelationField,
    build_sparse_relation_geometry,
    sparse_relation_contract_metadata,
    validate_sparse_relation_contract,
)
from tools import diagnose_hoi_d2ae as diagnose  # noqa: E402
from tools import benchmark_hoi_d2ae as d2ae_benchmark  # noqa: E402
from tools import run_hoi_d2ae_internal as d2ae_internal  # noqa: E402
from tools import smoke_hoi_d2ae as d2ae_smoke  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AE_MAXIMUM_ETA_HOURS,
    D2AE_MINIMUM_THROUGHPUT,
    _d2ae_formal_source_contract,
    _validate_d2ae_formal_run_id,
    _validate_d2ae_random_origin_checkpoint,
    _locked_loss_weights,
    _optimization_contract,
    _validate_d2ae_contract,
    _validate_fk_foot_temporal_routing_mode,
)


class D2AESparseAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose.sparse_asset_contract(ROOT)

    def test_existing_d2x_sparse_asset_is_exact_and_immutable(self):
        self.assertEqual(tuple(OBJECT_NAMES), tuple(sorted(OBJECT_NAMES)))
        self.assertEqual(self.result["stacked_shape"], [13, 100, 3])
        self.assertEqual(
            self.result["mapping_sha256"], SPARSE_POINT_MAPPING_SHA256,
        )
        self.assertEqual(
            self.result["manifest_sha256"], SPARSE_POINT_MANIFEST_SHA256,
        )
        self.assertEqual(
            self.result["stacked_tensor_sha256"], SPARSE_POINT_TENSOR_SHA256,
        )
        self.assertTrue(self.result["dataset_byte_exact"])
        self.assertTrue(all(self.result["checks"].values()))

    def test_manifest_records_exact_selection_and_hash_fields(self):
        records = self.result["manifest_payload"]["objects"]
        self.assertEqual([record["name"] for record in records], list(OBJECT_NAMES))
        for record in records:
            self.assertEqual(record["point_count"], 100)
            self.assertEqual(
                record["selection"],
                "numpy.linspace(0,n-1,min(100,n)).round().astype(int64)",
            )
            for key in (
                "source_mesh_sha256",
                "indices_sha256",
                "points_float32_yup_sha256",
            ):
                self.assertRegex(record[key], r"^[0-9a-f]{64}$")


class D2AEGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose.geometry_contract()

    def test_fixed_shapes_surface_parity_and_role_order(self):
        self.assertEqual(TEMPORAL_ANCHORS, (0, 5, 10, 15))
        self.assertEqual(ROLE_JOINTS, (24, 26, 0))
        self.assertEqual(ROLE_NAMES, ("left_hand", "right_hand", "pelvis"))
        self.assertEqual(self.result["surface_shape"], [2, 4, 100, 3])
        self.assertEqual(
            self.result["role_point_features_shape"], [2, 4, 3, 100, 4],
        )
        self.assertEqual(self.result["pooled_blocks_shape"], [2, 4, 3, 256])
        self.assertEqual(self.result["relation_vectors_shape"], [2, 4, 512])
        self.assertEqual(self.result["routed_relation_shape"], [2, 16, 512])
        self.assertEqual(
            self.result["surface_loss_transform_parity_max_abs"], 0.0,
        )

    def test_common_yaw_invariance_and_relative_pose_sensitivity(self):
        self.assertLessEqual(
            self.result["common_global_yaw_surface_max_abs"], 1.0e-6,
        )
        self.assertLessEqual(
            self.result["common_global_yaw_relation_max_abs"], 1.0e-6,
        )
        self.assertGreater(
            self.result["relative_translation_relation_max_abs"], 1.0e-8,
        )
        self.assertGreater(
            self.result["relative_rotation_relation_max_abs"], 1.0e-8,
        )

    def test_role_swap_temporal_routing_and_set_invariance(self):
        self.assertEqual(self.result["left_right_pooled_exchange_max_abs"], 0.0)
        self.assertLessEqual(
            self.result["point_permutation_output_max_abs"], 1.0e-6,
        )
        self.assertLessEqual(
            self.result["point_permutation_relation_max_abs"], 1.0e-6,
        )
        self.assertGreater(
            self.result["temporal_permutation_output_max_abs"], 1.0e-8,
        )
        self.assertTrue(all(self.result["checks"].values()))

    def test_builder_is_torch_only_and_fails_closed_on_shapes(self):
        values = diagnose.synthetic_inputs(batch=2)
        geometry = build_sparse_relation_geometry(
            values["current"],
            values["rest_object_points"],
            values["world_to_local_rotation"],
            values["object_rotation_reference"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
        )
        self.assertEqual(set(geometry), {
            "features", "surface", "role_anchors", "joints",
            "object_translation", "relative_rotation", "local_rotation",
        })
        self.assertTrue(all(torch.is_tensor(value) for value in geometry.values()))
        with self.assertRaisesRegex(ValueError, r"\[B,16,232\]"):
            build_sparse_relation_geometry(
                values["current"][:, :-1],
                values["rest_object_points"],
                values["world_to_local_rotation"],
                values["object_rotation_reference"],
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
            )
        with self.assertRaisesRegex(ValueError, "rest_object_points"):
            build_sparse_relation_geometry(
                values["current"],
                values["rest_object_points"][:, :-1],
                values["world_to_local_rotation"],
                values["object_rotation_reference"],
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
            )

    def test_runtime_snapshot_is_bounded_and_descriptive_only(self):
        field = SparseCurrentStateRelationField(512).eval()
        field.set_gate_override(0.1)
        field.set_capture(True)
        values = diagnose.synthetic_inputs(batch=2)
        motion = torch.randn(
            2, 16, 512, generator=torch.Generator().manual_seed(9),
        )
        with torch.no_grad():
            output = field(
                motion,
                values["current"],
                values["rest_object_points"],
                values["world_to_local_rotation"],
                values["object_rotation_reference"],
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
            )
        snapshot = field.snapshot()
        self.assertEqual(tuple(output.shape), (2, 16, 512))
        self.assertEqual(set(snapshot), {
            "pooled_block_norm",
            "pooled_block_variance",
            "relation_norm",
            "temporal_permutation_sensitivity",
            "role_swap_sensitivity",
            "gate",
        })
        self.assertTrue(all(value.device == motion.device for value in snapshot.values()))
        self.assertTrue(all(not value.requires_grad for value in snapshot.values()))
        self.assertNotIn(
            ".cpu()", inspect.getsource(SparseCurrentStateRelationField.forward),
        )


class D2AEModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose.model_contract()

    def test_exact_parameter_count_output_api_and_zero_gate_parity(self):
        self.assertEqual(BASE_PARAMETER_COUNT, 29_673_448)
        self.assertEqual(SPARSE_RELATION_PARAMETER_COUNT, 413_953)
        self.assertEqual(TOTAL_PARAMETER_COUNT, 30_087_401)
        self.assertLessEqual(PARAMETER_INCREASE_FRACTION, 0.015)
        self.assertEqual(self.result["base_parameters"], BASE_PARAMETER_COUNT)
        self.assertEqual(
            self.result["relation_parameters"], SPARSE_RELATION_PARAMETER_COUNT,
        )
        self.assertEqual(self.result["total_parameters"], TOTAL_PARAMETER_COUNT)
        self.assertEqual(self.result["output_shape"], [1, 16, 232])
        self.assertLessEqual(self.result["base_parity_max_abs"], 1.0e-6)
        self.assertEqual(self.result["shared_state_key_count"], 119)
        self.assertEqual(self.result["sparse_state_key_count"], 10)
        self.assertTrue(self.result["shared_state_exact"])
        self.assertTrue(all(
            key.startswith("network.sparse_relation_field.")
            for key in self.result["sparse_state_keys"]
        ))
        signature = inspect.signature(HOIPrior.forward)
        for name in (
            "rest_object_points", "world_to_local_rotation",
            "object_rotation_reference", "position_minimum", "position_maximum",
            "object_minimum", "object_maximum",
        ):
            self.assertEqual(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

    def test_initial_and_test_only_activated_gradients(self):
        initial = self.result["initial_alpha_gradient"]
        self.assertTrue(initial["alpha"]["finite"])
        self.assertTrue(initial["alpha"]["nonzero"])
        self.assertEqual(initial["gate_value"], 0.0)
        activated = self.result["activated_relation_gradients"]
        self.assertAlmostEqual(activated["gate_value"], 0.1, places=6)
        self.assertTrue(all(
            group["finite"] and group["nonzero"]
            for group in activated["relation_groups"].values()
        ))
        self.assertFalse(self.result["test_only_probe_saved"])
        self.assertEqual(self.result["test_only_probe_optimizer_updates"], 0)

    def test_finiteness_so3_dtype_device_and_batch_propagation(self):
        self.assertTrue(all(self.result["zero_constant_extreme_finite"].values()))
        self.assertTrue(self.result["so3_projection_finite"])
        self.assertTrue(all(
            value["finite"]
            and value["orthogonality_max_abs"] <= 1.0e-5
            and abs(value["determinant_minimum"] - 1.0) <= 1.0e-5
            and abs(value["determinant_maximum"] - 1.0) <= 1.0e-5
            for value in self.result["so3_projection_cases"].values()
        ))
        self.assertTrue(self.result["dtype_device_batch_propagation"])

    def test_train_sampler_builder_hsi_and_mixer_contracts(self):
        self.assertEqual(self.result["train_sampler_surface_parity_max_abs"], 0.0)
        self.assertEqual(self.result["train_sampler_feature_parity_max_abs"], 0.0)
        parity = self.result["native_sampler_metadata_parity"]
        self.assertEqual(len(parity["relation_keys"]), 7)
        self.assertTrue(all(
            value == 0.0 for value in parity["metadata_max_abs_by_key"].values()
        ))
        self.assertEqual(
            parity["sampler_audit"]["sparse_relation_metadata_calls"], 1,
        )
        self.assertTrue(self.result["hsiprior_parameter_storage_independent"])
        self.assertEqual(self.result["mixer_clean_output_contract"], [3, 16, 232])
        hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        self.assertNotIn("rest_object_points", inspect.signature(hsi.forward).parameters)

    def test_field_metadata_fixes_roles_slots_width_and_placement(self):
        metadata = SparseCurrentStateRelationField(512).contract_metadata()
        self.assertEqual(metadata["temporal_anchors"], [0, 5, 10, 15])
        self.assertEqual(metadata["role_joints"], [24, 26, 0])
        self.assertEqual(metadata["point_encoder"], [4, 128, 128])
        self.assertEqual(metadata["pooling"], ["mean", "max"])
        self.assertEqual(metadata["projection"], [768, 512])
        self.assertEqual(metadata["routing_slots"], list(ROUTING_SLOTS))
        self.assertEqual(metadata["alpha_initial"], 0.0)
        self.assertEqual(metadata["sparse_relation_parameters"], 413_953)
        self.assertIn("before_condition_concat", metadata["placement"])

    def test_diagnostic_paths_are_exactly_the_registered_four(self):
        expected = {
            "full", "relation_gate_ablated",
            "temporal_correspondence_permuted", "left_right_role_swapped",
        }
        self.assertEqual(set(MODEL_DIAGNOSTIC_VARIANTS), expected)
        self.assertEqual(tuple(LOCKED_DIAGNOSTIC_VARIANTS), (
            "full", "relation_gate_ablated",
            "temporal_correspondence_permuted", "left_right_role_swapped",
        ))
        self.assertEqual(tuple(DIAGNOSTIC_TEMPORAL_ANCHORS), TEMPORAL_ANCHORS)
        self.assertEqual(CONTACT_F1_POINT_MINIMUM, 0.6598838781)


class D2AEGovernanceTests(unittest.TestCase):
    @staticmethod
    def merged_config():
        current_date = datetime.now().astimezone().strftime("%Y%m%d")
        formal_run_id = (
            f"p1-hoi-d2ae-sparse-relation-field-s42-{current_date}"
        )
        cfg = OmegaConf.merge(
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml"),
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ae.yaml"),
        )
        cfg.repo_root = str(ROOT)
        cfg.split_manifest = str(
            ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )
        cfg.run_id = formal_run_id
        cfg.output_dir = str(ROOT / "results/experiments" / formal_run_id)
        cfg.checkpoint_dir = str(Path(cfg.output_dir) / "checkpoints")
        cfg.metrics_path = str(Path(cfg.output_dir) / "metrics.json")
        cfg.state_path = str(Path(cfg.output_dir) / "training_state.json")
        OmegaConf.resolve(cfg)
        return cfg

    def test_exact_training_optimizer_loss_and_budget_contract(self):
        cfg = self.merged_config()
        _validate_fk_foot_temporal_routing_mode(cfg)
        _validate_d2ae_contract(cfg, 4, require_performance_gate=False)
        self.assertEqual(int(cfg.batch_size), 512)
        self.assertEqual(int(cfg.effective_batch_size), 2048)
        self.assertEqual(int(cfg.gradient_accumulation_steps), 1)
        self.assertEqual(int(cfg.max_processed_windows), 61_440_000)
        self.assertEqual(int(cfg.max_processed_windows) // 2048, 30_000)
        self.assertEqual(
            _locked_loss_weights(cfg),
            {
                "fk": 0.3569973401779424,
                "object_surface": 0.4772322188400037,
                "velocity": 0.1,
                "terminal_goal": 1.0,
            },
        )
        optimization = _optimization_contract(cfg)
        self.assertEqual(optimization["optimizer"], "Adam")
        self.assertEqual(optimization["scheduler"], "none")

    def test_formal_training_requires_hash_bound_passing_performance_gate(self):
        cfg = self.merged_config()
        with self.assertRaisesRegex(ValueError, "sealed performance benchmark"):
            _validate_d2ae_contract(cfg, 4)

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        summary = {
            "schema_version": 1,
            "status": "passed",
            "classification": "performance-gate-passed",
            "run_id": (
                "p1-hoi-d2ae-performance-benchmark-s42-"
                f"{datetime.now().astimezone().strftime('%Y%m%d')}"
            ),
            "formal_run_id": str(cfg.run_id),
            "formal_training_authorized": True,
            "seed": 42,
            "world_size": 4,
            "micro_batch_per_gpu": 512,
            "effective_batch_size": 2048,
            "warmup_updates": 64,
            "measured_updates": 256,
            "total_updates": 320,
            "measured_windows": 524288,
            "throughput_windows_per_second": D2AE_MINIMUM_THROUGHPUT,
            "minimum_throughput_windows_per_second": D2AE_MINIMUM_THROUGHPUT,
            "full_budget_eta_hours": D2AE_MAXIMUM_ETA_HOURS,
            "maximum_full_budget_eta_hours": D2AE_MAXIMUM_ETA_HOURS,
            "memory_headroom_min_bytes": 3 * 1024**3,
            "memory_headroom_required_bytes": 2 * 1024**3,
            "memory_headroom_pass": True,
            "losses_finite": True,
            "gradients_finite": True,
            "relation_gpu_only": True,
            "all_rank_contract_pass": True,
            "contention_pass": True,
            "cpu_dynamic_geometry": False,
            "relation_build_device": "cuda",
            "cuda_timing_synchronized": True,
            "optimizer": "FP32 Adam",
            "optimizer_updates": 320,
            "checkpoint_loads": 0,
            "checkpoint_writes": 0,
            "benchmark_weights_reusable": False,
            "sweep_authorized_on_failure": False,
            "identity": {
                "git_commit": commit,
                "worktree_clean": True,
            },
            "formal_source_contract": _d2ae_formal_source_contract(ROOT),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "benchmark_summary.json"
            path.write_text(
                json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8",
            )
            cfg.d2ae_performance_benchmark_path = str(path.resolve())
            cfg.d2ae_performance_benchmark_sha256 = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            result = _validate_d2ae_contract(cfg, 4)
            self.assertTrue(result["performance_gate"]["checks"]["throughput"])
            self.assertEqual(
                result["performance_gate"]["formal_run_id"], str(cfg.run_id),
            )

            missing_formal = dict(summary)
            missing_formal.pop("formal_run_id")
            missing_formal_path = root / "benchmark_summary_missing_formal.json"
            missing_formal_path.write_text(
                json.dumps(missing_formal, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cfg.d2ae_performance_benchmark_path = str(
                missing_formal_path.resolve()
            )
            cfg.d2ae_performance_benchmark_sha256 = hashlib.sha256(
                missing_formal_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "formal_run_id"):
                _validate_d2ae_contract(cfg, 4)

            old_identity = dict(summary)
            old_identity["run_id"] = (
                "p1-hoi-d2ae-performance-benchmark-s42-19990101"
            )
            old_identity["formal_run_id"] = (
                "p1-hoi-d2ae-sparse-relation-field-s42-19990101"
            )
            old_identity_path = root / "benchmark_summary_old_identity.json"
            old_identity_path.write_text(
                json.dumps(old_identity, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cfg.d2ae_performance_benchmark_path = str(
                old_identity_path.resolve()
            )
            cfg.d2ae_performance_benchmark_sha256 = hashlib.sha256(
                old_identity_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "formal_run_id"):
                _validate_d2ae_contract(cfg, 4)

            retry_cfg = self.merged_config()
            retry_cfg.run_id = str(retry_cfg.run_id).replace(
                "-s42-", "-r1-s42-",
            )
            retry_summary = dict(summary)
            retry_summary["formal_run_id"] = str(retry_cfg.run_id)
            retry_path = root / "benchmark_summary_retry.json"
            retry_path.write_text(
                json.dumps(retry_summary, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            retry_cfg.d2ae_performance_benchmark_path = str(
                retry_path.resolve()
            )
            retry_cfg.d2ae_performance_benchmark_sha256 = hashlib.sha256(
                retry_path.read_bytes()
            ).hexdigest()
            _validate_d2ae_contract(retry_cfg, 4)
            retry_summary["formal_run_id"] = str(cfg.run_id)
            retry_path.write_text(
                json.dumps(retry_summary, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            retry_cfg.d2ae_performance_benchmark_sha256 = hashlib.sha256(
                retry_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "formal_run_id"):
                _validate_d2ae_contract(retry_cfg, 4)

            failed = dict(summary)
            failed["throughput_windows_per_second"] = (
                D2AE_MINIMUM_THROUGHPUT - 1.0
            )
            failed_path = root / "benchmark_summary_failed.json"
            failed_path.write_text(
                json.dumps(failed, sort_keys=True) + "\n", encoding="utf-8",
            )
            cfg.d2ae_performance_benchmark_path = str(failed_path.resolve())
            cfg.d2ae_performance_benchmark_sha256 = hashlib.sha256(
                failed_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "performance benchmark contract"):
                _validate_d2ae_contract(cfg, 4)

    def test_training_contract_rejects_any_locked_mutation(self):
        for field, value in (
            ("init_checkpoint", "/tmp/released.pth"),
            ("weight_init_checkpoint", "/tmp/d2ad.pth"),
            ("fk_foot_temporal_routing", False),
            ("batch_size", 256),
            ("effective_batch_size", 1024),
            ("gradient_accumulation_steps", 2),
            ("num_workers", 3),
            ("max_processed_windows", 3_072_000),
            ("learning_rate", 2.0e-4),
            ("amp", True),
            ("d2ac_interaction_adapter", True),
        ):
            cfg = self.merged_config()
            setattr(cfg, field, value)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "D2-AE"):
                    _validate_d2ae_contract(
                        cfg, 4, require_performance_gate=False,
                    )
        with self.assertRaisesRegex(ValueError, "D2-AE"):
            _validate_d2ae_contract(
                self.merged_config(), 1, require_performance_gate=False,
            )

    def test_formal_retry_run_id_and_random_origin_resume_are_fail_closed(self):
        current_date = datetime.now().astimezone().strftime("%Y%m%d")
        base_run_id = (
            f"p1-hoi-d2ae-sparse-relation-field-s42-{current_date}"
        )
        retry_run_id = (
            f"p1-hoi-d2ae-sparse-relation-field-r1-s42-{current_date}"
        )
        self.assertFalse(_validate_d2ae_formal_run_id(base_run_id)["retry"])
        self.assertTrue(_validate_d2ae_formal_run_id(retry_run_id)["retry"])
        cfg = self.merged_config()
        cfg.run_id = retry_run_id
        _validate_d2ae_contract(cfg, 4, require_performance_gate=False)
        with self.assertRaisesRegex(ValueError, "formal run id"):
            _validate_d2ae_formal_run_id(
                f"p1-hoi-d2ae-sparse-relation-field-r0-s42-{current_date}"
            )
        prior_run_id = "p1-hoi-d2ae-sparse-relation-field-s42-19990101"
        with self.assertRaisesRegex(ValueError, "actual date"):
            _validate_d2ae_formal_run_id(prior_run_id)
        self.assertFalse(
            _validate_d2ae_formal_run_id(
                prior_run_id, require_actual_date=False,
            )["date_is_actual"]
        )
        resume_cfg = self.merged_config()
        resume_cfg.run_id = prior_run_id
        resume_cfg.resume_checkpoint = (
            f"/tmp/{prior_run_id}_windows000000002048.pth"
        )
        _validate_d2ae_contract(
            resume_cfg, 4, require_performance_gate=False,
        )

        initialization = {
            "mode": "random",
            "source_checkpoint": None,
            "source_checkpoint_sha256": None,
            "source_model_state_sha256": None,
            "initial_model_state_sha256": "a" * 64,
            "restored_components": [],
            "old_optimizer_states_loaded": 0,
            "old_ema_models_loaded": 0,
            "old_scheduler_states_loaded": 0,
            "old_scaler_states_loaded": 0,
            "old_rng_states_loaded": 0,
        }
        checkpoint = {
            "schema_version": 2,
            "checkpoint_type": "hoi_prior_phase1b",
            "window_state_codec": "state-compositional-v1",
            "expert": "hoi",
            "initialization": "random",
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
            "weight_initialization": initialization,
            "ema_models": {},
            "primary_weight_variant": "online",
            "model": {},
        }
        self.assertTrue(all(
            _validate_d2ae_random_origin_checkpoint(
                checkpoint, "a" * 64,
            )["checks"].values()
        ))
        for field, value in (
            ("expert", "hsi"),
            ("initialization", "released"),
            ("architecture_variant", "base"),
        ):
            tampered = dict(checkpoint)
            tampered[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "random-origin provenance"):
                    _validate_d2ae_random_origin_checkpoint(tampered, "a" * 64)
        tampered = dict(checkpoint)
        tampered["weight_initialization"] = dict(initialization)
        tampered["weight_initialization"]["old_rng_states_loaded"] = 1
        with self.assertRaisesRegex(ValueError, "random-origin provenance"):
            _validate_d2ae_random_origin_checkpoint(tampered, "a" * 64)
        tampered = dict(checkpoint)
        tampered["weight_initialization"] = dict(initialization)
        tampered["weight_initialization"]["initial_model_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "random-origin provenance"):
            _validate_d2ae_random_origin_checkpoint(tampered, "a" * 64)

    def test_old_and_released_checkpoint_schemas_are_rejected(self):
        result = diagnose.checkpoint_rejection_contract()
        self.assertEqual(
            set(result["variants"]), {"released", "d2x_base", "d2ac", "d2ad"},
        )
        self.assertTrue(all(
            value["rejected"] for value in result["variants"].values()
        ))
        self.assertEqual(result["scientific_checkpoint_loads"], 0)

    def test_loader_rejects_locked_routing_contact_and_trunk_mutations(self):
        base = {
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AE,
            },
            "sparse_relation_contract": sparse_relation_contract_metadata(),
        }
        mutations = (
            ("routing_slots", [0] * 16),
            ("contact_used", True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            for field, value in mutations:
                checkpoint = dict(base)
                checkpoint["sparse_relation_contract"] = dict(
                    base["sparse_relation_contract"]
                )
                checkpoint["sparse_relation_contract"][field] = value
                path = temporary / f"mutated-{field}.pth"
                torch.save(checkpoint, path)
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        ValueError, "sparse-relation provenance mismatch",
                    ):
                        load_trained_hoi_prior(
                            str(path),
                            torch.device("cpu"),
                            use_ema=False,
                            expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
                        )
            wrong_trunk = dict(base)
            wrong_trunk["model_config"] = dict(base["model_config"])
            wrong_trunk["model_config"]["num_heads"] = 8
            path = temporary / "wrong-num-heads.pth"
            torch.save(wrong_trunk, path)
            with self.assertRaisesRegex(ValueError, "locked trunk"):
                load_trained_hoi_prior(
                    str(path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
                )

    def test_canonical_contract_rejects_missing_and_unexpected_fields(self):
        complete = sparse_relation_contract_metadata()
        missing = dict(complete)
        missing.pop("role_joints")
        with self.assertRaisesRegex(ValueError, "mismatched=role_joints"):
            validate_sparse_relation_contract(missing)
        unexpected = dict(complete)
        unexpected["extra_direction"] = True
        with self.assertRaisesRegex(ValueError, "unexpected=extra_direction"):
            validate_sparse_relation_contract(unexpected)
        # A D2-AG selfcond-relation-source contract is not a D2-AE contract.
        with self.assertRaises(ValueError):
            validate_sparse_relation_contract(
                selfcond_relation_source_contract_metadata()
            )

    def test_loader_rejects_a_d2ag_selfcond_relation_source_checkpoint(self):
        torch.manual_seed(42)
        d2ag = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AG,
        )
        d2ag_checkpoint = {
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AG,
            },
            "selfcond_relation_source_contract": (
                selfcond_relation_source_contract_metadata()
            ),
            "model": d2ag.state_dict(),
        }
        # A D2-AE checkpoint that also smuggles the D2-AG contract must fail.
        torch.manual_seed(42)
        d2ae = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AE,
        )
        contaminated = {
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AE,
            },
            "sparse_relation_contract": sparse_relation_contract_metadata(),
            "selfcond_relation_source_contract": (
                selfcond_relation_source_contract_metadata()
            ),
            "model": d2ae.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            d2ag_path = temporary / "d2ag.pth"
            contaminated_path = temporary / "d2ae-with-d2ag-contract.pth"
            torch.save(d2ag_checkpoint, d2ag_path)
            torch.save(contaminated, contaminated_path)
            with self.assertRaisesRegex(ValueError, "architecture variant mismatch"):
                load_trained_hoi_prior(
                    str(d2ag_path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
                )
            with self.assertRaisesRegex(
                ValueError, "D2-AG selfcond-relation-source contract",
            ):
                load_trained_hoi_prior(
                    str(contaminated_path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
                )

    def test_internal_causal_overlap_and_path_local_protocol_are_explicit(self):
        class DatasetStub:
            indices = np.asarray((0, 1, 2), dtype=np.int64)
            sequence_ids = np.asarray((0, 0, 0), dtype=np.int64)
            starts = np.asarray((100, 142, 184), dtype=np.int64)
            ends = np.asarray((148, 190, 232), dtype=np.int64)
            language = {"pi": np.asarray((14, 56, 98), dtype=np.int64)}
            scene_names = np.asarray(("sequence_object_action",), dtype=object)

        contract = d2ae_internal.causal_overlap_contract(
            DatasetStub(), ((0, 1, 2),),
        )
        self.assertTrue(contract["all_exact"])
        self.assertEqual(contract["prior_rollout_offsets"], [0, 42, 84])
        overlaps = contract["rows"][0]["sampled_tail_to_next_history"]
        self.assertEqual(overlaps[0]["previous_tail"], [142, 145])
        self.assertEqual(overlaps[0]["next_history"], [142, 145])
        source = inspect.getsource(d2ae_internal.rollout_chunk)
        for token in (
            "previous_generated_tail_from_same_variant",
            "same_path_local_frame.object_reference",
            "initial_latent_draws",
            "posterior_noise_draws",
            "zeros_without_generator_draw",
        ):
            self.assertIn(token, source)

    def test_static_scope_relation_source_and_official_evaluator_contract(self):
        result = diagnose.static_contract(ROOT)
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["forbidden_imports"])
        self.assertFalse(result["forbidden_builder_hits"])
        source = inspect.getsource(build_sparse_relation_geometry).lower()
        for forbidden in (
            "x_start", "future_gt", "contact_label", "scene", "ckdtree",
            "cdist", "full_mesh", "stored_relation",
        ):
            self.assertNotIn(forbidden, source)

    def test_cpu_runner_is_clean_only_and_lifecycle_ids_use_actual_date(self):
        current_date = datetime.now().astimezone().strftime("%Y%m%d")
        cpu_run_id = f"p1-hoi-d2ae-cpu-contract-s42-{current_date}"
        retry_cpu_run_id = (
            f"p1-hoi-d2ae-cpu-contract-r1-s42-{current_date}"
        )
        source = (ROOT / "tools/diagnose_hoi_d2ae.py").read_text(encoding="utf-8")
        self.assertNotIn("allow-dirty", source)
        self.assertIn("require_clean=True", source)
        self.assertIsNotNone(diagnose.RUN_ID_RE.fullmatch(cpu_run_id))
        self.assertIsNotNone(diagnose.RUN_ID_RE.fullmatch(retry_cpu_run_id))
        self.assertIs(
            inspect.signature(diagnose.run_contract).parameters["run_id"].default,
            inspect.Signature.empty,
        )
        self.assertIn('parser.add_argument("--run-id", required=True)', source)
        train_source = (
            ROOT / "code/train_hoi_prior.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sparse_relation_gradient_audit", train_source)
        self.assertIn("sparse_relation_field", train_source)
        for token in (
            '"initial_zero_gate_alpha_gradient"',
            '"activated_relation_gradients"',
            '"initial_model_instance_contract"',
            '"final_model_instance_contract"',
            '"terminal_model_state_sha256"',
            '"d2ae_lifecycle_contract"',
            '"d2ae_performance_benchmark_sha256"',
        ):
            self.assertIn(token, train_source)
        internal_source = (
            ROOT / "tools/run_hoi_d2ae_internal.py"
        ).read_text(encoding="utf-8")
        preflight_source = (
            ROOT / "tools/capture_hoi_worker_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "/home/yujinlun/data/envs/infbagel/bin/python", internal_source,
        )
        self.assertIn("configured_python", preflight_source)

    def test_registry_validation_and_resolved_config_contract(self):
        current_date = datetime.now().astimezone().strftime("%Y%m%d")
        formal_run_id = (
            f"p1-hoi-d2ae-sparse-relation-field-s42-{current_date}"
        )
        result = diagnose.training_and_registry_contract(ROOT, formal_run_id)
        self.assertEqual(result["registry_validation_returncode"], 0)
        self.assertFalse(result["resolved_config_has_unresolved_interpolation"])
        self.assertEqual(result["split_sha256"], diagnose.EXPECTED_SPLIT_SHA256)

    def test_functional_smoke_protocol_and_resolve_only_are_exact(self):
        current_date = datetime.now().astimezone().strftime("%Y%m%d")
        smoke_run_id = (
            f"p1-hoi-d2ae-gpu-functional-smoke-s42-{current_date}"
        )
        formal_run_id = (
            f"p1-hoi-d2ae-sparse-relation-field-s42-{current_date}"
        )
        self.assertEqual(d2ae_smoke.EXPECTED_BATCH_SIZE, 8)
        self.assertEqual(
            set(d2ae_smoke.REGISTERED_TIMESTEPS), {0, 249, 499},
        )
        self.assertIsNotNone(d2ae_smoke.RUN_ID_RE.fullmatch(smoke_run_id))
        self.assertIsNotNone(d2ae_smoke.RUN_ID_RE.fullmatch(
            f"p1-hoi-d2ae-gpu-functional-smoke-r1-s42-{current_date}"
        ))
        with self.assertRaisesRegex(ValueError, "actual date"):
            d2ae_smoke._validate_actual_run_id(
                "p1-hoi-d2ae-gpu-functional-smoke-s42-19990101"
            )
        cfg = d2ae_smoke._resolved_config(ROOT, formal_run_id)
        resolved = d2ae_smoke._resolved_workload_config(
            cfg,
            repo=ROOT,
            run_id=smoke_run_id,
            expected_commit="f" * 40,
            output=ROOT / "results/functional-smoke.json",
            resolved_config_output=ROOT / "results/functional-resolved.yaml",
        )
        self.assertNotIn("${", resolved)
        self.assertIn("visible_gpus: 1", resolved)
        self.assertIn("batch_size: 8", resolved)
        self.assertIn("optimizer_created: false", resolved)
        self.assertIn("checkpoint_writes: 0", resolved)
        source = (ROOT / "tools/smoke_hoi_d2ae.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--run-id", required=True)', source)
        self.assertIn("--resolve-only", source)
        self.assertIn("pre-archived resolved config", source)
        self.assertEqual(source.count('"initial_model_state_sha256"'), 1)

    def test_four_gpu_performance_gate_protocol_and_timing_are_exact(self):
        current_date = datetime.now().astimezone().strftime("%Y%m%d")
        benchmark_run_id = (
            f"p1-hoi-d2ae-performance-benchmark-s42-{current_date}"
        )
        formal_run_id = (
            f"p1-hoi-d2ae-sparse-relation-field-s42-{current_date}"
        )
        self.assertEqual(d2ae_benchmark.WORLD_SIZE, 4)
        self.assertEqual(d2ae_benchmark.MICRO_BATCH_PER_GPU, 512)
        self.assertEqual(d2ae_benchmark.EFFECTIVE_BATCH, 2048)
        self.assertEqual(d2ae_benchmark.WARMUP_UPDATES, 64)
        self.assertEqual(d2ae_benchmark.MEASURED_UPDATES, 256)
        self.assertEqual(d2ae_benchmark.TOTAL_UPDATES, 320)
        self.assertEqual(d2ae_benchmark.MEASURED_WINDOWS, 524_288)
        self.assertAlmostEqual(
            d2ae_benchmark.MINIMUM_THROUGHPUT, 2756.580356467847,
        )
        self.assertIsNotNone(
            d2ae_benchmark.RUN_ID_RE.fullmatch(benchmark_run_id)
        )
        cfg = d2ae_smoke._resolved_config(ROOT, formal_run_id)
        resolved = d2ae_benchmark._resolved_workload_config(
            cfg,
            repo=ROOT,
            run_id=benchmark_run_id,
            expected_commit="e" * 40,
            formal_source_contract=_d2ae_formal_source_contract(ROOT),
            output_dir=ROOT / "results/performance",
            resolved_config_output=ROOT / "results/performance-resolved.yaml",
        )
        self.assertNotIn("${", resolved)
        self.assertIn("torch.distributed.run", resolved)
        self.assertNotIn("torchrun", resolved)
        self.assertIn("--formal-run-id", resolved)
        self.assertIn(formal_run_id, resolved)
        for token in (
            "world_size: 4", "micro_batch_per_gpu: 512",
            "warmup_updates: 64", "measured_updates: 256",
            "checkpoint_writes: 0", "minimum_memory_headroom",
        ):
            self.assertIn(token, resolved)
        source = (
            ROOT / "tools/benchmark_hoi_d2ae.py"
        ).read_text(encoding="utf-8")
        for token in (
            "register_comm_hook", "gpu_relation_geometry", "gpu_relation_module",
            "forward_and_loss", "gradient_validation", "UtilizationMonitor",
            "torch.cuda.synchronize", "benchmark_weights_reusable",
            "--resolve-only", "all_rank_contract_pass", "contention_pass",
            "formal_source_contract", "--formal-run-id",
        ):
            self.assertIn(token, source)
        self.assertNotIn("torch.save", source)


if __name__ == "__main__":
    unittest.main()
