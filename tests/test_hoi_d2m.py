import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

import priors.optimizer_reset as optimizer_reset
from priors.data import PriorWindowDataset
from priors.models import HOIPrior
from priors.optimizer_reset import (
    CANDIDATES,
    NATIVE_SELECTION_SHA256,
    OPTIMIZER_UPDATES,
    TEACHER_SELECTION_SHA256,
    WEIGHTS,
    mechanism_gate,
    paired_difference,
    paired_mean_ratio,
    select_native_holdout,
    select_teacher_holdout,
    stable_seed,
)
from tools.run_hoi_d2m import candidate_overrides
from tools.evaluate_hoi_d2m import per_sequence_native_error
from tools.summarize_hoi_d2m import RUN_ID, validate_identity
from train_hoi_prior import _load_weight_initialization, _state_dict_sha256


class D2MSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )

    def test_teacher_and_native_holdouts_are_locked_fresh_and_deterministic(self):
        first_teacher = select_teacher_holdout(self.dataset)
        second_teacher = select_teacher_holdout(self.dataset)
        native = select_native_holdout(self.dataset)
        self.assertEqual(first_teacher["positions"], second_teacher["positions"])
        self.assertEqual(first_teacher["sha256"], TEACHER_SELECTION_SHA256)
        self.assertEqual(first_teacher["first_rank"], 1026)
        self.assertEqual(first_teacher["last_rank"], 1537)
        self.assertEqual(first_teacher["terminal_windows"], 5)
        self.assertEqual(native["sha256"], NATIVE_SELECTION_SHA256)
        self.assertEqual(native["first_rank"], 128)
        self.assertEqual(native["last_rank"], 159)
        self.assertEqual(native["sequences"], 32)

    def test_evaluation_rng_labels_are_candidate_independent(self):
        label = "D2M:teacher-q:499:0"
        self.assertEqual(stable_seed(label), stable_seed(label))
        self.assertNotIn("current", label)
        self.assertNotIn("balanced", label)


class D2MStatisticsAndGateTests(unittest.TestCase):
    def test_paired_statistics_use_shared_units(self):
        difference = paired_difference([3.0, 4.0, 5.0], [1.0, 2.0, 3.0], replicates=100)
        ratio = paired_mean_ratio([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], replicates=100)
        self.assertAlmostEqual(difference["paired_mean_first_minus_second"], 2.0)
        self.assertAlmostEqual(ratio["mean_ratio"], 0.5)
        self.assertEqual(len(difference["per_unit"]["first"]), 3)
        self.assertEqual(len(ratio["per_unit"]["denominator"]), 3)

    def test_native_object_goal_uses_third_window_and_contact_is_sequence_paired(self):
        metrics = {
            "per_sequence_window": [
                {"sequence": "a", "window": 1, "object_goal_error_cm": 100.0},
                {"sequence": "a", "window": 3, "object_goal_error_cm": 10.0},
                {"sequence": "b", "window": 1, "object_goal_error_cm": 200.0},
                {"sequence": "b", "window": 3, "object_goal_error_cm": 20.0},
            ],
            "per_sequence": [
                {"sequence": "b", "physical_contact_f1": 0.8},
                {"sequence": "a", "physical_contact_f1": 0.6},
            ],
        }
        np = __import__("numpy")
        np.testing.assert_allclose(
            per_sequence_native_error(metrics, "object_goal_error_cm"),
            [10.0, 20.0],
        )
        np.testing.assert_allclose(
            per_sequence_native_error(metrics, "physical_contact_f1"),
            [0.6, 0.8],
        )

    @staticmethod
    def _training():
        return {
            "all_finite": True,
            "source_checkpoint_hash_exact": True,
            "source_model_hash_exact": True,
            "initial_model_hashes_equal": True,
            "old_state_load_counts_zero": True,
            "paired_training_rng_audit": True,
            "candidates": {
                name: {
                    "initial_optimizer_state_count": 0,
                    "terminal_optimizer_state_count": 119,
                    "terminal_optimizer_step_min": OPTIMIZER_UPDATES,
                    "terminal_optimizer_step_max": OPTIMIZER_UPDATES,
                    "optimizer_updates": OPTIMIZER_UPDATES,
                }
                for name in CANDIDATES
            },
        }

    @staticmethod
    def _teacher():
        return {
            "all_fields_conditions_and_physical_reported": True,
            "timesteps": {
                str(timestep): {
                    "finite": True,
                    "history_max_abs": 0.0,
                    "current_minus_balanced": {
                        "joint_positions": {"bootstrap_95_ci": [0.1, 0.2]},
                    },
                    "balanced_over_current": {
                        "object_translation": {"bootstrap_95_ci": [0.8, 1.0]},
                    },
                    "balanced_over_source": {
                        "joint_positions": {"mean_ratio": 0.9},
                        "object_translation": {"mean_ratio": 1.0},
                    },
                }
                for timestep in (250, 499)
            },
        }

    @staticmethod
    def _native():
        return {
            "finite": True,
            "all_native_metrics_reported": True,
            "current_minus_balanced": {
                "mpjpe_cm": {"bootstrap_95_ci": [0.1, 0.2]},
                "object_goal_error_cm": {"bootstrap_95_ci": [0.1, 0.2]},
            },
            "balanced_over_current": {
                "pelvis_goal_error_cm": {"bootstrap_95_ci": [0.9, 1.0]},
            },
        }

    def test_gate_requires_training_teacher_and_native_conjunction(self):
        decision = mechanism_gate(self._training(), self._teacher(), self._native())
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "fresh-optimizer-balanced-smoke-positive-stop",
        )
        self.assertFalse(decision["full_training_authorized"])
        teacher = self._teacher()
        teacher["timesteps"]["499"]["current_minus_balanced"]["joint_positions"][
            "bootstrap_95_ci"
        ] = [-0.1, 0.2]
        decision = mechanism_gate(self._training(), teacher, self._native())
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "fresh-optimizer-balanced-smoke-negative-stop",
        )


class D2MWeightInitializationTests(unittest.TestCase):
    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fixture(self, root: Path):
        split = root / "split.json"
        split.write_text("{}\n", encoding="utf-8")
        model = HOIPrior(dim_model=32, num_heads=4, num_layers=1)
        checkpoint = root / "source.pth"
        torch.save({
            "schema_version": 2,
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "run_id": "synthetic-source",
            "git_commit": "source",
            "processed_windows": 10,
            "optimizer_updates": 2,
            "model_config": {"dim_model": 32, "num_heads": 4, "num_layers": 1},
            "data_contract_sha256": "data",
            "split_sha256": self._file_sha256(split),
            "model": model.state_dict(),
            "ema_models": {"0.999": model.state_dict(), "0.9999": model.state_dict()},
            "optimizer": {"state": {0: {"step": torch.tensor(2)}}},
            "scheduler": {"last_epoch": 2},
            "scaler": {"scale": 1.0},
        }, checkpoint)
        sha256 = self._file_sha256(checkpoint)
        cfg = OmegaConf.create({
            "weight_init_checkpoint": str(checkpoint),
            "weight_init_sha256": sha256,
            "weight_init_variant": "online",
            "d2m_candidate": "current",
            "dim_model": 32,
            "num_heads": 4,
            "num_layers": 1,
            "data_contract_sha256": "data",
            "split_manifest": str(split),
        })
        return model, checkpoint, sha256, cfg

    def test_only_online_model_is_restored_and_old_states_remain_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_model, _, sha256, cfg = self._fixture(Path(temporary))
            target = HOIPrior(dim_model=32, num_heads=4, num_layers=1)
            with mock.patch.object(optimizer_reset, "SOURCE_CHECKPOINT_SHA256", sha256), \
                    mock.patch.object(optimizer_reset, "SOURCE_RUN_ID", "synthetic-source"):
                result = _load_weight_initialization(cfg, target)
            self.assertEqual(result["restored_components"], ["model"])
            self.assertEqual(result["old_optimizer_states_loaded"], 0)
            self.assertEqual(result["old_ema_models_loaded"], 0)
            self.assertEqual(result["old_rng_states_loaded"], 0)
            self.assertEqual(
                result["initial_model_state_sha256"],
                _state_dict_sha256(source_model.state_dict()),
            )

    def test_wrong_variant_and_released_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, checkpoint, sha256, cfg = self._fixture(Path(temporary))
            target = HOIPrior(dim_model=32, num_heads=4, num_layers=1)
            cfg.weight_init_variant = "ema_0.9999"
            with mock.patch.object(optimizer_reset, "SOURCE_CHECKPOINT_SHA256", sha256), \
                    mock.patch.object(optimizer_reset, "SOURCE_RUN_ID", "synthetic-source"):
                with self.assertRaisesRegex(ValueError, "online"):
                    _load_weight_initialization(cfg, target)
            torch.save({"model": target.state_dict()}, checkpoint)
            released_sha = self._file_sha256(checkpoint)
            cfg.weight_init_variant = "online"
            cfg.weight_init_sha256 = released_sha
            with mock.patch.object(
                optimizer_reset, "SOURCE_CHECKPOINT_SHA256", released_sha,
            ), mock.patch.object(optimizer_reset, "SOURCE_RUN_ID", "synthetic-source"):
                with self.assertRaisesRegex(ValueError, "released checkpoint"):
                    _load_weight_initialization(cfg, target)


class D2MConfigAndLifecycleTests(unittest.TestCase):
    def test_candidate_commands_lock_budget_lr_weights_and_shared_source(self):
        source = Path("/tmp/source.pth")
        output = Path("/tmp/d2m")
        current = candidate_overrides(source, output, "current")
        balanced = candidate_overrides(source, output, "balanced")
        self.assertIn(f"weight_init_checkpoint={source}", current)
        self.assertIn("weight_init_variant=online", current)
        self.assertIn(f"fk_weight={WEIGHTS['current']['fk']}", current)
        self.assertIn(f"fk_weight={WEIGHTS['balanced']['fk']}", balanced)
        config = (ROOT / "code/config/config_train_hoi_prior_d2m.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_processed_windows: 196608", config)
        self.assertIn("effective_batch_size: 3072", config)
        self.assertIn("learning_rate: 0.00003", config)
        self.assertIn("minimum_lr_ratio: 1.0", config)
        self.assertIn("d2m_rng_audit: true", config)

    def test_default_random_contract_and_sampler_gt_prohibitions_remain(self):
        trainer = (ROOT / "code/train_hoi_prior.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "tools/evaluate_hoi_d2m.py").read_text(encoding="utf-8")
        production_rollout = (ROOT / "tools/evaluate_hoi_remediation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("init_checkpoint is forbidden", trainer)
        self.assertIn("weight-only initialization is restricted", trainer)
        self.assertIn('"sampler_future_gt": False', evaluator)
        self.assertIn('"sampler_stored_per_frame_bps": False', evaluator)
        self.assertIn("current_bps(", production_rollout)
        self.assertNotIn('batch["object_bps"]', inspect.getsource(
            __import__("tools.evaluate_hoi_remediation", fromlist=["rollout"]).rollout
        ))

    def test_runner_evaluator_and_summary_are_directly_executable(self):
        for relative in (
            "tools/run_hoi_d2m.py",
            "tools/evaluate_hoi_d2m.py",
            "tools/summarize_hoi_d2m.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(ROOT / relative), "--help"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_summary_requires_manifest_experiment_identifier_and_commit(self):
        metrics = {"run_id": RUN_ID, "git_commit": "commit", "status": "completed"}
        manifest = {"experiment_id": RUN_ID, "git": {"commit": "commit"}}
        validate_identity(metrics, manifest)
        with self.assertRaisesRegex(ValueError, "run-id mismatch"):
            validate_identity(metrics, {"run_id": RUN_ID, "git": {"commit": "commit"}})

    def test_preregistered_registry_contract_is_complete(self):
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        record = next(
            value for value in records
            if value["experiment_id"]
            == "p1-hoi-d2m-reset-paired-preregister-s42-20260716"
        )
        self.assertEqual(record["config"]["training"]["optimizer_updates_per_candidate"], 64)
        self.assertEqual(
            set(record["config"]["teacher_evaluation"]["condition_variants"]),
            {"matched", "text_permuted", "bps_permuted", "pelvis_permuted", "object_goal_permuted"},
        )
        self.assertFalse(record["results"]["full_training_authorized"])


if __name__ == "__main__":
    unittest.main()
