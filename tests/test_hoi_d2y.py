import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from priors.losses import (  # noqa: E402
    D2X_FOOT_XZ_VELOCITY_SLOTS,
    _velocity_loss,
)
from train_hoi_prior import (  # noqa: E402
    _build_optimizer,
    _build_scheduler,
    _locked_loss_weights,
    _loss_routing_contract,
    _resume_contract,
    _validate_d2x_contract,
    _validate_d2y_contract,
    _validate_d2y_execution_host,
    _validate_fk_foot_temporal_routing_mode,
)
from tools.run_hoi_d2y_evaluation import (  # noqa: E402
    CONTROL_AGGREGATE_SHA256,
    CONTROL_CHECKPOINT_SHA256,
    CONTROL_PER_SEQUENCE_SHA256,
    INTERNAL_SELECTION_SHA256,
    RUN_ID as EVAL_RUN_ID,
    _internal_decision,
    classify,
    sha256_file,
    validate_training_result,
)
from tools.diagnose_hoi_d2y import (  # noqa: E402
    gradient_cosine,
    mechanism_decision,
)


def merged_config(name):
    base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
    intervention = OmegaConf.load(ROOT / f"code/config/{name}.yaml")
    cfg = OmegaConf.merge(base, intervention)
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(
        ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    return cfg


class D2YConfigAndContractTests(unittest.TestCase):
    def test_exact_config_changes_only_registered_multiplier_and_mode_from_d2x(self):
        d2x = merged_config("config_train_hoi_prior_d2x")
        d2y = merged_config("config_train_hoi_prior_d2y")
        _validate_d2x_contract(d2x, 4)
        _validate_d2y_contract(d2y, 4)
        ignored = {
            "mode",
            "subphase",
            "run_id",
            "d2x_fk_foot_temporal_routing",
            "d2y_routed_foot_amplification",
            "routed_foot_residual_multiplier",
        }
        x_plain = OmegaConf.to_container(d2x, resolve=False)
        y_plain = OmegaConf.to_container(d2y, resolve=False)
        self.assertEqual(
            {key: value for key, value in x_plain.items() if key not in ignored},
            {key: value for key, value in y_plain.items() if key not in ignored},
        )
        self.assertFalse(d2y.d2x_fk_foot_temporal_routing)
        self.assertTrue(d2y.d2y_routed_foot_amplification)
        self.assertTrue(d2y.fk_foot_temporal_routing)
        self.assertEqual(d2y.routed_foot_residual_multiplier, 1024.0)

    def test_optimizer_budget_and_balanced_weights_match_d2x(self):
        d2x = merged_config("config_train_hoi_prior_d2x")
        d2y = merged_config("config_train_hoi_prior_d2y")
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = _build_optimizer(d2y, [parameter])
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertIsNone(_build_scheduler(d2y, optimizer, 30000, 0))
        self.assertEqual(_locked_loss_weights(d2y), _locked_loss_weights(d2x))
        for field in (
            "max_processed_windows",
            "effective_batch_size",
            "learning_rate",
            "fk_weight",
            "object_surface_weight",
            "velocity_weight",
            "goal_weight",
        ):
            self.assertEqual(getattr(d2y, field), getattr(d2x, field))

    def test_contract_fails_closed_on_multiplier_or_provenance_mutation(self):
        mutations = {
            "routed_foot_residual_multiplier": 512.0,
            "fk_foot_temporal_routing": False,
            "velocity_weight": 0.2,
            "learning_rate": 0.0003,
            "resume_checkpoint": "/tmp/forbidden.pth",
            "weight_init_checkpoint": "/tmp/d2x.pth",
            "d2m_candidate": "balanced",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                cfg = merged_config("config_train_hoi_prior_d2y")
                setattr(cfg, field, value)
                with self.assertRaisesRegex(ValueError, "D2-Y"):
                    _validate_d2y_contract(cfg, 4)

    def test_d2x_rejects_d2y_mode_and_multiplier(self):
        cfg = merged_config("config_train_hoi_prior_d2x")
        cfg.d2y_routed_foot_amplification = True
        cfg.routed_foot_residual_multiplier = 1024.0
        with self.assertRaisesRegex(ValueError, "D2-X"):
            _validate_d2x_contract(cfg, 4)

    def test_multiplier_cannot_be_enabled_outside_d2y(self):
        cfg = merged_config("config_train_hoi_prior_d2x")
        cfg.routed_foot_residual_multiplier = 1024.0
        with self.assertRaisesRegex(ValueError, "registered D2-Y"):
            _validate_fk_foot_temporal_routing_mode(cfg)
        _validate_fk_foot_temporal_routing_mode(
            merged_config("config_train_hoi_prior_d2y")
        )

    def test_resume_and_metrics_contract_extend_only_d2y(self):
        d2x = merged_config("config_train_hoi_prior_d2x")
        d2y = merged_config("config_train_hoi_prior_d2y")
        x_resume = _resume_contract(d2x)
        y_resume = _resume_contract(d2y)
        self.assertNotIn("d2y_routed_foot_amplification", x_resume)
        self.assertNotIn("routed_foot_residual_multiplier", x_resume)
        self.assertTrue(y_resume["d2y_routed_foot_amplification"])
        self.assertEqual(y_resume["routed_foot_residual_multiplier"], 1024.0)
        self.assertNotIn("routed_foot_residual_multiplier", _loss_routing_contract(d2x))
        self.assertEqual(
            _loss_routing_contract(d2y),
            {
                "fk_foot_temporal_routing": True,
                "foot_joint_indices": [7, 8, 10, 11],
                "routed_components": ["x", "z"],
                "velocity_weight": 0.1,
                "velocity_reduction": "mean_square",
                "routed_foot_residual_multiplier": 1024.0,
                "nonrouted_residual_multiplier": 1.0,
                "weighted_slots": 8,
                "total_velocity_slots": 87,
            },
        )

    def test_worker_host_and_python_are_fail_closed(self):
        cfg = merged_config("config_train_hoi_prior_d2y")
        with mock.patch("socket.gethostname", return_value="ubuntu"):
            with self.assertRaisesRegex(RuntimeError, "infbagel-4gpu/node01"):
                _validate_d2y_execution_host(cfg)
        environment = {
            "INFBAGEL_WORKER_EXPERT": "hoi",
            "INFBAGEL_PYTHON": "/home/yujinlun/data/envs/infbagel/bin/python",
        }
        with mock.patch("socket.gethostname", return_value="node01"), mock.patch.dict(
            os.environ, environment, clear=False,
        ), mock.patch("train_hoi_prior.sys.executable", environment["INFBAGEL_PYTHON"]):
            _validate_d2y_execution_host(cfg)


class D2YVelocityLossTests(unittest.TestCase):
    def test_unit_multiplier_is_exact_d2x_mse_path(self):
        generator = torch.Generator().manual_seed(42)
        prediction = torch.randn(2, 14, 87, generator=generator)
        target = torch.randn(2, 14, 87, generator=generator)
        expected = F.mse_loss(prediction, target)
        actual = _velocity_loss(
            prediction,
            target,
            fk_foot_temporal_routing=True,
            routed_foot_residual_multiplier=1.0,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_fixed_multiplier_weights_only_eight_registered_slots(self):
        prediction = torch.ones(1, 1, 87, requires_grad=True)
        target = torch.zeros_like(prediction)
        actual = _velocity_loss(
            prediction,
            target,
            fk_foot_temporal_routing=True,
            routed_foot_residual_multiplier=1024.0,
        )
        expected = (79.0 + 8.0 * 1024.0) / 87.0
        self.assertEqual(len(D2X_FOOT_XZ_VELOCITY_SLOTS), 8)
        self.assertAlmostEqual(actual.item(), expected, places=5)
        actual.backward()
        routed = prediction.grad[0, 0, D2X_FOOT_XZ_VELOCITY_SLOTS[0]].item()
        nonrouted = prediction.grad[0, 0, 0].item()
        self.assertAlmostEqual(routed / nonrouted, 1024.0, places=4)

    def test_amplification_requires_routing_and_valid_shape(self):
        residual = torch.zeros(1, 14, 87)
        with self.assertRaisesRegex(ValueError, "requires FK-foot"):
            _velocity_loss(
                residual,
                residual,
                fk_foot_temporal_routing=False,
                routed_foot_residual_multiplier=1024.0,
            )
        with self.assertRaisesRegex(ValueError, "87"):
            _velocity_loss(
                residual[..., :-1],
                residual[..., :-1],
                fk_foot_temporal_routing=True,
                routed_foot_residual_multiplier=1024.0,
            )
        for multiplier in (0.5, float("inf"), float("nan")):
            with self.subTest(multiplier=multiplier):
                with self.assertRaisesRegex(ValueError, "finite and at least one"):
                    _velocity_loss(
                        residual,
                        residual,
                        fk_foot_temporal_routing=True,
                        routed_foot_residual_multiplier=multiplier,
                    )


class D2YInternalDiagnosticTests(unittest.TestCase):
    def test_gradient_cosine_handles_alignment_conflict_and_zero(self):
        aligned = gradient_cosine(
            [torch.tensor([1.0, 2.0])],
            [torch.tensor([2.0, 4.0])],
        )
        conflict = gradient_cosine(
            [torch.tensor([1.0, 2.0])],
            [torch.tensor([-2.0, -4.0])],
        )
        zero = gradient_cosine(
            [torch.tensor([0.0, 0.0])],
            [torch.tensor([1.0, 1.0])],
        )
        self.assertAlmostEqual(aligned, 1.0)
        self.assertAlmostEqual(conflict, -1.0)
        self.assertIsNone(zero)

    def test_internal_gate_requires_both_registered_timesteps(self):
        def model(values):
            return {
                "final": {
                    "timesteps": {
                        timestep: {"routed_residual_mse_by_sequence": values}
                        for timestep in ("249", "499")
                    },
                },
            }

        positive = mechanism_decision({
            "d2x": model([2.0] * 32),
            "d2y": model([1.0] * 32),
        })
        self.assertTrue(positive["mechanism_passed"])
        target = model([1.0] * 32)
        target["final"]["timesteps"]["499"][
            "routed_residual_mse_by_sequence"
        ] = [3.0] * 32
        negative = mechanism_decision({
            "d2x": model([2.0] * 32),
            "d2y": target,
        })
        self.assertFalse(negative["mechanism_passed"])
        self.assertTrue(negative["timestep_checks"]["249"]["passed"])
        self.assertFalse(negative["timestep_checks"]["499"]["passed"])


class D2YGovernanceTests(unittest.TestCase):
    def test_plan_and_registry_bind_single_variable_and_stop_rules(self):
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("Phase 1B D2-Y routed-foot residual amplification", plan)
        self.assertIn("p1-hoi-d2y-routed-foot-amplification-s42-20260723", plan)
        self.assertIn("routed-foot-amplification-transfer-negative-stop", plan)
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        by_id = {record["experiment_id"]: record for record in records}
        training = by_id[
            "p1-hoi-d2y-routed-foot-amplification-preregister-s42-20260723"
        ]
        evaluation = by_id[
            "p1-hoi-d2y-native-eval-preregister-s42-20260723"
        ]
        self.assertEqual(
            training["config"]["manipulated_factor"]["routed_multiplier"], 1024.0
        )
        self.assertTrue(training["config"]["fixed"]["random_initialization"])
        self.assertFalse(training["config"]["consistency_authorized"])
        self.assertEqual(
            evaluation["config"]["control"]["checkpoint_sha256"],
            "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51",
        )


class D2YEvaluationTests(unittest.TestCase):
    @staticmethod
    def comparison(foot_lower=0.01, protection_upper=1.01):
        return {
            "control_minus_target_foot_sliding": {
                "bootstrap_95_ci": [foot_lower, 0.03],
            },
            "target_over_control_protection": {
                metric: {"bootstrap_95_ci": [0.99, protection_upper]}
                for metric in (
                    "mpjpe",
                    "end_obj_trans_err",
                    "xy_points_err",
                    "obj_trans_dist",
                    "hand_pen_loss_omomo",
                    "human_pen_loss_infbagel",
                )
            },
            "target_minus_control_contact_f1": {
                "bootstrap_95_ci": [-0.01, 0.01],
            },
            "penetration_mask_contract": {"passed": True},
        }

    def setUp(self):
        _internal_decision.clear()
        _internal_decision.update({
            "mechanism_passed": True,
            "timestep_checks": {
                "249": {"passed": True},
                "499": {"passed": True},
            },
        })

    def test_control_hashes_and_eval_id_are_sealed_d2x(self):
        self.assertEqual(EVAL_RUN_ID, "p1-hoi-d2y-native-eval-s42-20260723")
        self.assertEqual(
            CONTROL_CHECKPOINT_SHA256,
            "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51",
        )
        self.assertEqual(
            CONTROL_AGGREGATE_SHA256,
            "3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b",
        )
        self.assertEqual(
            CONTROL_PER_SEQUENCE_SHA256,
            "69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a",
        )

    def test_classification_distinguishes_registered_hypotheses(self):
        target = {"contact_f1": 0.64}
        baseline = {
            "mpjpe": 1.0,
            "end_obj_trans_err": 1.0,
            "xy_points_err": 1.0,
            "obj_trans_dist": 1.0,
            "foot_sliding": 1.0,
        }
        positive = classify(
            self.comparison(), target, baseline, contract_passed=True,
        )
        self.assertEqual(
            positive["classification"],
            "routed-foot-amplification-positive-candidate-stop",
        )
        transfer = classify(
            self.comparison(foot_lower=-0.001), target, baseline,
            contract_passed=True,
        )
        self.assertEqual(
            transfer["classification"],
            "routed-foot-amplification-transfer-negative-stop",
        )
        conflict = classify(
            self.comparison(protection_upper=1.11), target, baseline,
            contract_passed=True,
        )
        self.assertEqual(
            conflict["classification"],
            "routed-foot-amplification-conflict-negative-stop",
        )
        _internal_decision["mechanism_passed"] = False
        optimization = classify(
            self.comparison(), target, baseline, contract_passed=True,
        )
        self.assertEqual(
            optimization["classification"],
            "routed-foot-amplification-optimization-negative-stop",
        )

    def test_training_and_internal_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = (
                root / "p1-hoi-d2y-routed-foot-amplification-s42-20260723_"
                "windows061440000.pth"
            )
            checkpoint_sha = "a" * 64
            metrics_path = root / "metrics.json"
            metrics = {
                "status": "stable",
                "run_id": "p1-hoi-d2y-routed-foot-amplification-s42-20260723",
                "seed": 42,
                "initialization": "random",
                "training_start": "random",
                "released_checkpoint_used": False,
                "processed_windows": 61440000,
                "optimizer_updates": 30000,
                "world_size": 4,
                "effective_batch_size": 2048,
                "optimization_contract": {
                    "optimizer": "Adam",
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.0,
                    "learning_rate": 0.0001,
                    "scheduler": "none",
                    "warmup_windows": 0,
                    "gradient_clipping": False,
                    "gradient_clip_norm": None,
                    "amp": False,
                    "ema_decays": [],
                    "primary_weight_variant": "online",
                },
                "loss_weights": {
                    "fk": 0.3569973401779424,
                    "object_surface": 0.4772322188400037,
                    "velocity": 0.1,
                    "terminal_object_goal": 1.0,
                },
                "loss_routing": {
                    "fk_foot_temporal_routing": True,
                    "foot_joint_indices": [7, 8, 10, 11],
                    "routed_components": ["x", "z"],
                    "velocity_weight": 0.1,
                    "velocity_reduction": "mean_square",
                    "routed_foot_residual_multiplier": 1024.0,
                    "nonrouted_residual_multiplier": 1.0,
                    "weighted_slots": 8,
                    "total_velocity_slots": 87,
                },
                "ema_decays": [],
                "primary_weight_variant": "online",
                "weight_initialization": {
                    "mode": "random",
                    "restored_components": [],
                    "source_checkpoint": None,
                    "initial_model_state_sha256": (
                        "ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e"
                    ),
                },
                "checkpoint_hashes": [{
                    "processed_windows": 61440000,
                    "sha256": checkpoint_sha,
                    "path": str(checkpoint),
                }],
            }
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            diagnostic_path = root / "internal.json"
            diagnostic = {
                "schema_version": 1,
                "status": "completed",
                "run_id": (
                    "p1-hoi-d2y-routed-foot-amplification-internal-s42-20260723"
                ),
                "selection": {
                    "sha256": INTERNAL_SELECTION_SHA256,
                    "sequences": 32,
                    "windows": 96,
                },
                "checkpoints": {
                    "d2x": {"final_sha256": CONTROL_CHECKPOINT_SHA256},
                    "d2y": {"final_sha256": checkpoint_sha},
                },
                "decision": {
                    "timestep_checks": {
                        "249": {"bootstrap_95_ci": [0.001, 0.01], "passed": True},
                        "499": {"bootstrap_95_ci": [0.002, 0.02], "passed": True},
                    },
                    "mechanism_passed": True,
                },
            }
            diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
            args = SimpleNamespace(
                target_checkpoint=checkpoint,
                target_sha256=checkpoint_sha,
                training_metrics=metrics_path,
                internal_diagnostic=diagnostic_path,
                internal_diagnostic_sha256=sha256_file(diagnostic_path),
            )
            result = validate_training_result(args)
            self.assertTrue(result["checks"]["internal_diagnostic_contract"])
            diagnostic["selection"]["sha256"] = "0" * 64
            diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
            args.internal_diagnostic_sha256 = sha256_file(diagnostic_path)
            with self.assertRaisesRegex(ValueError, "internal diagnostic contract"):
                validate_training_result(args)


if __name__ == "__main__":
    unittest.main()
