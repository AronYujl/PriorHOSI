import hashlib
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
from pytorch3d import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from priors.losses import D2X_FOOT_JOINTS, _fk_positions, _velocity_residuals
from train_hoi_prior import (
    _build_optimizer,
    _build_scheduler,
    _optimization_contract,
    _validate_d2v_contract,
    _validate_d2x_contract,
    _validate_d2x_execution_host,
    _validate_fk_foot_temporal_routing_mode,
)
from tools.run_hoi_d2x_evaluation import (
    BASELINE_KEYS,
    CONTROL_AGGREGATE_SHA256,
    CONTROL_CHECKPOINT_SHA256,
    CONTROL_PER_SEQUENCE_SHA256,
    RUN_ID as EVAL_RUN_ID,
    classify as classify_evaluation,
    compare_records,
    validate_training_result,
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


class D2XConfigAndContractTests(unittest.TestCase):
    def test_exact_config_changes_only_registered_routing_and_mode_from_d2v(self):
        d2x = merged_config("config_train_hoi_prior_d2x")
        d2v = merged_config("config_train_hoi_prior_d2v")
        _validate_d2x_contract(d2x, 4)
        _validate_d2v_contract(d2v, 4)
        ignored = {
            "mode", "subphase", "run_id", "d2v_balanced_long_budget",
            "d2x_fk_foot_temporal_routing", "fk_foot_temporal_routing",
        }
        d2x_plain = OmegaConf.to_container(d2x, resolve=False)
        d2v_plain = OmegaConf.to_container(d2v, resolve=False)
        self.assertEqual(
            {key: value for key, value in d2x_plain.items() if key not in ignored},
            {key: value for key, value in d2v_plain.items() if key not in ignored},
        )
        self.assertTrue(d2x.d2x_fk_foot_temporal_routing)
        self.assertTrue(d2x.fk_foot_temporal_routing)

    def test_optimizer_and_budget_are_identical_to_d2v(self):
        cfg = merged_config("config_train_hoi_prior_d2x")
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = _build_optimizer(cfg, [parameter])
        scheduler = _build_scheduler(cfg, optimizer, 30000, 0)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertIsNone(scheduler)
        self.assertEqual(cfg.max_processed_windows, 61440000)
        self.assertEqual(cfg.max_processed_windows // cfg.effective_batch_size, 30000)
        self.assertEqual(
            _optimization_contract(cfg),
            {
                "optimizer": "Adam", "betas": [0.9, 0.999],
                "weight_decay": 0.0, "learning_rate": 0.0001,
                "scheduler": "none", "warmup_windows": 0,
                "gradient_clipping": False, "gradient_clip_norm": None,
                "amp": False, "ema_decays": [], "primary_weight_variant": "online",
            },
        )

    def test_contract_fails_closed_on_routing_or_provenance_mutation(self):
        mutations = {
            "fk_foot_temporal_routing": False,
            "velocity_weight": 0.2,
            "learning_rate": 0.0003,
            "resume_checkpoint": "/tmp/forbidden.pth",
            "weight_init_checkpoint": "/tmp/d2v.pth",
            "d2m_candidate": "balanced",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                cfg = merged_config("config_train_hoi_prior_d2x")
                setattr(cfg, field, value)
                with self.assertRaisesRegex(ValueError, "D2-X"):
                    _validate_d2x_contract(cfg, 4)

    def test_d2v_rejects_d2x_mode_or_routing(self):
        cfg = merged_config("config_train_hoi_prior_d2v")
        cfg.d2x_fk_foot_temporal_routing = True
        cfg.fk_foot_temporal_routing = True
        with self.assertRaisesRegex(ValueError, "D2-V"):
            _validate_d2v_contract(cfg, 4)

    def test_routing_cannot_be_enabled_outside_d2x_mode(self):
        cfg = merged_config("config_train_hoi_prior_d2v")
        cfg.fk_foot_temporal_routing = True
        with self.assertRaisesRegex(ValueError, "registered D2-X"):
            _validate_fk_foot_temporal_routing_mode(cfg)
        _validate_fk_foot_temporal_routing_mode(
            merged_config("config_train_hoi_prior_d2x")
        )

    def test_worker_host_and_python_are_fail_closed(self):
        cfg = merged_config("config_train_hoi_prior_d2x")
        with mock.patch("socket.gethostname", return_value="ubuntu"):
            with self.assertRaisesRegex(RuntimeError, "infbagel-4gpu/node01"):
                _validate_d2x_execution_host(cfg)
        environment = {
            "INFBAGEL_WORKER_EXPERT": "hoi",
            "INFBAGEL_PYTHON": "/home/yujinlun/data/envs/infbagel/bin/python",
        }
        with mock.patch("socket.gethostname", return_value="node01"), mock.patch.dict(
            os.environ, environment, clear=False,
        ), mock.patch("train_hoi_prior.sys.executable", environment["INFBAGEL_PYTHON"]):
            _validate_d2x_execution_host(cfg)


class D2XVelocityRoutingTests(unittest.TestCase):
    @staticmethod
    def tensors():
        generator = torch.Generator().manual_seed(42)
        prediction = torch.randn(2, 16, 232, generator=generator)
        target = torch.randn(2, 16, 232, generator=generator)
        fk = torch.randn(2, 16, 24, 3, generator=generator)
        minimum = torch.tensor([-2.0, -4.0, -6.0])
        maximum = torch.tensor([2.0, 4.0, 6.0])
        return prediction, target, fk, minimum, maximum

    def test_disabled_routing_is_exact_original_formula(self):
        prediction, target, fk, minimum, maximum = self.tensors()
        actual_prediction, actual_target = _velocity_residuals(
            prediction, target, fk, minimum, maximum,
            fk_foot_temporal_routing=False,
        )
        channels = torch.cat((prediction[..., :84], prediction[..., 216:219]), dim=-1)
        target_channels = torch.cat((target[..., :84], target[..., 216:219]), dim=-1)
        previous = torch.cat((target_channels[:, 1:2], channels[:, 2:-1]), dim=1)
        expected_prediction = channels[:, 2:] - previous
        expected_target = target_channels[:, 2:] - target_channels[:, 1:-1]
        torch.testing.assert_close(actual_prediction, expected_prediction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_target, expected_target, rtol=0.0, atol=0.0)

    def test_only_registered_foot_xz_slots_change(self):
        prediction, target, fk, minimum, maximum = self.tensors()
        direct_prediction, direct_target = _velocity_residuals(
            prediction, target, fk, minimum, maximum,
            fk_foot_temporal_routing=False,
        )
        routed_prediction, routed_target = _velocity_residuals(
            prediction, target, fk, minimum, maximum,
            fk_foot_temporal_routing=True,
        )
        routed_indices = sorted(
            joint * 3 + component
            for joint in D2X_FOOT_JOINTS
            for component in (0, 2)
        )
        protected_indices = [
            index for index in range(direct_prediction.shape[-1])
            if index not in routed_indices
        ]
        torch.testing.assert_close(
            routed_prediction[..., protected_indices],
            direct_prediction[..., protected_indices],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(routed_target, direct_target, rtol=0.0, atol=0.0)
        self.assertTrue(torch.any(
            routed_prediction[..., routed_indices] != direct_prediction[..., routed_indices]
        ))

    def test_first_future_previous_is_immutable_gt_history(self):
        prediction, target, fk, minimum, maximum = self.tensors()
        changed_history_fk = fk.clone()
        changed_history_fk[:, 1, D2X_FOOT_JOINTS, :] += 1000.0
        original, _ = _velocity_residuals(
            prediction, target, fk, minimum, maximum,
            fk_foot_temporal_routing=True,
        )
        changed, _ = _velocity_residuals(
            prediction, target, changed_history_fk, minimum, maximum,
            fk_foot_temporal_routing=True,
        )
        torch.testing.assert_close(original, changed, rtol=0.0, atol=0.0)
        normalized_fk = (
            2.0 * (fk - minimum.reshape(1, 1, 1, 3))
            / (maximum - minimum).reshape(1, 1, 1, 3)
            - 1.0
        )
        for joint in D2X_FOOT_JOINTS:
            for component in (0, 2):
                channel = joint * 3 + component
                expected = normalized_fk[:, 2, joint, component] - target[:, 1, channel]
                torch.testing.assert_close(original[:, 0, channel], expected)

    def test_velocity_gradient_moves_from_direct_foot_xz_to_root_and_rotations(self):
        prediction = torch.zeros(1, 16, 232)
        identity_6d = transforms.matrix_to_rotation_6d(torch.eye(3)).reshape(1, 1, 1, 6)
        prediction[..., 84:216] = identity_6d.expand(1, 16, 22, 6).reshape(1, 16, 132)
        prediction.requires_grad_()
        target = torch.zeros_like(prediction)
        for frame in range(2, 16):
            for joint in D2X_FOOT_JOINTS:
                target[0, frame, joint * 3] = 0.02 * (frame - 1)
                target[0, frame, joint * 3 + 2] = -0.01 * (frame - 1)
        direct_positions = prediction[..., :84].reshape(1, 16, 28, 3)
        rotations = transforms.rotation_6d_to_matrix(
            prediction[..., 84:216].reshape(1, 16, 22, 6)
        )
        offsets = torch.zeros(1, 24, 3)
        offsets[:, 1:, :] = torch.tensor([0.5, -1.0, 0.3])
        parents = torch.zeros(24, dtype=torch.long)
        parents[0] = -1
        fk = _fk_positions(direct_positions[..., 0, :], rotations, offsets, parents)
        predicted_residual, target_residual = _velocity_residuals(
            prediction,
            target,
            fk,
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
            fk_foot_temporal_routing=True,
        )
        F.mse_loss(predicted_residual, target_residual).backward()
        self.assertGreater(float(prediction.grad[..., :3].abs().sum()), 0.0)
        self.assertGreater(float(prediction.grad[..., 84:216].abs().sum()), 0.0)
        for joint in D2X_FOOT_JOINTS:
            for component in (0, 2):
                self.assertEqual(
                    float(prediction.grad[..., joint * 3 + component].abs().sum()),
                    0.0,
                )


class D2XGovernanceTests(unittest.TestCase):
    def test_plan_and_registry_lock_training_and_evaluation(self):
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-X FK-foot temporal gradient-routing screen", plan)
        self.assertIn("p1-hoi-d2x-native-eval-s42-20260723", plan)
        records = [
            json.loads(line) for line in
            (ROOT / "experiments/registry.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        training = next(
            item for item in records
            if item["experiment_id"] ==
            "p1-hoi-d2x-fk-foot-temporal-routing-preregister-s42-20260723"
        )
        evaluation = next(
            item for item in records
            if item["experiment_id"] ==
            "p1-hoi-d2x-native-eval-preregister-s42-20260723"
        )
        factor = training["config"]["manipulated_factor"]
        self.assertEqual(factor["replaced_velocity_slots"], 8)
        self.assertEqual(factor["velocity_weight_unchanged"], 0.1)
        self.assertFalse(factor["new_tunable_weight"])
        self.assertTrue(training["config"]["fixed"]["random_initialization"])
        self.assertFalse(training["config"]["consistency_authorized"])
        self.assertFalse(training["config"]["checkpoint_selection"])
        self.assertTrue(evaluation["config"]["control"]["reused_without_regeneration"])
        self.assertFalse(evaluation["config"]["consistency_authorized"])
        self.assertFalse(evaluation["config"]["checkpoint_selection"])


class D2XEvaluationTests(unittest.TestCase):
    @staticmethod
    def records(*, foot=0.30, contact=0.65, ratio=1.0):
        return {
            f"sequence-{index:03d}": {
                "mpjpe": 10.0 * ratio,
                "end_obj_trans_err": 4.0 * ratio,
                "pelvis_goal_error_cm": 3.0 * ratio,
                "obj_trans_dist": 12.0 * ratio,
                "foot_sliding": foot,
                "contact_f1": contact,
                "hand_pen_loss_omomo": 0.2 * ratio,
                "human_pen_loss_infbagel": 3.0 * ratio,
            }
            for index in range(32)
        }

    @staticmethod
    def compare(control, target):
        sequence_ids = sorted(control)
        digest = hashlib.sha256(
            ("".join(f"{sequence}\n" for sequence in sequence_ids)).encode("utf-8")
        ).hexdigest()
        with mock.patch(
            "tools.run_hoi_d2x_evaluation.PENETRATION_SEQUENCE_COUNT",
            len(sequence_ids),
        ), mock.patch(
            "tools.run_hoi_d2x_evaluation.PENETRATION_SEQUENCE_IDS_SHA256",
            digest,
        ):
            return compare_records(control, target)

    def test_mechanism_and_absolute_gate_classifications(self):
        control = self.records(foot=0.40, contact=0.66)
        target = self.records(foot=0.30, contact=0.65)
        comparison = self.compare(control, target)
        target_metrics = {key: 1.0 for key in BASELINE_KEYS}
        target_metrics["contact_f1"] = 0.65
        ratios = {
            "mpjpe": 1.0,
            "end_obj_trans_err": 1.0,
            "xy_points_err": 1.0,
            "obj_trans_dist": 1.0,
            "foot_sliding": 1.0,
        }
        result = classify_evaluation(
            comparison, target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(
            result["classification"], "fk-foot-temporal-routing-positive-candidate-stop",
        )
        self.assertTrue(result["mechanism_passed"])
        self.assertFalse(result["checkpoint_selected"])

        negative = classify_evaluation(
            self.compare(control, self.records(foot=0.41, contact=0.65)),
            target_metrics,
            ratios,
            contract_passed=True,
        )
        self.assertEqual(
            negative["classification"], "fk-foot-temporal-routing-negative-stop",
        )

        ratios["mpjpe"] = 1.31
        ineffective = classify_evaluation(
            comparison, target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(
            ineffective["classification"],
            "fk-foot-temporal-routing-positive-but-not-effective-stop",
        )

        failed = classify_evaluation(
            comparison, target_metrics, ratios, contract_passed=False,
        )
        self.assertEqual(
            failed["classification"], "fk-foot-temporal-routing-contract-failure-stop",
        )

    def test_protection_ratio_gate_includes_penetration(self):
        control = self.records(foot=0.40)
        target = self.records(foot=0.30, ratio=1.11)
        result = classify_evaluation(
            self.compare(control, target),
            {**{key: 1.0 for key in BASELINE_KEYS}, "contact_f1": 0.65},
            {
                "mpjpe": 1.0,
                "end_obj_trans_err": 1.0,
                "xy_points_err": 1.0,
                "obj_trans_dist": 1.0,
                "foot_sliding": 1.0,
            },
            contract_passed=True,
        )
        self.assertFalse(result["mechanism_checks"]["hand_pen_loss_omomo_preserved"])
        self.assertFalse(result["mechanism_checks"]["human_pen_loss_infbagel_preserved"])
        self.assertEqual(
            result["classification"], "fk-foot-temporal-routing-negative-stop",
        )

    def test_penetration_mask_mismatch_is_contract_failure(self):
        control = self.records(foot=0.40)
        target = self.records(foot=0.30)
        target["sequence-000"]["hand_pen_loss_omomo"] = None
        comparison = self.compare(control, target)
        result = classify_evaluation(
            comparison,
            {**{key: 1.0 for key in BASELINE_KEYS}, "contact_f1": 0.65},
            {
                "mpjpe": 1.0,
                "end_obj_trans_err": 1.0,
                "xy_points_err": 1.0,
                "obj_trans_dist": 1.0,
                "foot_sliding": 1.0,
            },
            contract_passed=True,
        )
        self.assertFalse(comparison["penetration_mask_contract"]["passed"])
        self.assertEqual(
            result["classification"], "fk-foot-temporal-routing-contract-failure-stop",
        )

    def test_training_metrics_bind_final_random_checkpoint_and_routing(self):
        target_sha = "a" * 64
        checkpoint = Path(
            "/tmp/p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723_"
            "windows061440000.pth"
        )
        metrics = {
            "status": "stable",
            "run_id": "p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723",
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
            },
            "ema_decays": [],
            "primary_weight_variant": "online",
            "weight_initialization": {
                "mode": "random",
                "source_checkpoint": None,
                "restored_components": [],
            },
            "checkpoint_hashes": [{
                "path": str(checkpoint),
                "sha256": target_sha,
                "processed_windows": 61440000,
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            args = SimpleNamespace(
                target_checkpoint=checkpoint,
                target_sha256=target_sha,
                training_metrics=path,
            )
            self.assertTrue(all(validate_training_result(args)["checks"].values()))
            metrics["loss_routing"]["fk_foot_temporal_routing"] = False
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "loss_routing"):
                validate_training_result(args)

    def test_evaluator_ids_and_control_hashes_are_locked(self):
        self.assertEqual(EVAL_RUN_ID, "p1-hoi-d2x-native-eval-s42-20260723")
        self.assertEqual(
            CONTROL_CHECKPOINT_SHA256,
            "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4",
        )
        self.assertEqual(
            CONTROL_AGGREGATE_SHA256,
            "21f6bb27fe8d38a5203c2e40dee02815470bc40b638ee3a616101faae5cf8f0e",
        )
        self.assertEqual(
            CONTROL_PER_SEQUENCE_SHA256,
            "4d147ef0a76977146639bab4260c6f7c5c2f96d9b253fb52ad968269f649ce1a",
        )


if __name__ == "__main__":
    unittest.main()
