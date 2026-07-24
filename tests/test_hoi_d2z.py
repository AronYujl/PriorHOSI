import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch3d import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.d2z import d2z_hoi_training_losses, d2z_velocity_loss  # noqa: E402
from priors.losses import (  # noqa: E402
    D2X_FOOT_XZ_VELOCITY_SLOTS,
    _velocity_loss,
    hoi_training_losses,
)
from priors.near_ground import (  # noqa: E402
    D2Z_FLOOR_ALGORITHM,
    D2Z_FLOOR_ALGORITHM_FILE_SHA256,
    D2Z_GATE_AUDIT_RUN_ID,
    D2Z_GATE_AUDIT_SCHEMA,
    immutable_gt_near_ground_gate,
    load_gate_audit_floors,
)
from train_hoi_prior import (  # noqa: E402
    _locked_loss_weights,
    _loss_routing_contract,
    _resume_contract,
    _validate_d2z_contract,
    _validate_fk_foot_temporal_routing_mode,
)
from tools.audit_hoi_d2z_gate import (  # noqa: E402
    EXPECTED_SELECTION_COUNTS,
    RUN_ID as AUDIT_RUN_ID,
    resolved_config as audit_resolved_config,
)
from tools.diagnose_hoi_d2z import (  # noqa: E402
    RUN_ID as INTERNAL_RUN_ID,
    _masked_per_sequence,
    diagnostic_summary,
)
from tools.run_hoi_d2z_evaluation import (  # noqa: E402
    RUN_ID as EVALUATION_RUN_ID,
    classify,
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


def bind_audit(cfg, directory: Path):
    path = directory / "gate_audit.json"
    path.write_text('{"test":true}\n', encoding="utf-8")
    cfg.d2z_gate_audit_path = str(path)
    cfg.d2z_gate_audit_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


class D2ZGateAndLossTests(unittest.TestCase):
    @staticmethod
    def full_loss_inputs():
        prediction = torch.zeros(1, 16, 232, requires_grad=True)
        target = torch.zeros_like(prediction)
        identity_6d = transforms.matrix_to_rotation_6d(torch.eye(3))
        prediction.data[..., 84:216] = identity_6d.reshape(1, 1, 1, 6).expand(
            1, 16, 22, 6,
        ).reshape(1, 16, 132)
        target.data[..., 84:216] = prediction.data[..., 84:216]
        identity = torch.eye(3).reshape(1, 1, 9).expand(1, 16, 9)
        prediction.data[..., 219:228] = identity
        target.data[..., 219:228] = identity
        for frame in range(2, 16):
            target.data[0, frame, 7 * 3] = 0.02 * frame
            target.data[0, frame, 7 * 3 + 2] = -0.01 * frame
        return (
            prediction,
            target,
            torch.zeros(1, 9),
            torch.zeros(1, 24, 3),
            torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long),
            torch.full((3,), -2.0),
            torch.full((3,), 2.0),
            torch.full((3,), -2.0),
            torch.full((3,), 2.0),
            torch.zeros(1, dtype=torch.bool),
            torch.randn(1, 100, 3, generator=torch.Generator().manual_seed(42)),
            torch.eye(3).reshape(1, 3, 3),
            torch.eye(3).reshape(1, 3, 3),
        )

    def test_gate_uses_previous_sampled_frame_and_strict_thresholds(self):
        joints = np.ones((16, 28, 3), dtype=np.float32)
        joints[..., 1] = 1.0
        joints[1, 7, 1] = 0.079
        joints[1, 8, 1] = 0.08
        joints[1, 10, 1] = 0.039
        joints[1, 11, 1] = 0.04
        # Current residual frame height must not affect residual 0's gate.
        joints[2, [7, 8, 10, 11], 1] = -100.0
        gate = immutable_gt_near_ground_gate(joints, 0.0)
        self.assertEqual(gate.shape, (14, 4))
        self.assertEqual(gate.dtype, np.bool_)
        self.assertEqual(gate[0].tolist(), [True, False, True, False])
        self.assertTrue(gate[1].all())
        with self.assertRaisesRegex(ValueError, r"\[16,28,3\]"):
            immutable_gt_near_ground_gate(joints[:, :24], 0.0)

    def test_all_false_gate_is_exact_d2x_and_all_true_is_exact_d2y(self):
        generator = torch.Generator().manual_seed(42)
        prediction = torch.randn(2, 14, 87, generator=generator)
        target = torch.randn(2, 14, 87, generator=generator)
        d2x = F.mse_loss(prediction, target)
        d2y = _velocity_loss(
            prediction,
            target,
            fk_foot_temporal_routing=True,
            routed_foot_residual_multiplier=1024.0,
        )
        all_false = d2z_velocity_loss(
            prediction,
            target,
            torch.zeros(2, 14, 4, dtype=torch.bool),
        )
        all_true = d2z_velocity_loss(
            prediction,
            target,
            torch.ones(2, 14, 4, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(all_false, d2x))
        self.assertTrue(torch.equal(all_true, d2y))

    def test_full_loss_wrapper_changes_only_velocity_reduction(self):
        inputs = self.full_loss_inputs()
        weights = {
            "fk_weight": 0.3569973401779424,
            "object_surface_weight": 0.4772322188400037,
            "velocity_weight": 0.1,
            "goal_weight": 1.0,
        }
        d2x = hoi_training_losses(
            *inputs,
            **weights,
            fk_foot_temporal_routing=True,
            routed_foot_residual_multiplier=1.0,
        )
        d2y = hoi_training_losses(
            *inputs,
            **weights,
            fk_foot_temporal_routing=True,
            routed_foot_residual_multiplier=1024.0,
        )
        d2z_false = d2z_hoi_training_losses(
            *inputs,
            torch.zeros(1, 14, 4, dtype=torch.bool),
            **weights,
        )
        d2z_true = d2z_hoi_training_losses(
            *inputs,
            torch.ones(1, 14, 4, dtype=torch.bool),
            **weights,
        )
        self.assertTrue(torch.equal(d2z_false["velocity"], d2x["velocity"]))
        self.assertTrue(torch.equal(d2z_false["total"], d2x["total"]))
        self.assertTrue(torch.equal(d2z_true["velocity"], d2y["velocity"]))
        self.assertTrue(torch.equal(d2z_true["total"], d2y["total"]))
        for name in d2x:
            if name not in {"velocity", "total"}:
                self.assertTrue(torch.equal(d2z_false[name], d2x[name]), name)
                self.assertTrue(torch.equal(d2z_true[name], d2x[name]), name)

    def test_only_active_joint_xz_slots_receive_1024(self):
        prediction = torch.ones(1, 14, 87, requires_grad=True)
        target = torch.zeros_like(prediction)
        gate = torch.zeros(1, 14, 4, dtype=torch.bool)
        gate[0, 0, 0] = True
        actual = d2z_velocity_loss(
            prediction,
            target,
            gate,
        )
        total = prediction.numel()
        self.assertAlmostEqual(
            actual.item(), (total + 2 * 1023.0) / total, places=6,
        )
        actual.backward()
        active_x = D2X_FOOT_XZ_VELOCITY_SLOTS[0]
        active_z = D2X_FOOT_XZ_VELOCITY_SLOTS[1]
        inactive_routed = D2X_FOOT_XZ_VELOCITY_SLOTS[2]
        reference = prediction.grad[0, 0, 0].item()
        self.assertAlmostEqual(
            prediction.grad[0, 0, active_x].item() / reference, 1024.0, places=4,
        )
        self.assertAlmostEqual(
            prediction.grad[0, 0, active_z].item() / reference, 1024.0, places=4,
        )
        self.assertAlmostEqual(
            prediction.grad[0, 0, inactive_routed].item() / reference, 1.0, places=6,
        )

    def test_gate_loss_fails_closed(self):
        residual = torch.zeros(1, 14, 87)
        valid = torch.zeros(1, 14, 4, dtype=torch.bool)
        for gate in (None, valid.float(), valid[..., :3]):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                d2z_velocity_loss(residual, residual, gate)
        with self.assertRaisesRegex(ValueError, "87"):
            d2z_velocity_loss(residual[..., :-1], residual[..., :-1], valid)


class D2ZAuditAndConfigTests(unittest.TestCase):
    @staticmethod
    def audit_value():
        return {
            "schema": D2Z_GATE_AUDIT_SCHEMA,
            "run_id": D2Z_GATE_AUDIT_RUN_ID,
            "seed": 42,
            "floor_algorithm": D2Z_FLOOR_ALGORITHM,
            "floor_algorithm_file_sha256": D2Z_FLOOR_ALGORITHM_FILE_SHA256,
            "gate_previous_frame": "immutable_gt",
            "thresholds_m": {"7": 0.08, "8": 0.08, "10": 0.04, "11": 0.04},
            "split_sha256": "a" * 64,
            "partitions": {
                "train": {
                    "sequence_indices": [1, 2],
                    "floors_m": {"1": 0.01, "2": 0.02},
                    "nonfinite_floor_count": 0,
                },
            },
        }

    def test_audit_loader_binds_hash_schema_split_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(
                json.dumps(self.audit_value(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            floors = load_gate_audit_floors(
                str(path),
                digest,
                partition="train",
                expected_sequence_indices=[1, 2],
                expected_split_sha256="a" * 64,
            )
            self.assertEqual(dict(floors), {1: 0.01, 2: 0.02})
            for mutation in (
                ("sha", "b" * 64),
                ("coverage", [1]),
                ("split", "b" * 64),
            ):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_gate_audit_floors(
                        str(path),
                        mutation[1] if mutation[0] == "sha" else digest,
                        partition="train",
                        expected_sequence_indices=(
                            mutation[1] if mutation[0] == "coverage" else [1, 2]
                        ),
                        expected_split_sha256=(
                            mutation[1] if mutation[0] == "split" else "a" * 64
                        ),
                    )

    def test_d2z_config_is_single_gating_delta_from_d2y_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            d2z = merged_config("config_train_hoi_prior_d2z")
            bind_audit(d2z, Path(temporary))
            _validate_fk_foot_temporal_routing_mode(d2z)
            _validate_d2z_contract(d2z, 4)
            d2y = merged_config("config_train_hoi_prior_d2y")
            ignored = {
                "mode", "subphase", "run_id",
                "d2y_routed_foot_amplification",
                "d2z_immutable_gt_near_ground_gating",
                "immutable_gt_near_ground_gating",
                "d2z_gate_audit_path", "d2z_gate_audit_sha256",
            }
            y_plain = OmegaConf.to_container(d2y, resolve=False)
            z_plain = OmegaConf.to_container(d2z, resolve=False)
            self.assertEqual(
                {key: value for key, value in y_plain.items() if key not in ignored},
                {key: value for key, value in z_plain.items() if key not in ignored},
            )
            self.assertEqual(_locked_loss_weights(d2z), _locked_loss_weights(d2y))
            for field, value in (
                ("immutable_gt_near_ground_gating", False),
                ("routed_foot_residual_multiplier", 512.0),
                ("resume_checkpoint", "/tmp/d2y.pth"),
                ("d2z_gate_audit_sha256", "0" * 64),
            ):
                mutated = merged_config("config_train_hoi_prior_d2z")
                bind_audit(mutated, Path(temporary))
                setattr(mutated, field, value)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "D2-Z"):
                    _validate_d2z_contract(mutated, 4)

    def test_training_config_binds_completed_r1_audit(self):
        raw = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2z.yaml")
        unresolved = OmegaConf.to_container(raw, resolve=False)
        self.assertEqual(
            unresolved["d2z_gate_audit_path"],
            "${repo_root}/results/experiments/"
            "p1-hoi-d2z-gate-audit-r1-s42-20260724/gate_audit.json",
        )
        self.assertEqual(
            raw.d2z_gate_audit_sha256,
            "d56f1cbc5297b82d768cd396ab1a49c6e33d4101d156c0375501bf32ae055faa",
        )

    def test_resume_and_metrics_bind_gate_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = merged_config("config_train_hoi_prior_d2z")
            path = bind_audit(cfg, Path(temporary))
            resume = _resume_contract(cfg)
            self.assertTrue(resume["d2z_immutable_gt_near_ground_gating"])
            self.assertEqual(resume["routed_foot_residual_multiplier"], 1024.0)
            self.assertEqual(resume["d2z_gate_audit_sha256"], cfg.d2z_gate_audit_sha256)
            routing = _loss_routing_contract(cfg)
            self.assertEqual(routing["gate_audit_path"], str(path.resolve()))
            self.assertEqual(routing["gate_audit_sha256"], cfg.d2z_gate_audit_sha256)
            self.assertEqual(routing["gate_shape_per_window"], [14, 4])
            self.assertTrue(routing["gate_stop_gradient"])

    def test_cpu_audit_resolved_contract_forbids_gpu_and_training(self):
        args = type("Args", (), {
            "python": Path("/data/yujinlun/anaconda3/envs/infbagel/bin/python"),
            "output": Path("/tmp/audit.json"),
        })()
        config = audit_resolved_config(args)
        self.assertEqual(AUDIT_RUN_ID, D2Z_GATE_AUDIT_RUN_ID)
        self.assertTrue(config["execution"]["cpu_only"])
        self.assertEqual(config["execution"]["checkpoint_loads"], 0)
        self.assertEqual(config["execution"]["training_updates"], 0)
        self.assertEqual(
            config["sealed_selection"]["expected_active_counts"],
            EXPECTED_SELECTION_COUNTS,
        )
        self.assertFalse(config["checkpoint_selection"])
        self.assertFalse(config["consistency_authorized"])


class D2ZDiagnosticAndEvaluatorTests(unittest.TestCase):
    def test_masked_per_sequence_reports_exact_active_and_inactive_mse(self):
        error = torch.ones(96, 14, 8)
        mask = torch.zeros_like(error, dtype=torch.bool)
        mask[..., ::2] = True
        error[..., 1::2] = 4.0
        active, active_counts = _masked_per_sequence(error, mask)
        inactive, inactive_counts = _masked_per_sequence(error, ~mask)
        torch.testing.assert_close(active, torch.ones(32))
        torch.testing.assert_close(inactive, torch.full((32,), 4.0))
        self.assertTrue(torch.all(active_counts == inactive_counts))

    @staticmethod
    def diagnostic_results():
        item = {
            "active_routed_residual_mse_by_sequence": [1.0] * 32,
            "inactive_routed_residual_mse_by_sequence": [1.0] * 32,
            "active_routed_residual_rms": 1.0,
            "inactive_routed_residual_rms": 1.0,
            "gate_occupancy": 0.5,
            "uniform_vs_gated": {
                "gated_gradient_norm": 1.0,
                "uniform_gradient_norm": 2.0,
            },
        }
        return {
            expert: {
                stratum: {
                    "timesteps": {
                        str(timestep): item
                        for timestep in (0, 249, 499)
                    },
                }
                for stratum in ("early", "mid", "final")
            }
            for expert in ("d2x", "d2y", "d2z")
        }

    def test_internal_summary_is_diagnostic_only(self):
        summary = diagnostic_summary(self.diagnostic_results())
        self.assertTrue(summary["contract_passed"])
        self.assertFalse(summary["selection_use"])
        self.assertFalse(summary["checkpoint_selected"])
        self.assertFalse(summary["consistency_authorized"])

    @staticmethod
    def comparison(*, foot_lower=0.01, ratio_upper=1.0, contact_lower=-0.01):
        return {
            "penetration_mask_contract": {"passed": True},
            "control_minus_target_foot_sliding": {
                "bootstrap_95_ci": [foot_lower, 0.1],
            },
            "target_over_control_protection": {
                metric: {"bootstrap_95_ci": [0.9, ratio_upper]}
                for metric in (
                    "mpjpe", "end_obj_trans_err", "xy_points_err",
                    "obj_trans_dist", "hand_pen_loss_omomo",
                    "human_pen_loss_infbagel",
                )
            },
            "target_minus_control_contact_f1": {
                "bootstrap_95_ci": [contact_lower, 0.01],
            },
        }

    @staticmethod
    def baseline_ratios(value=1.0):
        return {
            metric: value
            for metric in (
                "mpjpe", "end_obj_trans_err", "xy_points_err",
                "obj_trans_dist", "foot_sliding",
            )
        }

    def test_evaluator_classifies_every_preregistered_branch(self):
        target = {"contact_f1": 0.65}
        cases = (
            (self.comparison(), True, "immutable-gt-near-ground-positive-candidate-stop"),
            (
                self.comparison(foot_lower=-0.01), True,
                "immutable-gt-near-ground-transfer-negative-stop",
            ),
            (
                self.comparison(ratio_upper=1.11), True,
                "immutable-gt-near-ground-conflict-negative-stop",
            ),
            (
                self.comparison(foot_lower=-0.01, ratio_upper=1.11), True,
                "immutable-gt-near-ground-joint-negative-stop",
            ),
            (
                self.comparison(), False,
                "immutable-gt-near-ground-contract-failure-stop",
            ),
        )
        for comparison, contract, expected in cases:
            with self.subTest(expected=expected):
                result = classify(
                    comparison,
                    target,
                    self.baseline_ratios(),
                    contract_passed=contract,
                )
                self.assertEqual(result["classification"], expected)
                self.assertFalse(result["checkpoint_selected"])
                self.assertFalse(result["consistency_authorized"])
        ineffective = classify(
            self.comparison(),
            target,
            self.baseline_ratios(2.1),
            contract_passed=True,
        )
        self.assertEqual(
            ineffective["classification"],
            "immutable-gt-near-ground-positive-but-not-effective-stop",
        )


class D2ZGovernanceTests(unittest.TestCase):
    def test_plan_registry_and_lifecycle_ids_are_bound_without_gpu_authority(self):
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-Z immutable-GT near-ground routed amplification", plan)
        self.assertIn("p1-hoi-d2z-gate-audit-s42-20260724", plan)
        self.assertIn("p1-hoi-d2z-gate-audit-r1-s42-20260724", plan)
        self.assertIn(
            "p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724", plan,
        )
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        preregistration = next(
            value for value in records
            if value["experiment_id"]
            == "p1-hoi-d2z-immutable-gt-near-ground-gating-preregister-s42-20260724"
        )
        identity = next(
            value for value in records
            if value["experiment_id"]
            == "p1-hoi-d2z-implementation-lifecycle-binding-s42-20260724"
        )
        self.assertEqual(
            preregistration["config"]["manipulated_factor"]["active_multiplier"],
            1024.0,
        )
        self.assertEqual(
            preregistration["config"]["manipulated_factor"]["gate_source"],
            "immutable GT previous sampled frame",
        )
        self.assertTrue(identity["config"]["authorized"]["implementation"])
        self.assertTrue(identity["config"]["not_authorized"]["gpu_smoke"])
        self.assertTrue(identity["config"]["not_authorized"]["training"])
        self.assertEqual(INTERNAL_RUN_ID, identity["config"]["identities"]["internal"]["run_id"])
        self.assertEqual(
            EVALUATION_RUN_ID,
            identity["config"]["identities"]["evaluation"]["run_id"],
        )


if __name__ == "__main__":
    unittest.main()
