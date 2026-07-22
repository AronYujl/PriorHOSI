import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from priors.diffusion import GaussianDiffusion  # noqa: E402
from tools.diagnose_hoi_d2w import (  # noqa: E402
    EXPECTED_CHECKPOINTS,
    RUN_ID,
    classify_frontier,
    official_foot_sliding,
    resolved_config,
    sample_with_noise_audit,
    state_dict_sha256,
)


class ZeroModel(torch.nn.Module):
    def forward(self, current, timesteps, text, bps, goals, progress):
        del timesteps, text, bps, goals, progress
        return torch.zeros_like(current)


class D2WTrajectoryParityTests(unittest.TestCase):
    def test_noise_audit_replays_production_sampler_exactly(self):
        diffusion = GaussianDiffusion(500)
        model = ZeroModel()
        fixed = torch.zeros(2, 2, 232)
        text = torch.zeros(2, 512)
        bps = torch.zeros(2, 1024, 3)
        goals = torch.zeros(2, 9)
        progress = torch.zeros(2, 3)
        production_generator = torch.Generator().manual_seed(42)
        audit_generator = torch.Generator().manual_seed(42)
        production = diffusion.sample(
            model, fixed, text, bps, goals, progress,
            generator=production_generator,
        )
        replay, digest = sample_with_noise_audit(
            diffusion, model, fixed, text, bps, goals, progress,
            generator=audit_generator,
        )
        self.assertTrue(torch.equal(production, replay))
        self.assertEqual(len(digest), 64)
        replay_two, digest_two = sample_with_noise_audit(
            diffusion, model, fixed, text, bps, goals, progress,
            generator=torch.Generator().manual_seed(42),
        )
        self.assertTrue(torch.equal(replay, replay_two))
        self.assertEqual(digest, digest_two)

    def test_torch_foot_sliding_matches_official_numpy_formula(self):
        joints = torch.zeros(2, 20, 24, 3, dtype=torch.float64)
        joints[..., 1] = 1.0
        joints[:, :, 7, 1] = 0.06
        joints[:, :, 8, 1] = 0.06
        joints[:, :, 10, 1] = 0.02
        joints[:, :, 11, 1] = 0.02
        joints[0, :, 7, 0] = torch.linspace(0.0, 0.02, 20)
        joints[0, :, 10, 0] = torch.linspace(0.0, 0.02, 20)
        joints[1, :, 8, 2] = torch.linspace(0.0, 0.03, 20)
        joints[1, :, 11, 2] = torch.linspace(0.0, 0.03, 20)
        result = official_foot_sliding(joints)
        self.assertTrue(result["parity_passed"])
        self.assertLessEqual(result["maximum_torch_official_abs"], 1e-9)
        self.assertEqual(result["values"].shape, (2,))

    def test_model_state_hash_is_key_order_stable(self):
        first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
        second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        self.assertEqual(state_dict_sha256(first), state_dict_sha256(second))
        second["b"] = torch.tensor([3.0])
        self.assertNotEqual(state_dict_sha256(first), state_dict_sha256(second))


class D2WGateAndGovernanceTests(unittest.TestCase):
    @staticmethod
    def _comparison(*, foot_lower=0.01, ratio_upper=1.0, contact_lower=-0.01,
                    improvement_lower=0.01):
        return {
            "final_minus_midpoint_fk_foot_sliding": {
                "bootstrap_95_ci": [foot_lower, 0.1],
            },
            "midpoint_over_final_lower_is_better": {
                metric: {"bootstrap_95_ci": [0.8, ratio_upper]}
                for metric in (
                    "fk_mpjpe_cm", "pelvis_goal_error_cm",
                    "object_goal_error_cm", "object_translation_mae_cm",
                )
            },
            "midpoint_minus_final_contact_f1": {
                "bootstrap_95_ci": [contact_lower, 0.02],
            },
            "control_minus_midpoint_improvement": {
                metric: {"bootstrap_95_ci": [improvement_lower, 1.0]}
                for metric in (
                    "fk_mpjpe_cm", "object_goal_error_cm",
                    "object_translation_mae_cm",
                )
            },
        }

    def test_gate_requires_every_preregistered_conjunct(self):
        positive = classify_frontier(self._comparison(), contract_passed=True)
        self.assertEqual(
            positive["classification"], "midbudget-protection-supported-stop",
        )
        self.assertTrue(positive["gate_passed"])
        for mutation in (
            {"foot_lower": 0.0},
            {"ratio_upper": 1.11},
            {"contact_lower": -0.03},
            {"improvement_lower": 0.0},
        ):
            with self.subTest(mutation=mutation):
                result = classify_frontier(
                    self._comparison(**mutation), contract_passed=True,
                )
                self.assertEqual(
                    result["classification"], "midbudget-protection-negative-stop",
                )
                self.assertFalse(result["gate_passed"])
        failed = classify_frontier(self._comparison(), contract_passed=False)
        self.assertEqual(
            failed["classification"], "midbudget-protection-contract-failure-stop",
        )
        self.assertFalse(failed["checkpoint_selected"])
        self.assertFalse(failed["training_authorized"])
        self.assertFalse(failed["consistency_authorized"])

    def test_resolved_config_is_internal_only_and_inference_only(self):
        args = SimpleNamespace(
            checkpoint_control=Path("/tmp/control.pth"),
            checkpoint_midpoint=Path("/tmp/midpoint.pth"),
            checkpoint_final=Path("/tmp/final.pth"),
            python=Path("/home/yujinlun/data/envs/infbagel/bin/python"),
            device="cuda:0",
            output=Path("/tmp/metrics.json"),
        )
        config = resolved_config(args)
        self.assertEqual(config["run_id"], RUN_ID)
        self.assertEqual(config["selection"]["sequences"], 32)
        self.assertEqual(
            config["selection"]["selection_sha256"],
            "30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae",
        )
        self.assertFalse(config["official_test_used"])
        self.assertFalse(config["chois_used"])
        self.assertFalse(config["optimizer_created"])
        self.assertEqual(config["training_updates"], 0)
        self.assertFalse(config["checkpoint_write"])
        self.assertFalse(config["checkpoint_selection"])
        self.assertFalse(config["production_change"])
        self.assertFalse(config["consistency_authorized"])

    def test_checkpoint_hashes_plan_and_registry_are_locked(self):
        self.assertEqual(
            EXPECTED_CHECKPOINTS["control"]["file_sha256"],
            "be8233c0a4c013d973c4140ba5c1f472332f1fdd6be8efa21585deeb250506d3",
        )
        self.assertEqual(
            EXPECTED_CHECKPOINTS["midpoint"]["file_sha256"],
            "efab7f55d6a719ac85659de0aa66c2f94235e1875ae5e6951e9c4334017ee9a3",
        )
        self.assertEqual(
            EXPECTED_CHECKPOINTS["final"]["file_sha256"],
            "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4",
        )
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-W fixed-checkpoint FK foot-sliding frontier", plan)
        self.assertIn("本 subphase 不选择任何 D2-V checkpoint", plan)
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        record = next(
            value for value in records
            if value["experiment_id"]
            == "p1-hoi-d2w-checkpoint-frontier-preregister-s42-20260722"
        )
        self.assertTrue(record["config"]["diagnostic_authorized"])
        self.assertFalse(record["config"]["training_authorized"])
        self.assertFalse(record["config"]["official_test_used"])
        self.assertFalse(record["config"]["checkpoint_selection"])
        self.assertFalse(record["config"]["consistency_authorized"])


if __name__ == "__main__":
    unittest.main()
