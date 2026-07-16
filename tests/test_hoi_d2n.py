import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_hoi_d2n import (
    BASELINE_SHA256,
    BOOTSTRAP_REPLICATES,
    CANDIDATES,
    EVAL_CONFIG_SHA256,
    EVAL_METRICS_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    GATE_LOWER_METRICS,
    RUN_ID,
    SAMPLE_COUNT,
    TEST_SCRIPT_SHA256,
    candidate_overrides,
    comparisons,
    paired_difference,
    paired_ratio,
    transfer_gate,
)


class D2NConfigTests(unittest.TestCase):
    def test_locked_hashes_and_checkpoint_order(self):
        self.assertEqual(CANDIDATES, ("source", "current", "balanced"))
        self.assertEqual(
            EXPECTED_CHECKPOINT_SHA256,
            {
                "source": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
                "current": "76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f",
                "balanced": "ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8",
            },
        )
        self.assertEqual(
            TEST_SCRIPT_SHA256,
            "22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524",
        )
        self.assertEqual(
            EVAL_METRICS_SHA256,
            "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547",
        )
        self.assertEqual(
            EVAL_CONFIG_SHA256,
            "89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73",
        )
        self.assertEqual(
            BASELINE_SHA256,
            "76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6",
        )

    def test_overrides_force_online_full_author_native_without_chois(self):
        with tempfile.TemporaryDirectory() as directory:
            overrides = candidate_overrides(
                "balanced",
                Path(directory) / "balanced.pth",
                EXPECTED_CHECKPOINT_SHA256["balanced"],
                Path(directory) / "run",
                "cuda:0",
            )
        text = "\n".join(overrides)
        self.assertIn("checkpoint_weight_variant=online", text)
        self.assertIn("hoi_expected_sequences=438", text)
        self.assertIn("hoi_sequence_limit=null", text)
        self.assertIn("save_chois_eval_npz=false", text)
        self.assertIn("load_scene=false", text)
        self.assertIn("sample_type=diffusion", text)
        self.assertNotIn("ema_0.9999", text)

    def test_registry_preregistration_is_complete(self):
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        record = next(
            value for value in records
            if value["experiment_id"] == "p1-hoi-d2n-author-native-preregister-s42-20260716"
        )
        self.assertEqual(record["config"]["run_id"], RUN_ID)
        self.assertEqual(record["config"]["evaluation"]["official_test_sequences"], SAMPLE_COUNT)
        self.assertEqual(
            record["config"]["gate"]["balanced_improvement_ci_lower_gt_zero_for"],
            ["mpjpe", "end_obj_trans_err", "xy_points_err", "obj_trans_dist"],
        )
        self.assertFalse(record["config"]["evaluation"]["chois_used"])
        self.assertFalse(record["config"]["training_started"])


class D2NStatisticsTests(unittest.TestCase):
    def test_paired_statistics_are_deterministic_and_directional(self):
        first = np.asarray([3.0, 4.0, 5.0])
        second = np.asarray([1.0, 2.0, 3.0])
        first_result = paired_difference(first, second, replicates=100)
        second_result = paired_difference(first, second, replicates=100)
        self.assertEqual(first_result, second_result)
        self.assertAlmostEqual(
            first_result["paired_mean_first_minus_second"], 2.0
        )
        ratio = paired_ratio(second, first, replicates=100)
        self.assertAlmostEqual(ratio["mean_ratio"], 0.5)

    @staticmethod
    def _records(offset):
        return {
            f"sequence-{index:03d}": {
                "mpjpe": 10.0 + offset,
                "end_obj_trans_err": 5.0 + offset,
                "pelvis_goal_error_cm": 4.0 + offset,
                "obj_trans_dist": 8.0 + offset,
                "foot_sliding": 0.3 + offset * 0.01,
                "contact_f1": 0.7 - offset * 0.01,
            }
            for index in range(16)
        }

    def test_all_three_checkpoints_are_paired_and_all_gate_metrics_reported(self):
        records = {
            "source": self._records(2.0),
            "current": self._records(1.0),
            "balanced": self._records(0.0),
        }
        result = comparisons(records)
        self.assertEqual(
            set(result), {"balanced_vs_source", "balanced_vs_current"}
        )
        for value in result.values():
            self.assertEqual(
                set(value["comparator_minus_balanced_lower_is_better"]),
                set(GATE_LOWER_METRICS),
            )
            self.assertIn("balanced_over_comparator_foot_sliding", value)
            self.assertIn("balanced_minus_comparator_contact_f1", value)
            self.assertEqual(
                value["comparator_minus_balanced_lower_is_better"]["mpjpe"][
                    "bootstrap_replicates"
                ],
                BOOTSTRAP_REPLICATES,
            )

    def test_gate_requires_quality_improvement_and_physical_preservation(self):
        candidate_results = {
            name: {"sample_count": SAMPLE_COUNT, "finite": True}
            for name in CANDIDATES
        }
        comparisons_value = {}
        for comparator in ("source", "current"):
            comparisons_value[f"balanced_vs_{comparator}"] = {
                "comparator_minus_balanced_lower_is_better": {
                    metric: {"bootstrap_95_ci": [0.1, 0.2]}
                    for metric in GATE_LOWER_METRICS
                },
                "balanced_over_comparator_foot_sliding": {
                    "bootstrap_95_ci": [0.9, 1.0]
                },
                "balanced_minus_comparator_contact_f1": {
                    "bootstrap_95_ci": [-0.01, 0.02]
                },
            }
        decision = transfer_gate(
            candidate_results,
            comparisons_value,
            hashes_exact=True,
            sampler_contract_exact=True,
        )
        self.assertTrue(decision["passed"])
        self.assertFalse(decision["training_authorized"])
        comparisons_value["balanced_vs_source"][
            "balanced_over_comparator_foot_sliding"
        ]["bootstrap_95_ci"] = [1.1, 1.2]
        decision = transfer_gate(
            candidate_results,
            comparisons_value,
            hashes_exact=True,
            sampler_contract_exact=True,
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "author-native-latest-transfer-negative-stop",
        )

    def test_sequence_identity_mismatch_is_rejected(self):
        records = {
            "source": self._records(2.0),
            "current": self._records(1.0),
            "balanced": self._records(0.0),
        }
        records["source"].pop("sequence-000")
        with self.assertRaisesRegex(ValueError, "sequence identities"):
            comparisons(records)


class D2NProductionContractTests(unittest.TestCase):
    def test_author_metrics_are_unchanged_and_sampler_recomputes_rollout_bps(self):
        current_eval = (ROOT / "code/eval_metrics.py").read_bytes()
        author_eval = subprocess.check_output([
            "git",
            "show",
            "b9a158f75ab0740c91c9cfc8863a65fa381b014c:code/eval_metrics.py",
        ], cwd=ROOT)
        self.assertEqual(current_eval, author_eval)
        script = (ROOT / "code/test_infbagel_hoi.py").read_text(encoding="utf-8")
        self.assertIn("recompute_rollout_bps(", script)
        self.assertIn("current_frame.object_reference", script)
        sampler = (ROOT / "code/priors/diffusion.py").read_text(encoding="utf-8")
        self.assertNotIn("future_gt", sampler)
        self.assertNotIn("stored_per_frame_bps", sampler)

    def test_runner_is_directly_executable(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_hoi_d2n.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--balanced-checkpoint", completed.stdout)


if __name__ == "__main__":
    unittest.main()
