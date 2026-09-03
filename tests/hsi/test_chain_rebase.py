"""Mechanism checks for the preregistered D4-B chain-history rebase."""

import ast
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import rebase_model_output
from priors.hsi.diagnostics import (
    chain_rebase_rollout_telemetry,
    summarize_chain_rebase,
)
from priors.hsi.metrics import StitchedSequence


class ChainRebaseArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros(1, 16, 232)
        self.x[:, 0, :84] = 1.0
        self.x[:, 1, :84] = 3.0
        self.x[:, 0, 84:216] = 2.0
        self.x[:, 1, 84:216] = 5.0
        self.output = torch.randn(1, 16, 232, generator=torch.Generator().manual_seed(4))

    def test_off_returns_the_same_object(self):
        self.assertIs(rebase_model_output(self.output, self.x, "off"), self.output)

    def test_c1_uses_fixed_history_and_preserves_future_differences(self):
        result = rebase_model_output(self.output, self.x, "c1")
        self.assertTrue(torch.allclose(result[:, 2, :84], torch.full((1, 84), 5.0)))
        self.assertTrue(
            torch.allclose(
                result[:, 3:, :84] - result[:, 2:-1, :84],
                self.output[:, 3:, :84] - self.output[:, 2:-1, :84],
                atol=1e-6,
            )
        )
        self.assertTrue(torch.equal(result[:, :, 84:], self.output[:, :, 84:]))

    def test_c2_uses_the_oracle_position(self):
        oracle = torch.full((1, 84), -7.0)
        result = rebase_model_output(self.output, self.x, "c2", oracle)
        self.assertTrue(torch.allclose(result[:, 2, :84], oracle, atol=1e-6))

    def test_c3_adds_the_rotation_rebase(self):
        result = rebase_model_output(self.output, self.x, "c3")
        self.assertTrue(torch.allclose(result[:, 2, :84], torch.full((1, 84), 5.0)))
        self.assertTrue(
            torch.allclose(result[:, 2, 84:216], torch.full((1, 132), 8.0), atol=1e-6)
        )
        self.assertTrue(torch.equal(result[:, :, 216:], self.output[:, :, 216:]))

    def test_min_timestep_zero_is_identical_to_the_existing_c3_path(self):
        existing = rebase_model_output(self.output, self.x, "c3")
        default = rebase_model_output(
            self.output, self.x, "c3", timestep=0, min_timestep=0
        )
        self.assertTrue(torch.equal(default, existing))

    def test_rebase_runs_only_at_or_above_the_minimum_timestep(self):
        below = rebase_model_output(
            self.output, self.x, "c3", timestep=183, min_timestep=184
        )
        at_threshold = rebase_model_output(
            self.output, self.x, "c3", timestep=184, min_timestep=184
        )
        self.assertIs(below, self.output)
        self.assertTrue(torch.equal(at_threshold, rebase_model_output(self.output, self.x, "c3")))


class ChainRebaseCallSiteTests(unittest.TestCase):
    def test_diffusion_path_applies_rebase_before_trace_and_posterior(self):
        tree = ast.parse((REPO / "code" / "models" / "infbagel.py").read_text())
        sampler = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Sampler"
        )
        methods = {
            node.name: node for node in sampler.body if isinstance(node, ast.FunctionDef)
        }
        p_sample = methods["p_sample"]
        calls = [
            node.lineno for node in ast.walk(p_sample)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rebase_model_output"
        ]
        posterior = [
            node.lineno for node in ast.walk(p_sample)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "model_mean" for target in node.targets)
        ]
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0], posterior[0])
        self.assertNotIn("rebase_model_output", ast.dump(methods["cm_sample"]))


class ChainRebaseRolloutTelemetryTests(unittest.TestCase):
    def test_a1_a2_use_the_two_accelerations_that_straddle_the_seam(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 12.0, 15.0, 19.0])
        frames = torch.zeros(8, 28, 3)
        frames[:, :, 0] = x[:, None]
        joints = StitchedSequence(
            frames=frames,
            seams=(4,),
            window_lengths=(4, 4),
            history_frames=2,
        )
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        rotations = StitchedSequence(
            frames=identity_6d.reshape(1, 1, 6).repeat(8, 22, 1),
            seams=(4,),
            window_lengths=(4, 4),
            history_frames=2,
        )

        result = chain_rebase_rollout_telemetry(joints, rotations, fps=1.0)

        self.assertAlmostEqual(result["coarse_seam_a1_fk_acc_mps2"], 6.0)
        self.assertAlmostEqual(result["coarse_seam_a2_fk_acc_mps2"], 5.0)
        self.assertAlmostEqual(result["coarse_cross_seam_third_difference_mps3"], 11.0)
        self.assertEqual(result["coarse_seam_count"], 1.0)
        self.assertAlmostEqual(result["rotation6d_abs_cosine_max"], 0.0)
        self.assertLess(result["rotation_matrix_orthogonality_max"], 1e-12)


class PhaseLimitedRebaseDecisionTests(unittest.TestCase):
    def test_third_difference_ratio_controls_the_d6_a_decision(self):
        c0, candidate = [], []
        for index, stratum in enumerate(("s1", "s1", "s2", "s2")):
            base = {
                "gt_first2_fk_acc_mps2": 1.0,
                "final_first2_fk_acc_mps2": 3.0,
                "gt_cross_seam_third_difference_mps3": 10.0,
                "final_cross_seam_third_difference_mps3": 30.0,
                "final_a2_fk_acc_mps2": 2.0,
                "final_frame3_6_fk_error_m": 0.4,
                "final_internal_frame3_8_fk_acc_mps2": 2.0,
            }
            phase_limited = dict(base)
            phase_limited["final_first2_fk_acc_mps2"] = 2.0
            phase_limited["final_cross_seam_third_difference_mps3"] = 18.0
            phase_limited["final_a2_fk_acc_mps2"] = 1.9
            phase_limited["final_frame3_6_fk_error_m"] = 0.3
            phase_limited["final_internal_frame3_8_fk_acc_mps2"] = 1.8
            row = {"episode_id": "e%d" % index, "stratum": stratum, "data_idx": index}
            c0.append({**row, "metrics": base})
            candidate.append({**row, "metrics": phase_limited})

        result = summarize_chain_rebase(
            c0,
            candidate,
            {"s1": 0.5, "s2": 0.5},
            arm="c3",
            min_timestep=184,
            seed=42,
            replicates=50,
        )

        self.assertAlmostEqual(
            result["derived"]["third_difference_excess_ratio"], 0.4
        )
        self.assertEqual(result["decision"], "PROCEED")

    def test_raw_6d_deviation_is_visible_before_projection(self):
        joints = StitchedSequence(
            frames=torch.zeros(6, 28, 3),
            seams=(3,),
            window_lengths=(3, 3),
            history_frames=2,
        )
        raw = torch.tensor([2.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        rotations = StitchedSequence(
            frames=raw.reshape(1, 1, 6).repeat(6, 22, 1),
            seams=(3,),
            window_lengths=(3, 3),
            history_frames=2,
        )

        result = chain_rebase_rollout_telemetry(joints, rotations, fps=10.0)

        self.assertGreater(result["rotation6d_first_axis_norm_mae"], 0.0)
        self.assertGreater(result["rotation6d_abs_cosine_mean"], 0.0)
        self.assertLess(result["rotation_matrix_orthogonality_max"], 1e-12)


if __name__ == "__main__":
    unittest.main()
