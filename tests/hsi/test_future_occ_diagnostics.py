"""Closed-form checks for the registered future-occ motion diagnostics."""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.diagnostics import (
    future_occ_motion_diagnostics,
    predictor_decomp_metrics,
    single_window_chain_metrics,
    summarize_predictor_decomp,
    summarize_single_window_chain,
    summarize_teacher_forced_boundary,
    teacher_forced_boundary_metrics,
)
from priors.hsi.metrics import StitchedSequence


class FutureOccMotionDiagnosticsTests(unittest.TestCase):
    def test_first_two_acceleration_centres_and_pelvis_displacement(self):
        time = torch.arange(8, dtype=torch.float64)
        x = time**3
        frames = torch.zeros(8, 2, 3, dtype=torch.float64)
        frames[:, :, 0] = x[:, None]
        joints = StitchedSequence(
            frames=frames,
            seams=(5,),
            window_lengths=(5, 3),
            history_frames=2,
        )

        result = future_occ_motion_diagnostics(joints, fps=2.0)

        self.assertAlmostEqual(result["first_window_first2_fk_acc_mps2"], 60.0)
        self.assertAlmostEqual(result["seam_first2_fk_acc_mps2"], 132.0)
        self.assertAlmostEqual(result["all_window_first2_fk_acc_mps2"], 96.0)
        self.assertAlmostEqual(result["pelvis_path_length_m"], 343.0)
        self.assertAlmostEqual(result["pelvis_net_displacement_m"], 343.0)


class TeacherForcedBoundaryDiagnosticsTests(unittest.TestCase):
    def test_internal_history_separates_model_continuity_from_clamped_history(self):
        target_joints = torch.zeros(1, 4, 2, 3, dtype=torch.float64)
        target_joints[0, :, :, 0] = torch.tensor([0.0, 1.0, 4.0, 9.0])[:, None]
        predicted_joints = target_joints + torch.tensor([10.0, 0.0, 0.0])
        target_repr = torch.zeros(1, 4, 216, dtype=torch.float64)
        predicted_repr = target_repr.clone()
        predicted_repr[:, :2, :84] = 3.0
        predicted_repr[:, :2, 84:216] = 2.0

        result = teacher_forced_boundary_metrics(
            predicted_joints,
            target_joints,
            predicted_repr,
            target_repr,
            fps=1.0,
        )

        self.assertAlmostEqual(float(result["gt_first2_fk_acc_mps2"]), 2.0)
        self.assertAlmostEqual(float(result["internal_first2_fk_acc_mps2"]), 2.0)
        self.assertAlmostEqual(float(result["clamped_first2_fk_acc_mps2"]), 10.0)
        self.assertAlmostEqual(float(result["history_pelvis_error_m"]), 10.0)
        self.assertAlmostEqual(float(result["history_fk_joint_error_m"]), 10.0)
        self.assertAlmostEqual(float(result["history_rotation_channel_mae"]), 2.0)

    def test_registered_decisions_use_episode_weighted_holdout(self):
        names = {
            "gt_first2_fk_acc_mps2": 10.0,
            "clamped_first2_fk_acc_mps2": 30.0,
            "internal_first2_fk_acc_mps2": 12.0,
            "clamped_a1_fk_acc_mps2": 20.0,
            "clamped_a2_fk_acc_mps2": 25.0,
            "pelvis_frame2_error_m": 0.2,
            "pelvis_frame3_error_m": 0.2,
            "history_pelvis_error_m": 0.1,
            "history_fk_joint_error_m": 0.1,
            "history_position_error_m": 0.1,
            "history_rotation_channel_mae": 0.1,
        }

        def record(episode_id, stratum, offset=0.0):
            metrics = {}
            for timestep in (498, 250, 50):
                values = dict(names)
                values["clamped_a1_fk_acc_mps2"] += offset
                values["clamped_a2_fk_acc_mps2"] -= offset
                metrics[str(timestep)] = values
            return {
                "episode_id": episode_id,
                "stratum": stratum,
                "metrics": metrics,
            }

        holdout = [
            record("a0", "a", 0.0),
            record("a1", "a", 1.0),
            record("b0", "b", 2.0),
            record("b1", "b", 3.0),
        ]
        train = [record("t0", "train"), record("t1", "train")]
        for item in train:
            for values in item["metrics"].values():
                values["pelvis_frame2_error_m"] = 0.1

        result = summarize_teacher_forced_boundary(
            holdout,
            train,
            {"a": 0.5, "b": 0.5},
            seed=42,
            replicates=100,
        )

        self.assertEqual(result["decisions"]["j1_single_forward_seam"], "SUPPORTED")
        self.assertEqual(result["decisions"]["j2_history_clamp_mechanism"], "SUPPORTED")
        self.assertEqual(
            result["decisions"]["j3_generalization"],
            "HOLDOUT_AT_LEAST_30_PERCENT_HIGHER",
        )
        self.assertIs(
            result["decisions"]["r3_history_supervision_may_be_proposed"], True
        )


class PredictorDecompDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _record(episode_id, stratum):
        metrics = {
            "gt_first2_fk_acc_mps2": 1.0,
            "conditional_first2_fk_acc_mps2": 2.0,
            "unconditional_first2_fk_acc_mps2": 1.8,
            "cfg_w0_first2_fk_acc_mps2": 2.0,
            "cfg_w0.5_first2_fk_acc_mps2": 2.1,
            "cfg_w1_first2_fk_acc_mps2": 2.2,
            "zero_velocity_history_first2_fk_acc_mps2": 2.0,
            "history_velocity_gain": 0.4,
        }
        for frame in range(2, 6):
            metrics["conditional_frame%d_fk_error_m" % frame] = 1.0
            metrics["constant_position_frame%d_fk_error_m" % frame] = 1.5
            metrics["constant_velocity_frame%d_fk_error_m" % frame] = 0.8
        return {"episode_id": episode_id, "stratum": stratum, "metrics": metrics}

    def test_registered_k_decisions_and_d3_gate(self):
        holdout = [
            self._record("a0", "a"),
            self._record("a1", "a"),
            self._record("b0", "b"),
            self._record("b1", "b"),
        ]
        train = [self._record("t0", "train"), self._record("t1", "train")]
        result = summarize_predictor_decomp(
            holdout, train, {"a": 0.5, "b": 0.5}, seed=42, replicates=50
        )

        self.assertEqual(result["decisions"]["k1_cfg"], "SUPPORTED")
        self.assertEqual(result["decisions"]["k2_constant_velocity"], "SUPPORTED")
        self.assertEqual(result["decisions"]["k3_history_velocity_use"], "SUPPORTED")
        self.assertIs(result["decisions"]["k4_d1_erratum_required"], True)
        self.assertIs(result["decisions"]["d3_w0_authorized"], True)
        self.assertIs(result["decisions"]["r3_res_may_be_proposed"], True)

    def test_window_metrics_use_clamped_history_and_velocity_response(self):
        target_joints = torch.zeros(1, 6, 2, 3)
        target_joints[0, :, :, 0] = torch.arange(6)[:, None]
        target_positions = torch.zeros(1, 6, 84)
        target_positions[:, 1] = 1.0
        predicted_joints = {
            name: target_joints.clone()
            for name in (
                "conditional", "unconditional", "cfg_w0", "cfg_w0.5", "cfg_w1",
                "zero_velocity_history",
            )
        }
        predicted_positions = {
            name: target_positions.clone() for name in predicted_joints
        }
        predicted_positions["zero_velocity_history"][:, 2] = -1.0

        result = predictor_decomp_metrics(
            predicted_joints,
            target_joints,
            predicted_positions,
            target_positions,
            fps=1.0,
        )

        self.assertAlmostEqual(float(result["history_velocity_gain"]), 0.5)
        self.assertAlmostEqual(float(result["conditional_frame2_fk_error_m"]), 0.0)


class SingleWindowChainDiagnosticsTests(unittest.TestCase):
    def test_rho_preservation_threshold(self):
        records = []
        for episode_id, stratum in (("a0", "a"), ("a1", "a"), ("b0", "b"), ("b1", "b")):
            records.append(
                {
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "metrics": {
                        "gt_first2_fk_acc_mps2": 1.0,
                        "trace_t498_first2_fk_acc_mps2": 3.0,
                        "final_first2_fk_acc_mps2": 2.8,
                    },
                }
            )
        result = summarize_single_window_chain(
            records, {"a": 0.5, "b": 0.5}, seed=42, replicates=50
        )

        self.assertAlmostEqual(result["derived"]["rho"], 0.9)
        self.assertEqual(
            result["decision"]["chain_retains_predictor_seam"], "SUPPORTED"
        )

    def test_window_terms_share_the_same_gt_history(self):
        target = torch.zeros(1, 4, 2, 3)
        target[0, :, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])[:, None]
        trace = target.clone()
        final = target.clone()
        trace[:, 2:, :, 0] += 1.0
        final[:, 2:, :, 0] += 0.5

        result = single_window_chain_metrics(trace, final, target, fps=1.0)

        self.assertGreater(float(result["trace_t498_first2_fk_acc_mps2"]), 0.0)
        self.assertGreater(float(result["final_first2_fk_acc_mps2"]), 0.0)


if __name__ == "__main__":
    unittest.main()
