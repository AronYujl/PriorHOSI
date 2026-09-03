"""Closed-form checks for the registered future-occ motion diagnostics."""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.diagnostics import (
    future_occ_motion_diagnostics,
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


if __name__ == "__main__":
    unittest.main()
