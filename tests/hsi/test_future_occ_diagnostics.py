"""Closed-form checks for the registered future-occ motion diagnostics."""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.diagnostics import future_occ_motion_diagnostics
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


if __name__ == "__main__":
    unittest.main()
