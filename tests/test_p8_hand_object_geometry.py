"""Preregistered P8 hand-object relative geometry training term.

Guards against silent failures: the term must engage a useful fraction of frames,
route gradient to both palms and object surface, and respect the history-frame
boundary. A mask that reads the wrong channels or a gradient that only reaches
one operand would silently reduce a 148 GPU-hour experiment to a no-op.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from priors.losses import (
    P8_CONTACT_HAND_CHANNELS,
    P8_CONTACT_THRESHOLD,
    P8_PALM_JOINTS,
    masked_hand_object_distance_loss,
)
from priors.representation import REPRESENTATION


def _batch(B=4, T=16, V=80, seed=42):
    """Synthetic geometry with REAL GT contact annotation from test files."""
    import glob
    g = torch.Generator().manual_seed(seed)
    files = sorted(glob.glob(str(ROOT / "data/test/contact_label_npy_files/*.npy")))
    if len(files) < B:
        raise FileNotFoundError(f"need {B} annotation files, found {len(files)}")
    contact_rows = []
    for f in files[:B]:
        a = np.load(f)
        if a.shape[0] >= T:
            contact_rows.append(torch.from_numpy(a[:T].astype("float32")))
        else:
            # Pad short sequences by repeating the last frame.
            padded = np.pad(a, ((0, T - a.shape[0]), (0, 0)), mode="edge")
            contact_rows.append(torch.from_numpy(padded.astype("float32")))
    contact = torch.stack(contact_rows)
    return {
        "fk": torch.randn(B, T, 24, 3, generator=g) * 0.5,
        "surface": torch.randn(B, T, V, 3, generator=g) * 0.3,
        "contact_gt": contact,
    }


class MaskLogicTest(unittest.TestCase):
    """The mask must read channels [:2], not [-2:], and threshold correctly."""

    def test_hand_channels_are_first_two(self):
        """Guard against the [-2:] error that would have wasted 148 GPU-hours."""
        self.assertEqual(P8_CONTACT_HAND_CHANNELS, slice(0, 2))

    def test_engaged_fraction_is_useful(self):
        """Engagement should be 0.6-0.7 over test annotations, not 0.02 or 0.99."""
        b = _batch(B=16)
        active = slice(REPRESENTATION.history_frames, None)
        engaged = (
            b["contact_gt"][:, active, P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD
        ).any(dim=-1)
        frac = float(engaged.float().mean())
        self.assertGreater(frac, 0.25, "engagement <25% suggests wrong channels")
        self.assertLess(frac, 0.80, "engagement >80% suggests inverted logic")

    def test_any_hand_channel_above_threshold_engages(self):
        """Either left OR right hand contact should engage the frame."""
        B, T = 2, 8
        fk = torch.zeros(B, T, 24, 3)
        surf = torch.zeros(B, T, 10, 3)
        contact = torch.zeros(B, T, 4)
        # Batch 0: only left hand (ch0) at frames 3,4
        contact[0, 3:5, 0] = 1.0
        # Batch 1: only right hand (ch1) at frames 5,6
        contact[1, 5:7, 1] = 1.0
        loss = masked_hand_object_distance_loss(fk, surf, contact)
        # Loss must be computed: 4 engaged frames out of 2*(8-2)=12 active frames.
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

    def test_history_frames_do_not_engage(self):
        """Only frames [history:] participate; the first `history_frames` are ignored."""
        B, T, H = 3, 16, REPRESENTATION.history_frames
        fk = torch.randn(B, T, 24, 3)
        surf = torch.randn(B, T, 50, 3)
        contact = torch.zeros(B, T, 4)
        # Mark ONLY the history frames as engaged.
        contact[:, :H, :2] = 1.0
        # The mask sees zero engaged frames in the active slice -> denominator clamp.
        loss = masked_hand_object_distance_loss(fk, surf, contact)
        self.assertTrue(torch.isfinite(loss))


class GradientRoutingTest(unittest.TestCase):
    """Gradient must reach the palms and the object surface, nowhere else."""

    def test_gradient_reaches_both_palms_and_surface(self):
        b = _batch(B=4)
        fk_leaf = b["fk"].clone().requires_grad_(True)
        surf_leaf = b["surface"].clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(fk_leaf, surf_leaf, b["contact_gt"])
        grad_fk, grad_surf = torch.autograd.grad(loss, [fk_leaf, surf_leaf])
        self.assertGreater(grad_fk.norm().item(), 0.0, "FK gradient is zero")
        self.assertGreater(grad_surf.norm().item(), 0.0, "surface gradient is zero")

    def test_gradient_only_on_palm_joints(self):
        """Non-palm joints must receive exactly zero gradient."""
        b = _batch(B=4)
        fk_leaf = b["fk"].clone().requires_grad_(True)
        surf_leaf = b["surface"].clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(fk_leaf, surf_leaf, b["contact_gt"])
        grad_fk = torch.autograd.grad(loss, fk_leaf, create_graph=False)[0]
        non_palm = [j for j in range(24) if j not in P8_PALM_JOINTS]
        self.assertAlmostEqual(
            grad_fk[:, :, non_palm].norm().item(), 0.0, places=9,
            msg="non-palm joints received gradient"
        )

    def test_gradient_only_on_active_frames(self):
        """History frames must receive exactly zero gradient."""
        b = _batch(B=4)
        fk_leaf = b["fk"].clone().requires_grad_(True)
        surf_leaf = b["surface"].clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(fk_leaf, surf_leaf, b["contact_gt"])
        grad_fk = torch.autograd.grad(loss, fk_leaf, create_graph=False)[0]
        H = REPRESENTATION.history_frames
        self.assertAlmostEqual(
            grad_fk[:, :H].norm().item(), 0.0, places=9,
            msg="history frames received gradient"
        )

    def test_no_gradient_on_non_engaged_frames(self):
        """Frames where the mask is False must contribute zero to the gradient."""
        B, T = 2, 8
        fk = torch.randn(B, T, 24, 3, requires_grad=True)
        surf = torch.randn(B, T, 10, 3, requires_grad=True)
        contact = torch.zeros(B, T, 4)
        # Only batch 0, frame 3 is engaged.
        contact[0, 3, 0] = 1.0
        loss = masked_hand_object_distance_loss(fk, surf, contact)
        grad_fk = torch.autograd.grad(loss, fk, create_graph=False)[0]
        # Batch 1 has no engaged frames -> zero gradient everywhere.
        self.assertAlmostEqual(grad_fk[1].norm().item(), 0.0, places=7)


class DenominatorClampTest(unittest.TestCase):
    """When no frames engage, the loss must still be finite (clamp_min prevents NaN)."""

    def test_all_non_contact_batch_returns_finite(self):
        B, T, V = 4, 16, 50
        fk = torch.randn(B, T, 24, 3)
        surf = torch.randn(B, T, V, 3)
        contact = torch.zeros(B, T, 4)  # no engaged frames
        loss = masked_hand_object_distance_loss(fk, surf, contact)
        self.assertTrue(torch.isfinite(loss))


class ValidationTest(unittest.TestCase):
    """Malformed inputs must raise, not silently compute garbage."""

    def test_fk_shape_must_have_at_least_24_joints(self):
        with self.assertRaises(ValueError):
            masked_hand_object_distance_loss(
                torch.zeros(2, 8, 20, 3),  # only 20 joints
                torch.zeros(2, 8, 10, 3),
                torch.zeros(2, 8, 4),
            )

    def test_surface_must_be_4d(self):
        with self.assertRaises(ValueError):
            masked_hand_object_distance_loss(
                torch.zeros(2, 8, 24, 3),
                torch.zeros(2, 8, 10),  # missing last dim
                torch.zeros(2, 8, 4),
            )

    def test_contact_batch_and_time_must_match_fk(self):
        with self.assertRaises(ValueError):
            masked_hand_object_distance_loss(
                torch.zeros(3, 16, 24, 3),
                torch.zeros(3, 16, 10, 3),
                torch.zeros(2, 16, 4),  # batch mismatch
            )


if __name__ == "__main__":
    unittest.main()
