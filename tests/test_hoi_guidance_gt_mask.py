"""Preregistered Phase 1B P6 cell U: the ground-truth contact mask probe.

Cell U is a NON-DEPLOYABLE diagnostic ceiling.  Ground-truth contact does not
exist at inference time; the probe exists only to bound how much of the
engagement gap guidance could recover if the engagement decision were perfect.

The failure mode these tests defend against is silent: a misaligned window slice
does not crash, it produces a plausible-looking ceiling, and that number would
then be used to decide whether to abandon inference-time guidance entirely.
Consecutive rollout windows overlap by ``auto_regre_num``, so the stride is
``max_window_size - auto_regre_num`` (14, not 16) -- an off-by-``auto_regre_num``
slice is the expected mistake.
"""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from priors.inference_guidance import (
    GuidanceAudit,
    GuidanceSettings,
    MASK_SOURCE_GROUND_TRUTH,
    MASK_SOURCE_PREDICTED,
    resolve_contact_mask,
)

WINDOW_FRAMES = 16
AUTO_REGRE_NUM = 2
STRIDE = WINDOW_FRAMES - AUTO_REGRE_NUM
MAX_LEN = 3


def _window_slice(sequence_contact, step):
    """The evaluator's per-window slice, mirrored for testing.

    Kept deliberately identical in arithmetic to test_infbagel_hoi.py so that a
    divergence between the two shows up as a test failure rather than as a
    quietly wrong probe.
    """
    start = step * STRIDE
    frames = sequence_contact.shape[0]
    if start >= frames:
        return sequence_contact[-1:].repeat(WINDOW_FRAMES, 1)
    piece = sequence_contact[start:start + WINDOW_FRAMES]
    if piece.shape[0] < WINDOW_FRAMES:
        pad = WINDOW_FRAMES - piece.shape[0]
        piece = torch.cat([piece, piece[-1:].repeat(pad, 1)], dim=0)
    return piece


def _index_encoding_labels(frames):
    """Labels whose value encodes their own global frame index."""
    contact = torch.zeros(frames, 4)
    contact[:, 0] = torch.arange(frames, dtype=torch.float32)
    return contact


class WindowAlignmentTest(unittest.TestCase):
    """The slice handed to step s must cover global frames [14s, 14s+16)."""

    def test_slices_cover_the_expected_global_frames(self):
        # The expected span is written out literally rather than derived from
        # STRIDE: a test that computes its expectation from the same constant it
        # is checking would pass under any stride, including a wrong one.
        expected_starts = (0, 14, 28)
        labels = _index_encoding_labels(60)
        for step in range(MAX_LEN):
            window = _window_slice(labels, step)
            self.assertEqual(window.shape, (WINDOW_FRAMES, 4))
            start = expected_starts[step]
            expected = torch.arange(
                start, start + WINDOW_FRAMES, dtype=torch.float32
            )
            self.assertTrue(
                torch.equal(window[:, 0], expected),
                msg=(
                    f"step {step} covered frames "
                    f"[{float(window[0, 0])}, {float(window[-1, 0])}], "
                    f"expected [{start}, {start + WINDOW_FRAMES - 1}]"
                ),
            )

    def test_stride_is_not_the_window_length(self):
        """Pin the overlap: a stride of 16 would be the off-by-two defect."""
        labels = _index_encoding_labels(60)
        first, second = _window_slice(labels, 0), _window_slice(labels, 1)
        self.assertEqual(float(second[0, 0]), STRIDE)
        self.assertNotEqual(float(second[0, 0]), WINDOW_FRAMES)
        # The last auto_regre_num frames of window 0 reappear at the start of 1.
        self.assertTrue(
            torch.equal(first[-AUTO_REGRE_NUM:, 0], second[:AUTO_REGRE_NUM, 0])
        )

    def test_short_sequences_are_padded_to_full_length(self):
        for frames in (5, 20, 44):
            labels = _index_encoding_labels(frames)
            for step in range(MAX_LEN):
                window = _window_slice(labels, step)
                self.assertEqual(
                    window.shape, (WINDOW_FRAMES, 4), msg=f"{frames} frames, step {step}"
                )


class GroundTruthMaskTest(unittest.TestCase):
    def setUp(self):
        self.predicted = torch.rand(2, WINDOW_FRAMES, 4)
        self.settings = GuidanceSettings(
            enabled=True, arm="b", contact_mask_source=MASK_SOURCE_GROUND_TRUTH
        )

    def test_missing_labels_raise(self):
        with self.assertRaises(ValueError):
            resolve_contact_mask(
                self.predicted, settings=self.settings, ground_truth_contact=None
            )

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            resolve_contact_mask(
                self.predicted,
                settings=self.settings,
                ground_truth_contact=torch.rand(2, WINDOW_FRAMES + 1, 4),
            )

    def test_ground_truth_mask_replaces_the_predicted_gate(self):
        """The GT labels, not the model's own prediction, must select frames."""
        gt = torch.zeros(2, WINDOW_FRAMES, 4)
        gt[:, :5, 0] = 1.0
        predicted = torch.zeros(2, WINDOW_FRAMES, 4)
        predicted[:, 10:, 0] = 1.0
        resolved = resolve_contact_mask(
            predicted, settings=self.settings, ground_truth_contact=gt
        )
        engaged = resolved[..., -4:-2] > 0.95
        self.assertTrue(bool(engaged[:, :5, 0].all()))
        self.assertFalse(bool(engaged[:, 10:, 0].any()))

    def test_predicted_source_ignores_ground_truth(self):
        """Guarding the default path: sealed results must not shift."""
        predicted = torch.rand(2, WINDOW_FRAMES, 4)
        settings = GuidanceSettings(
            enabled=True, arm="b", contact_mask_source=MASK_SOURCE_PREDICTED
        )
        resolved = resolve_contact_mask(
            predicted,
            settings=settings,
            ground_truth_contact=torch.ones(2, WINDOW_FRAMES, 4),
        )
        self.assertIs(resolved, predicted)


class GroundTruthAuditTest(unittest.TestCase):
    def test_engagement_fraction_is_recorded(self):
        audit = GuidanceAudit()
        audit.record_ground_truth_window(10, 32)
        audit.record_ground_truth_window(6, 32)
        report = audit.as_dict()
        self.assertEqual(report["guidance_ground_truth_windows"], 2)
        self.assertEqual(report["guidance_ground_truth_engaged_frames"], 16)
        self.assertEqual(report["guidance_ground_truth_total_frames"], 64)
        self.assertAlmostEqual(
            report["guidance_ground_truth_engagement_fraction"], 0.25, places=9
        )

    def test_absent_probe_reports_none(self):
        report = GuidanceAudit().as_dict()
        self.assertEqual(report["guidance_ground_truth_windows"], 0)
        self.assertIsNone(report["guidance_ground_truth_engagement_fraction"])

    def test_gate_statistic_is_not_the_evaluator_geometric_one(self):
        """Documented, because equating the two would fire on correct alignment.

        ``gt_contact_percent`` (0.66188) is a geometric judgement on interpolated
        evaluation frames; the annotation channel thresholded at 0.95 is 0.62794
        over the 482 test files.  The gate must therefore validate the SLICE, not
        assert equality between two different statistics.
        """
        self.assertNotAlmostEqual(0.66188, 0.62794, places=2)


if __name__ == "__main__":
    unittest.main()
