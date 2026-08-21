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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from priors.hoi.inference_guidance import (
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


# --- 2026-08-21 corrected-mask amendment -------------------------------------
# The sealed 2026-08-05 cell U fed _gt_contact_window a 16-frame WINDOW instead
# of the sequence-length track its stride-14 arithmetic assumes.  The slicer was
# never the defect; the source was.  These tests pin the defect signature, the
# corrected source, the gate that keeps the fix off the default path, and the
# two preregistered engagement constants.
CORRECTED_ENGAGED_FRAMES = 13902
CORRECTED_TOTAL_FRAMES = 21024
CORRECTED_ENGAGEMENT_FRACTION = 0.6612442922374430
SEALED_DEGENERATE_ENGAGED_FRAMES = 16591
SEALED_DEGENERATE_ENGAGEMENT_FRACTION = 0.7891457382039574


class CorrectedGroundTruthTrackTests(unittest.TestCase):
    """The cell-U mask must come from a sequence-length track."""

    def test_window_length_track_is_the_defect_signature(self):
        # What the sealed run actually fed the sampler.  Kept as a test so the
        # defect can never be reintroduced silently by "simplifying" the source.
        track = _index_encoding_labels(WINDOW_FRAMES)
        step0 = _window_slice(track, 0)
        self.assertEqual(len(set(step0[:, 0].tolist())), WINDOW_FRAMES)
        step1 = _window_slice(track, 1)
        self.assertEqual(len(set(step1[:, 0].tolist())), 2)
        self.assertEqual([int(step1[0, 0]), int(step1[1, 0])], [14, 15])
        step2 = _window_slice(track, 2)
        self.assertEqual(len(set(step2[:, 0].tolist())), 1)
        self.assertEqual(int(step2[0, 0]), WINDOW_FRAMES - 1)

    def test_sequence_length_track_recovers_global_frames(self):
        frames = (MAX_LEN - 1) * STRIDE + WINDOW_FRAMES
        track = _index_encoding_labels(frames)
        for step in range(MAX_LEN):
            window = _window_slice(track, step)
            self.assertEqual(int(window[0, 0]), step * STRIDE)
            self.assertEqual(len(set(window[:, 0].tolist())), WINDOW_FRAMES)

    def test_accessor_equals_the_per_window_reads_it_replaces(self):
        # The unit form of the real-data equivalence measured over the official
        # protocol: the accessor sliced at stride 14 equals each window's own
        # contact_label on all 438 x 3 windows.
        import numpy as np
        from datasets.infbagel import InfBaGelDataset

        raw_frames, step = 3 * WINDOW_FRAMES * 3, 3
        annotation = np.zeros((raw_frames, 4), dtype=np.float32)
        annotation[:, 1] = np.arange(raw_frames, dtype=np.float32)
        base_raw = 1000

        class _Fake:
            load_language = True
            load_object_goal = True
            max_window_size = WINDOW_FRAMES
            need_object = {i: True for i in range(MAX_LEN)}
            ori_sequence_idx = {i: 0 for i in range(MAX_LEN)}
            scene_name = {0: 'seq'}
            ori_sequence_start_idx = {0: base_raw}
            contact_label = {'seq': annotation}
            start_ind = {i: base_raw + i * STRIDE * step for i in range(MAX_LEN)}

        fake = _Fake()
        fake.step = step
        accessor = InfBaGelDataset.sequence_contact_label(fake, 0)
        self.assertGreater(accessor.shape[0], (MAX_LEN - 1) * STRIDE + WINDOW_FRAMES - 1)
        for window in range(MAX_LEN):
            start = fake.start_ind[window] - base_raw
            per_window = annotation[start:start + WINDOW_FRAMES * step:step]
            sliced = _window_slice(torch.from_numpy(accessor), window)
            self.assertTrue(
                torch.equal(sliced, torch.from_numpy(per_window)),
                "window %d: accessor slice must equal the per-window read" % window,
            )

    def test_accessor_is_not_a_getitem_key(self):
        # A new __getitem__ key would hand the default collate a variable-length
        # entry on every path that batches this dataset.  A method cannot.
        source = (ROOT / "code" / "datasets" / "infbagel.py").read_text()
        getitem = source.split("    def __getitem__(self, idx):", 1)[1]
        getitem = getitem.split("\n    def ", 1)[0]
        self.assertNotIn("sequence_contact_label", getitem)
        self.assertIn("    def sequence_contact_label(self, idx):", source)

    def test_cell_u_feed_is_gated_and_uses_the_sequence_accessor(self):
        source = (ROOT / "code" / "test_infbagel_hoi.py").read_text()
        self.assertIn(
            "if _hoi_guidance_uses_ground_truth(cfg):\n"
            "            gt_contact_label_batch.append(torch.from_numpy(\n"
            "                synhsi_dataset.sequence_contact_label(seg_id_dict[seg_id])",
            source,
            "the cell-U feed must be gated and must read the sequence-length track",
        )
        self.assertNotIn("gt_contact_label_batch.append(contact_label)", source)

    def test_teacher_forcing_accumulator_stays_independent(self):
        # Two fixes, one file, no overlap: this one repairs the SOURCE of the
        # guidance mask; the teacher-forcing one BYPASSES the source with its own
        # per-window accumulator.  Neither may absorb the other.
        source = (ROOT / "code" / "test_infbagel_hoi.py").read_text()
        self.assertIn("contact_all_gt = torch.zeros(0, cfg.max_window_size, 4)", source)
        feed = source.split("if _hoi_guidance_uses_ground_truth(cfg):", 1)[1][:400]
        self.assertNotIn("contact_all_gt", feed)

    def test_corrected_engagement_constants_are_preregistered(self):
        # Fixed BEFORE the corrected run, computed CPU-only from the annotation
        # files with no model and no GPU
        # (.claude/scratch/cellu_fix/verify_mask_arithmetic.py).  The corrected
        # run must reproduce these exactly or its mask is still wrong.
        self.assertEqual(CORRECTED_TOTAL_FRAMES, 438 * MAX_LEN * WINDOW_FRAMES)
        self.assertAlmostEqual(
            CORRECTED_ENGAGED_FRAMES / CORRECTED_TOTAL_FRAMES,
            CORRECTED_ENGAGEMENT_FRACTION, places=15,
        )
        self.assertAlmostEqual(
            SEALED_DEGENERATE_ENGAGED_FRAMES / CORRECTED_TOTAL_FRAMES,
            SEALED_DEGENERATE_ENGAGEMENT_FRACTION, places=15,
        )
        self.assertGreater(
            SEALED_DEGENERATE_ENGAGED_FRAMES, CORRECTED_ENGAGED_FRAMES,
            "the degenerate mask over-engaged; the sealed ceiling is biased, "
            "not merely noisy",
        )

    def test_population_matched_statistic_is_close_to_the_geometric_one(self):
        """Why the preregistered engagement gate was dropped for a bad reason.

        ``cells/U/probe_validity/preregistered_gate_was_wrong`` justified dropping
        the "engagement must equal gt_contact_percent 0.66188" gate by noting the
        annotation channel measures 0.62794 -- but that 0.62794 is every frame of
        all 482 annotation files, a different population from the probe's 438
        sequences x first 3 windows.  On the probe's own population the corrected
        annotation statistic is 0.66124, which the dropped gate would have passed
        and the degenerate 0.78915 would have failed by 0.128.  Empirical, not
        analytic: the restored gate is therefore stated with a tolerance, and the
        binding gate remains the exact 13902/21024.
        """
        self.assertAlmostEqual(CORRECTED_ENGAGEMENT_FRACTION, 0.66188, places=2)
        self.assertGreater(abs(SEALED_DEGENERATE_ENGAGEMENT_FRACTION - 0.66188), 0.1)


class AuditRecordsWhichMaskTests(unittest.TestCase):
    """Gate G6 of the corrected cell-U preregistration.

    A sealed run's ``normalization_audit`` reported the guidance decomposition but
    never which contact mask produced it, so nothing in the artifact separated a
    NON-DEPLOYABLE ground-truth-mask probe from a deployable predicted-mask cell.
    That is how the degenerate cell-U mask survived to be cited: the number was
    readable, its provenance was not.  Both keys are reported even when they are
    at their defaults, because "the field is absent" and "the field is predicted"
    must not look the same to a reader.
    """

    def test_a_ground_truth_probe_says_so_in_its_audit(self):
        audit = GuidanceAudit(
            GuidanceSettings(
                enabled=True,
                contact_mask_source=MASK_SOURCE_GROUND_TRUTH,
                contact_mask_threshold=0.5,
            )
        )
        report = audit.as_dict()
        self.assertEqual(
            report["guidance_contact_mask_source"], MASK_SOURCE_GROUND_TRUTH
        )
        self.assertEqual(report["guidance_contact_mask_threshold"], 0.5)

    def test_a_deployable_cell_says_predicted_rather_than_nothing(self):
        report = GuidanceAudit(GuidanceSettings(enabled=True)).as_dict()
        self.assertEqual(
            report["guidance_contact_mask_source"], MASK_SOURCE_PREDICTED
        )
        self.assertIsInstance(report["guidance_contact_mask_threshold"], float)

    def test_an_unbound_audit_reports_null_not_a_default(self):
        """No settings means unknown, which must not read as ``predicted``."""
        report = GuidanceAudit().as_dict()
        self.assertIsNone(report["guidance_contact_mask_source"])
        self.assertIsNone(report["guidance_contact_mask_threshold"])

    def test_the_keys_survive_a_populated_audit(self):
        """The populated branch takes a different code path to the empty one."""
        audit = GuidanceAudit(
            GuidanceSettings(
                enabled=True, contact_mask_source=MASK_SOURCE_GROUND_TRUTH
            )
        )
        gradient = torch.randn(
            2, 16, 232, generator=torch.Generator().manual_seed(19)
        )
        audit.record(
            gradient,
            gradient * 0.1,
            torch.tensor(1.0),
            torch.tensor(0.5),
        )
        report = audit.as_dict()
        self.assertEqual(report["guidance_applied_steps"], 1)
        self.assertEqual(
            report["guidance_contact_mask_source"], MASK_SOURCE_GROUND_TRUTH
        )


if __name__ == "__main__":
    unittest.main()
