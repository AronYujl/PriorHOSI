"""The gating operator's two identity anchors, and the reserved `state` slot.

    x_hat_0,h = G * x_hat_0^HSI + (1 - G) * x_hat_0^HOI

A1  G == 0 returns the HOI tensor BIT FOR BIT, and never reads the HSI side.
A2  G == 1 returns the HSI tensor BIT FOR BIT, and never reads the HOI side.

A1 is what makes the HOI-alone HOSI-test row an anchor of the operator rather
than a separate measurement that happens to sit next to it.  It is asserted here
against a sentinel that raises on any access, so "never reads" is measured and
not inferred from reading the arithmetic -- `0 * nan` is `nan`, so the naive
expression does NOT satisfy A1.
"""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from mixer.composition import (  # noqa: E402
    OBJECT_CHANNEL_START,
    ExpertOutputs,
    compose_x0,
    gate_is_identity,
    human_gate_mask,
)
from priors.core.representation import REPRESENTATION  # noqa: E402


class TripwireTensor:
    """Stands in for an expert output that must not be touched."""

    def __init__(self):
        self.accesses = []

    def __getattr__(self, name):
        self.accesses.append(name)
        raise AssertionError(f'the unused expert output was read: .{name}')

    def __mul__(self, other):
        raise AssertionError('the unused expert output was multiplied')

    __rmul__ = __mul__

    def __add__(self, other):
        raise AssertionError('the unused expert output was added')

    __radd__ = __add__


class AnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hoi = torch.randn(2, 16, 232)
        self.hsi = torch.randn(2, 16, 232)

    def test_gate_zero_returns_hoi_bitwise(self):
        for gate in (0, 0.0, torch.zeros(1), torch.zeros(2, 16, 232)):
            with self.subTest(gate=type(gate).__name__):
                out = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), gate)
                self.assertTrue(torch.equal(out, self.hoi))

    def test_gate_one_returns_hsi_bitwise(self):
        for gate in (1, 1.0, torch.ones(1), torch.ones(2, 16, 232)):
            with self.subTest(gate=type(gate).__name__):
                out = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), gate)
                self.assertTrue(torch.equal(out, self.hsi))

    def test_gate_zero_never_reads_the_hsi_side(self):
        tripwire = TripwireTensor()
        out = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=tripwire), 0)
        self.assertTrue(torch.equal(out, self.hoi))
        self.assertEqual(tripwire.accesses, [])

    def test_gate_one_never_reads_the_hoi_side(self):
        tripwire = TripwireTensor()
        out = compose_x0(ExpertOutputs(hoi=tripwire, hsi=self.hsi), 1)
        self.assertTrue(torch.equal(out, self.hsi))
        self.assertEqual(tripwire.accesses, [])

    def test_anchor_survives_a_nonfinite_unused_expert(self):
        """The property the naive arithmetic fails: 0 * nan is nan."""
        poisoned = torch.full((2, 16, 232), float('nan'))
        out = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=poisoned), 0)
        self.assertTrue(torch.equal(out, self.hoi))
        self.assertTrue(torch.isfinite(out).all())
        naive = 0.0 * poisoned + 1.0 * self.hoi
        self.assertTrue(torch.isnan(naive).all())

    def test_anchor_needs_only_the_expert_it_returns(self):
        self.assertTrue(torch.equal(
            compose_x0(ExpertOutputs(hoi=self.hoi), 0), self.hoi))
        self.assertTrue(torch.equal(
            compose_x0(ExpertOutputs(hsi=self.hsi), 1), self.hsi))
        with self.assertRaises(ValueError):
            compose_x0(ExpertOutputs(hsi=self.hsi), 0)
        with self.assertRaises(ValueError):
            compose_x0(ExpertOutputs(hoi=self.hoi), 1)


class BlendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.hoi = torch.randn(1, 16, 232)
        self.hsi = torch.randn(1, 16, 232)
        self.outputs = ExpertOutputs(hoi=self.hoi, hsi=self.hsi)

    def test_scalar_half_is_the_midpoint_on_the_human_channels(self):
        """The midpoint holds where both experts are supervised, and only there.

        Under the default human mask the object and contact channels stay at HOI,
        so a scalar 0.5 is the midpoint on 0:216 and the identity on 216:232.
        `channel_mask=None` recovers the unmasked midpoint on all 232.
        """
        out = compose_x0(self.outputs, 0.5)
        midpoint = 0.5 * (self.hoi + self.hsi)
        self.assertTrue(torch.allclose(
            out[..., :OBJECT_CHANNEL_START], midpoint[..., :OBJECT_CHANNEL_START]
        ))
        self.assertTrue(torch.equal(
            out[..., OBJECT_CHANNEL_START:], self.hoi[..., OBJECT_CHANNEL_START:]
        ))
        unmasked = compose_x0(self.outputs, 0.5, channel_mask=None)
        self.assertTrue(torch.allclose(unmasked, midpoint))

    def test_per_channel_gate_broadcasts(self):
        gate = torch.zeros(1, 16, 232)
        gate[..., :100] = 1.0
        out = compose_x0(self.outputs, gate)
        self.assertTrue(torch.equal(out[..., :100], self.hsi[..., :100]))
        self.assertTrue(torch.equal(out[..., 100:], self.hoi[..., 100:]))

    def test_a_nearly_zero_gate_is_not_an_anchor(self):
        """One non-zero channel takes the arithmetic path, not the short circuit."""
        gate = torch.zeros(1, 16, 232)
        gate[0, 0, 0] = 1e-3
        self.assertFalse(gate_is_identity(gate, 0))
        out = compose_x0(self.outputs, gate)
        self.assertFalse(torch.equal(out, self.hoi))
        # ...and only that one channel moved.
        self.assertTrue(torch.equal(out[0, 0, 1:], self.hoi[0, 0, 1:]))

    def test_a_subnormal_gate_takes_the_arithmetic_path_and_still_rounds_to_hoi(self):
        """Measured, and the reason the anchor is a short circuit rather than a limit.

        At 1e-8 in float32, ``(1 - gate)`` rounds to exactly 1.0 and ``gate * hsi``
        underflows relative to it, so the arithmetic path returns the HOI tensor
        bit for bit anyway -- the two paths agree here.  They do NOT agree when the
        unused expert is non-finite, which is why A1 cannot be left to rounding.
        """
        gate = torch.zeros(1, 16, 232)
        gate[0, 0, 0] = 1e-8
        self.assertFalse(gate_is_identity(gate, 0))
        self.assertTrue(torch.equal(compose_x0(self.outputs, gate), self.hoi))
        poisoned = ExpertOutputs(hoi=self.hoi, hsi=torch.full_like(self.hsi, float('nan')))
        self.assertTrue(torch.isnan(compose_x0(poisoned, gate)[0, 0, 0]))

    def test_rejects_out_of_range_and_mismatched_gate(self):
        for bad in (-0.1, 1.1, torch.full((1, 16, 232), -1.0)):
            with self.assertRaises(ValueError):
                compose_x0(self.outputs, bad)
        with self.assertRaises(ValueError):
            compose_x0(self.outputs, torch.zeros(1, 16, 7) + 0.5)

    def test_rejects_shape_disagreement_between_experts(self):
        with self.assertRaises(ValueError):
            compose_x0(ExpertOutputs(hoi=self.hoi, hsi=torch.randn(1, 16, 231)), 0.5)

    def test_non_anchor_gate_needs_both_experts(self):
        with self.assertRaises(ValueError):
            compose_x0(ExpertOutputs(hoi=self.hoi), 0.5)


class ReservedStateTests(unittest.TestCase):
    """`state` exists so adding the LLM state machine changes no signature."""

    def test_state_is_accepted_as_a_parameter(self):
        import inspect
        self.assertIn('state', inspect.signature(compose_x0).parameters)

    def test_supplying_a_state_raises_rather_than_being_ignored(self):
        outputs = ExpertOutputs(hoi=torch.zeros(1, 16, 232), hsi=torch.ones(1, 16, 232))
        with self.assertRaises(NotImplementedError):
            compose_x0(outputs, 0.5, state='walk')
        # ...including on the anchor path, so no caller can come to rely on the
        # anchor silently ignoring a state it was given.
        with self.assertRaises(NotImplementedError):
            compose_x0(outputs, 0, state='walk')


class ExpertOutputsTests(unittest.TestCase):
    def test_present_reports_only_supplied_experts(self):
        t = torch.zeros(1, 16, 232)
        self.assertEqual(ExpertOutputs().present(), ())
        self.assertEqual(ExpertOutputs(hoi=t).present(), ('hoi',))
        self.assertEqual(ExpertOutputs(hsi=t).present(), ('hsi',))
        self.assertEqual(ExpertOutputs(hoi=t, hsi=t).present(), ('hoi', 'hsi'))

    def test_absent_expert_is_none_not_zeros(self):
        """A zero tensor would halve an anchor; absence must be distinguishable."""
        self.assertIsNone(ExpertOutputs().hoi)
        self.assertIsNone(ExpertOutputs().hsi)

    def test_gate_is_identity_rejects_a_non_anchor_value(self):
        with self.assertRaises(ValueError):
            gate_is_identity(0.5, 0.5)

    def test_gate_is_identity_on_an_empty_tensor_is_false(self):
        """torch.all on an empty tensor is True; that must not become an anchor."""
        self.assertFalse(gate_is_identity(torch.zeros(0), 0))


class ChannelMaskTests(unittest.TestCase):
    """The gate is masked to the human channels, and why.

    HSI is never supervised on 216:232 -- priors/hsi/data.py calls codec.encode
    with no object arguments and window_codec.py starts from torch.zeros -- so
    blending there pulls toward the origin of the normalized box rather than
    toward a second opinion.
    """

    def setUp(self):
        self.hoi = torch.full((2, 16, 232), 0.5)
        self.hsi = torch.full((2, 16, 232), -0.5)

    def test_hsi_training_target_on_the_object_channels_is_exactly_zero(self):
        """The premise of the mask, asserted against the real codec.

        If this ever fails, HSI has gained object supervision and the mask is a
        choice again rather than a consequence.
        """
        import numpy as np

        from priors.core.window_codec import WindowStateCodec

        norm = np.load(REPO / 'data' / 'train' / 'norm.npy')
        codec = WindowStateCodec(
            torch.from_numpy(norm[0].astype('float32')),
            torch.from_numpy(norm[1].astype('float32')),
        )
        joints = torch.zeros(16, 28, 3)
        joints[:, :, 1] = 1.0
        joints[:, 0, 0] = torch.linspace(0, 1, 16)
        rotations = torch.eye(3).expand(16, 22, 3, 3).clone()
        encoded, _ = codec.encode(joints, rotations)
        self.assertEqual(encoded.shape[-1], 232)
        self.assertTrue(torch.all(encoded[..., OBJECT_CHANNEL_START:] == 0).item())

    def test_mask_is_one_on_human_channels_and_zero_after(self):
        mask = human_gate_mask()
        self.assertEqual(tuple(mask.shape), (232,))
        self.assertTrue(torch.all(mask[:OBJECT_CHANNEL_START] == 1).item())
        self.assertTrue(torch.all(mask[OBJECT_CHANNEL_START:] == 0).item())
        self.assertEqual(OBJECT_CHANNEL_START, 216)

    def test_object_channel_start_tracks_the_frozen_representation(self):
        self.assertEqual(
            OBJECT_CHANNEL_START, REPRESENTATION.field('object_translation').start
        )

    def test_default_mask_keeps_object_and_contact_from_hoi(self):
        composed = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 0.5)
        self.assertTrue(
            torch.equal(
                composed[..., OBJECT_CHANNEL_START:], self.hoi[..., OBJECT_CHANNEL_START:]
            )
        )
        # And the human channels really did blend.
        self.assertTrue(torch.allclose(composed[..., :OBJECT_CHANNEL_START],
                                       torch.zeros(2, 16, OBJECT_CHANNEL_START)))

    def test_mask_none_applies_the_gate_to_all_232_channels(self):
        composed = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 0.5,
                              channel_mask=None)
        self.assertTrue(torch.allclose(composed, torch.zeros(2, 16, 232)))

    def test_a_custom_mask_is_honoured(self):
        mask = torch.zeros(232)
        mask[7] = 1.0
        composed = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 1.0,
                              channel_mask=mask)
        # Gate 1.0 is an ANCHOR and short-circuits before the mask, so use a
        # non-anchor gate to exercise the mask itself.
        composed = compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 0.99,
                              channel_mask=mask)
        self.assertAlmostEqual(float(composed[0, 0, 7]), 0.5 - 0.99, places=5)
        self.assertAlmostEqual(float(composed[0, 0, 8]), 0.5, places=6)

    def test_both_anchors_ignore_the_mask_and_stay_bitwise(self):
        outputs = ExpertOutputs(hoi=self.hoi, hsi=self.hsi)
        self.assertIs(compose_x0(outputs, 0), self.hoi)
        self.assertIs(compose_x0(outputs, 1), self.hsi)
        self.assertIs(compose_x0(outputs, 0, channel_mask=None), self.hoi)
        self.assertIs(compose_x0(outputs, 1, channel_mask=None), self.hsi)

    def test_the_operator_is_discontinuous_at_the_hsi_anchor_under_the_mask(self):
        """Documented, not accidental: G==1 is HSI alone, its limit is not.

        The limit from below keeps HOI's object channels; exactly 1 returns HSI's
        never-supervised zeros there.  A learned gate reaches the anchor only by
        emitting exactly 1.0 on all 232 channels of every batch element.
        """
        outputs = ExpertOutputs(hoi=self.hoi, hsi=self.hsi)
        near = compose_x0(outputs, 1.0 - 1e-6)
        at = compose_x0(outputs, 1.0)
        self.assertTrue(
            torch.allclose(
                near[..., OBJECT_CHANNEL_START:], self.hoi[..., OBJECT_CHANNEL_START:],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.equal(
                at[..., OBJECT_CHANNEL_START:], self.hsi[..., OBJECT_CHANNEL_START:]
            )
        )

    def test_a_bad_mask_shape_raises(self):
        with self.assertRaises(ValueError):
            compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 0.5,
                       channel_mask=torch.ones(999))

    def test_an_unknown_mask_name_raises(self):
        with self.assertRaises(ValueError):
            compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 0.5,
                       channel_mask='everything')

    def test_a_mask_of_the_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            compose_x0(ExpertOutputs(hoi=self.hoi, hsi=self.hsi), 0.5,
                       channel_mask=7)

    def test_contact_scaling_is_what_the_mask_prevents(self):
        """The measured reason, as an assertion: unmasked, contact scales by (1-G).

        contact_percent is the metric the 15% budget is written against, so an
        unmasked gate would spend that budget on channels HSI has no opinion
        about.
        """
        hoi = torch.zeros(1, 16, 232)
        hoi[..., 228:232] = 1.0
        hsi = torch.zeros(1, 16, 232)
        unmasked = compose_x0(ExpertOutputs(hoi=hoi, hsi=hsi), 0.3, channel_mask=None)
        self.assertTrue(torch.allclose(unmasked[..., 228:232], torch.full((1, 16, 4), 0.7)))
        masked = compose_x0(ExpertOutputs(hoi=hoi, hsi=hsi), 0.3)
        self.assertTrue(torch.allclose(masked[..., 228:232], torch.ones(1, 16, 4)))


if __name__ == '__main__':
    unittest.main()
