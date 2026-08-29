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

from mixer.composition import ExpertOutputs, compose_x0, gate_is_identity  # noqa: E402


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

    def test_scalar_half_is_the_midpoint(self):
        out = compose_x0(self.outputs, 0.5)
        self.assertTrue(torch.allclose(out, 0.5 * (self.hoi + self.hsi)))

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


if __name__ == '__main__':
    unittest.main()
