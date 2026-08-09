"""Preregistered Phase 1B P6 guidance hand sub-term reweighting.

The load-bearing test here is the reconciliation: ``code/guidance_loss.py`` is
author code that is never edited, so
:func:`priors.inference_guidance.author_hand_subterms` reimplements the author's
arithmetic verbatim in order to expose the two halves of the hand term
separately.  If that reimplementation diverges from the author's own function by
any amount, every comparison against a sealed result silently becomes a
comparison against a different objective.  ``test_reconciles_with_author_loss``
pins it to exact float32 equality.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from guidance_loss import (
    apply_feet_floor_contact_guidance,
    apply_hoi_guidance_loss,
)
from priors.inference_guidance import (
    AUTHOR_FEET_WEIGHT,
    AUTHOR_HAND_WEIGHT,
    CONSISTENCY_NORMALIZATION_AUTHOR,
    CONSISTENCY_NORMALIZATION_MASKED_PAIRS,
    DEFAULT_CONSISTENCY_NORMALIZATION,
    DEFAULT_CONSISTENCY_WEIGHT,
    DEFAULT_CONTACT_WEIGHT,
    GuidanceAudit,
    GuidanceSettings,
    author_full_hoi_loss,
    author_hand_subterms,
)


# (batch, frames, object vertices).  The single-frame and batch>1 cases matter:
# the author multiplies by ``bs`` and normalises the consistency term by every
# T*T pair, so a reimplementation can agree at one shape and diverge at another.
SHAPES = ((1, 10, 64), (2, 16, 128), (3, 8, 32), (1, 1, 16), (4, 20, 96))


def _inputs(batch, frames, vertices, seed=42):
    """Shape-valid guidance inputs in the author's conventions."""
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=generator)

    rotation, _ = torch.linalg.qr(normal(batch, frames, 3, 3))
    return {
        "fk_joints": normal(batch, frames, 24, 3),
        "object_vertices": normal(batch, frames, vertices, 3),
        "object_translation": normal(batch, frames, 3),
        "object_rotation": rotation,
        "contact": torch.rand(batch, frames, 4, generator=generator),
    }


def _author_loss(inputs):
    return apply_hoi_guidance_loss(
        inputs["fk_joints"],
        inputs["object_vertices"],
        inputs["object_translation"],
        inputs["object_rotation"],
        inputs["contact"],
        None,
        None,
    )


def _contact_with_frames_on(batch, frames, on_frames):
    """Contact whose two consumed channels are on for exactly ``on_frames``."""
    contact = torch.zeros(batch, frames, 4)
    contact[:, :on_frames, 0] = 1.0
    contact[:, :on_frames, 1] = 1.0
    return contact


class AuthorReconciliationTest(unittest.TestCase):
    def test_reconciles_with_author_loss(self):
        """10*bs*(contact+consistency) + 500*feet == the author's own scalar."""
        worst = 0.0
        for batch, frames, vertices in SHAPES:
            inputs = _inputs(batch, frames, vertices)
            author = _author_loss(inputs)
            contact_term, consistency_term = author_hand_subterms(**inputs)
            rebuilt = (
                AUTHOR_HAND_WEIGHT * batch * (contact_term + consistency_term)
                + AUTHOR_FEET_WEIGHT
                * apply_feet_floor_contact_guidance(inputs["fk_joints"])
            )
            worst = max(worst, abs(float(author - rebuilt)))
            self.assertEqual(
                float(author),
                float(rebuilt),
                msg=(
                    f"reimplementation diverges at bs={batch} T={frames} "
                    f"V={vertices}: author={float(author)!r} "
                    f"rebuilt={float(rebuilt)!r}"
                ),
            )
        self.assertEqual(worst, 0.0)

    def test_subterms_are_finite_and_positive(self):
        for batch, frames, vertices in SHAPES:
            contact_term, consistency_term = author_hand_subterms(
                **_inputs(batch, frames, vertices)
            )
            self.assertTrue(torch.isfinite(contact_term))
            self.assertTrue(torch.isfinite(consistency_term))
            self.assertGreaterEqual(float(contact_term), 0.0)


class DefaultPathIdentityTest(unittest.TestCase):
    """The sealed path must remain the author's own call, not an equal-valued one."""

    def test_none_settings_match_author(self):
        for batch, frames, vertices in SHAPES:
            inputs = _inputs(batch, frames, vertices)
            self.assertEqual(
                float(author_full_hoi_loss(**inputs)),
                float(_author_loss(inputs)),
            )

    def test_default_settings_match_author(self):
        settings = GuidanceSettings(enabled=True)
        self.assertTrue(settings.uses_default_hand_decomposition)
        for batch, frames, vertices in SHAPES:
            inputs = _inputs(batch, frames, vertices)
            self.assertEqual(
                float(author_full_hoi_loss(**inputs, settings=settings)),
                float(_author_loss(inputs)),
            )

    def test_gradient_identity_at_defaults(self):
        """Bitwise-equal gradients, since the sample is what the gradient moves.

        ``guidance_gradient`` needs a codec and a window frame to decode a state,
        which is disproportionate here; the quantity that matters is the gradient
        of ``-loss`` with respect to the tensor being guided, so this asserts it
        directly on the FK leaf.
        """
        for batch, frames, vertices in SHAPES:
            inputs = _inputs(batch, frames, vertices)

            def gradient(loss_fn):
                leaf = inputs["fk_joints"].detach().clone().requires_grad_(True)
                arguments = dict(inputs, fk_joints=leaf)
                return torch.autograd.grad(-loss_fn(arguments), leaf)[0]

            author = gradient(_author_loss)
            routed = gradient(
                lambda arguments: author_full_hoi_loss(
                    **arguments, settings=GuidanceSettings(enabled=True)
                )
            )
            self.assertTrue(torch.equal(author, routed))


class WeightArithmeticTest(unittest.TestCase):
    def setUp(self):
        self.batch, self.frames, self.vertices = 2, 16, 128
        self.inputs = _inputs(self.batch, self.frames, self.vertices)
        self.contact_term, self.consistency_term = author_hand_subterms(**self.inputs)
        self.feet = apply_feet_floor_contact_guidance(self.inputs["fk_joints"])

    def _loss(self, **overrides):
        settings = GuidanceSettings(enabled=True, **overrides)
        return author_full_hoi_loss(**self.inputs, settings=settings)

    def test_zero_consistency_weight_leaves_contact_and_feet(self):
        expected = (
            AUTHOR_HAND_WEIGHT * self.batch * self.contact_term
            + AUTHOR_FEET_WEIGHT * self.feet
        )
        self.assertAlmostEqual(
            float(self._loss(consistency_weight=0.0)),
            float(expected),
            delta=abs(float(expected)) * 1e-6,
        )

    def test_zero_contact_weight_leaves_consistency_and_feet(self):
        expected = (
            AUTHOR_HAND_WEIGHT * self.batch * self.consistency_term
            + AUTHOR_FEET_WEIGHT * self.feet
        )
        self.assertAlmostEqual(
            float(self._loss(contact_weight=0.0)),
            float(expected),
            delta=abs(float(expected)) * 1e-6,
        )

    def test_contact_weight_scales_linearly(self):
        """The increment must match ``10*bs*(k-1)*loss_contact``.

        The tolerance is relative to the TOTAL loss, not to the increment: the
        feet term puts the total near 1e3 while the increment is near 4e-1, so
        differencing two totals loses the low bits to catastrophic cancellation.
        float32 resolution at 1e3 is about 1.3e-4, which is larger than the
        agreement we can demand of a 4e-1 difference.
        """
        baseline = float(self._loss(contact_weight=1.0))
        resolution = torch.finfo(torch.float32).eps * abs(baseline)
        for factor in (2.0, 10.0):
            increment = float(self._loss(contact_weight=factor)) - baseline
            expected = float(
                AUTHOR_HAND_WEIGHT * self.batch * (factor - 1.0) * self.contact_term
            )
            self.assertAlmostEqual(
                increment, expected, delta=max(resolution, abs(expected) * 1e-5),
            )

    def test_both_weights_zero_leaves_only_feet(self):
        expected = AUTHOR_FEET_WEIGHT * self.feet
        self.assertAlmostEqual(
            float(self._loss(contact_weight=0.0, consistency_weight=0.0)),
            float(expected),
            delta=abs(float(expected)) * 1e-6,
        )


class ConsistencyNormalizationTest(unittest.TestCase):
    """The author divides by every T*T pair; masked_pairs divides by the mask-on ones.

    Per hand the author's mean is ``S / (B*T*T)`` and the masked form is
    ``S / (B*k*k)``, so with both hands sharing the same ``k`` the two assembled
    terms satisfy ``(2 - masked) == (2 - author) * T*T / (k*k)``.
    """

    def test_masked_pairs_rescales_by_pair_count(self):
        batch, frames, vertices = 2, 12, 64
        inputs = _inputs(batch, frames, vertices)
        for on_frames in (3, 5, 8):
            inputs = dict(
                inputs, contact=_contact_with_frames_on(batch, frames, on_frames)
            )
            _, author = author_hand_subterms(
                **inputs, consistency_normalization=CONSISTENCY_NORMALIZATION_AUTHOR
            )
            _, masked = author_hand_subterms(
                **inputs,
                consistency_normalization=CONSISTENCY_NORMALIZATION_MASKED_PAIRS,
            )
            ratio = (frames * frames) / (on_frames * on_frames)
            self.assertAlmostEqual(
                float(2.0 - masked),
                float(2.0 - author) * ratio,
                delta=max(abs(float(2.0 - author) * ratio) * 1e-5, 1e-6),
                msg=f"k={on_frames} of T={frames}",
            )

    def test_full_mask_is_identical_to_author(self):
        batch, frames, vertices = 2, 12, 64
        inputs = dict(
            _inputs(batch, frames, vertices),
            contact=_contact_with_frames_on(batch, frames, frames),
        )
        _, author = author_hand_subterms(
            **inputs, consistency_normalization=CONSISTENCY_NORMALIZATION_AUTHOR
        )
        _, masked = author_hand_subterms(
            **inputs, consistency_normalization=CONSISTENCY_NORMALIZATION_MASKED_PAIRS
        )
        self.assertAlmostEqual(float(author), float(masked), delta=1e-6)

    def test_empty_mask_does_not_divide_by_zero(self):
        batch, frames, vertices = 2, 12, 64
        inputs = dict(
            _inputs(batch, frames, vertices),
            contact=_contact_with_frames_on(batch, frames, 0),
        )
        for normalization in (
            CONSISTENCY_NORMALIZATION_AUTHOR,
            CONSISTENCY_NORMALIZATION_MASKED_PAIRS,
        ):
            _, consistency = author_hand_subterms(
                **inputs, consistency_normalization=normalization
            )
            self.assertTrue(torch.isfinite(consistency))
            self.assertAlmostEqual(float(consistency), 2.0, delta=1e-6)

    def test_unknown_normalization_raises(self):
        inputs = _inputs(1, 8, 32)
        with self.assertRaises(ValueError):
            author_hand_subterms(**inputs, consistency_normalization="mean")


class SettingsValidationTest(unittest.TestCase):
    def test_negative_weights_raise(self):
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, contact_weight=-1.0)
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, consistency_weight=-0.5)

    def test_nonfinite_weights_raise(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GuidanceSettings(enabled=True, contact_weight=value)
            with self.assertRaises(ValueError):
                GuidanceSettings(enabled=True, consistency_weight=value)

    def test_unknown_normalization_raises(self):
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, consistency_normalization="mean")

    def test_default_decomposition_predicate(self):
        self.assertTrue(GuidanceSettings(enabled=True).uses_default_hand_decomposition)
        for overrides in (
            {"contact_weight": 3.0},
            {"consistency_weight": 0.0},
            {"consistency_normalization": CONSISTENCY_NORMALIZATION_MASKED_PAIRS},
        ):
            self.assertFalse(
                GuidanceSettings(enabled=True, **overrides)
                .uses_default_hand_decomposition,
                msg=str(overrides),
            )

    def test_config_round_trip(self):
        settings = GuidanceSettings.from_config({
            "enabled": True,
            "arm": "b",
            "contact_weight": 3.0,
            "consistency_weight": 0.0,
            "consistency_normalization": CONSISTENCY_NORMALIZATION_MASKED_PAIRS,
        })
        self.assertEqual(settings.contact_weight, 3.0)
        self.assertEqual(settings.consistency_weight, 0.0)
        self.assertEqual(
            settings.consistency_normalization,
            CONSISTENCY_NORMALIZATION_MASKED_PAIRS,
        )
        self.assertFalse(settings.uses_default_hand_decomposition)
        exported = settings.as_dict()
        self.assertEqual(exported["contact_weight"], 3.0)
        self.assertEqual(exported["consistency_weight"], 0.0)

    def test_defaults_are_the_author_configuration(self):
        self.assertEqual(DEFAULT_CONTACT_WEIGHT, 1.0)
        self.assertEqual(DEFAULT_CONSISTENCY_WEIGHT, 1.0)
        self.assertEqual(
            DEFAULT_CONSISTENCY_NORMALIZATION, CONSISTENCY_NORMALIZATION_AUTHOR
        )


class AuditCompatibilityTest(unittest.TestCase):
    """Sealed artifacts were parsed with the pre-P6 key meanings."""

    def test_empty_audit_reports_none_for_means(self):
        report = GuidanceAudit().as_dict()
        for key in (
            "guidance_loss_mean",
            "guidance_feet_loss_mean",
            "guidance_hand_loss_mean",
            "guidance_loss_contact_mean",
            "guidance_loss_consistency_mean",
        ):
            self.assertIn(key, report)
            self.assertIsNone(report[key], msg=key)
        self.assertEqual(report["guidance_applied_steps"], 0)

    def test_hand_loss_mean_keeps_its_historical_definition(self):
        audit = GuidanceAudit()
        gradient = torch.randn(2, 16, 232, generator=torch.Generator().manual_seed(7))
        loss = torch.tensor(5063.738426)
        feet = torch.tensor(0.4373729493882921)
        audit.record(gradient, (gradient * 0.1).clamp(-1.0, 1.0), loss, feet)
        report = audit.as_dict()
        self.assertAlmostEqual(
            report["guidance_hand_loss_mean"],
            (float(loss) - AUTHOR_FEET_WEIGHT * float(feet)) / AUTHOR_HAND_WEIGHT,
            places=4,
        )
        self.assertAlmostEqual(
            report["guidance_feet_weighted_mean"],
            AUTHOR_FEET_WEIGHT * float(feet),
            places=4,
        )

    def test_subterm_and_saturation_keys_populate(self):
        audit = GuidanceAudit()
        gradient = torch.randn(2, 16, 232, generator=torch.Generator().manual_seed(11))
        update = (gradient * 5.0).clamp(-1.0, 1.0)
        saturated = (update.abs() >= 1.0 - 1e-6).sum()
        audit.record(
            gradient,
            update,
            torch.tensor(5063.738426),
            torch.tensor(0.4373729493882921),
            loss_contact=torch.tensor(0.02117535),
            loss_consistency=torch.tensor(1.39546636),
            clamp_saturated=saturated,
            clamp_elements=update.numel(),
        )
        report = audit.as_dict()
        self.assertAlmostEqual(report["guidance_loss_contact_mean"], 0.02117535, places=6)
        self.assertAlmostEqual(
            report["guidance_loss_consistency_mean"], 1.39546636, places=6
        )
        self.assertEqual(report["guidance_clamp_saturated_elements"], int(saturated))
        self.assertEqual(report["guidance_clamp_total_elements"], update.numel())
        self.assertAlmostEqual(
            report["guidance_clamp_saturation_fraction"],
            int(saturated) / update.numel(),
            places=9,
        )


class _StubCodec:
    """Decodes to fixed tensors so ``guidance_gradient`` can be driven directly.

    ``guidance_gradient`` needs a codec and a window frame.  The real ones pull a
    dataset and SMPL rest offsets; this stub returns the same decoded fields the
    author's loss consumes, keeping the differentiable path from the guided
    tensor through to the loss intact.
    """

    def __init__(self, inputs):
        self._inputs = inputs

    def decode(self, clean, frame):
        del frame
        batch, frames = self._inputs["object_translation"].shape[:2]
        # Route the guided tensor through the decode so the gradient is real.
        bias = clean.reshape(batch, frames, -1)[..., :1]
        return {
            "object_translation": self._inputs["object_translation"] + bias,
            "object_rotation": self._inputs["object_rotation"],
            "contact": self._inputs["contact"],
            "_fk": self._inputs["fk_joints"] + bias[..., None, :],
        }


class GuidanceGradientRoutingTest(unittest.TestCase):
    """The weights must reach the loss the GUIDED UPDATE is taken from.

    This is not redundant with :class:`WeightArithmeticTest`, and the difference
    is the whole point.  A first P6 round ran on GPU and silently measured
    nothing: ``guidance_gradient`` called ``author_full_hoi_loss`` without
    forwarding ``settings``, so the gradient always came from the author's
    unweighted loss while the audit faithfully reported the intended per-cell
    weights.  All four cells came back bit-exact to baseline (0 of 438 sequences
    differed) and the then-existing 52 tests passed, because they exercised
    ``author_full_hoi_loss`` directly and never drove ``guidance_gradient``.
    Tests that assert around the defective call site instead of through it are
    decorative; these drive the real function.
    """

    def setUp(self):
        self.inputs = _inputs(2, 16, 128)
        self.batch, self.frames = 2, 16

    def _run(self, settings):
        from priors.inference_guidance import guidance_gradient

        clean = torch.zeros(
            self.batch, self.frames, 8, dtype=torch.float32
        ).requires_grad_(False)
        codec = _StubCodec(self.inputs)
        with mock.patch(
            "priors.inference_guidance.decoded_fk_positions",
            side_effect=lambda decoded, *_: decoded["_fk"],
        ), mock.patch(
            "priors.inference_guidance.transformed_object_vertices",
            side_effect=lambda *_: self.inputs["object_vertices"],
        ):
            gradient, loss, _ = guidance_gradient(
                clean,
                codec=codec,
                frame=None,
                rest_human_offsets=torch.zeros(self.batch, 24, 3),
                parents_24=torch.zeros(24, dtype=torch.long),
                rest_vertices=torch.zeros(self.batch, 128, 3),
                settings=settings,
            )
        return gradient, float(loss)

    def test_nondefault_weights_change_the_guided_loss(self):
        _, baseline = self._run(GuidanceSettings(enabled=True))
        for overrides in (
            {"consistency_weight": 0.0},
            {"contact_weight": 3.0},
            {"contact_weight": 10.0},
            {"consistency_normalization": CONSISTENCY_NORMALIZATION_MASKED_PAIRS},
        ):
            _, loss = self._run(GuidanceSettings(enabled=True, **overrides))
            self.assertNotAlmostEqual(
                loss,
                baseline,
                delta=1e-4,
                msg=(
                    f"{overrides} left the guided loss unchanged: the settings "
                    "are not reaching the loss that produces the gradient"
                ),
            )

    def test_contact_weight_changes_the_guided_gradient(self):
        base_gradient, _ = self._run(GuidanceSettings(enabled=True))
        scaled, _ = self._run(GuidanceSettings(enabled=True, contact_weight=10.0))
        self.assertFalse(
            torch.equal(base_gradient, scaled),
            msg="contact_weight did not alter the gradient guidance_gradient returns",
        )

    def test_default_settings_still_match_the_author(self):
        """Forwarding must not disturb the sealed path."""
        _, routed = self._run(GuidanceSettings(enabled=True))
        _, unset = self._run(None)
        self.assertAlmostEqual(routed, unset, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
