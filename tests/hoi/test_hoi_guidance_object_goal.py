"""Preregistered Phase 1B P7: the object-goal terminal guidance term.

Two failure modes are defended against here, both silent.

1. A no-op.  The first P6 round ran a full GPU sweep that measured nothing
   because ``settings`` never reached the loss the gradient came from, while the
   audit faithfully reported the intended configuration.  A non-zero
   ``object_goal_weight`` that fails to change the gradient would repeat that.
2. A frame mismatch.  The codec decodes ``object_translation`` in GLOBAL
   coordinates (window_codec.py:251) while ``sample_step`` has already converted
   ``object_goal`` to the WINDOW-LOCAL frame (test_infbagel_hoi.py:415).
   Differencing them directly pulls the object toward a meaningless point and
   raises nothing.
"""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from guidance_loss import apply_hoi_guidance_loss
from priors.hoi.inference_guidance import (
    DEFAULT_OBJECT_GOAL_WEIGHT,
    GuidanceSettings,
    author_full_hoi_loss,
    object_goal_terminal_loss,
)
from priors.core.window_codec import WindowFrame, WindowStateCodec


def _inputs(batch=2, frames=16, vertices=64, seed=42):
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


class TerminalLossTest(unittest.TestCase):
    def test_scores_only_the_final_frame(self):
        """end_obj_trans_err scores one frame; earlier frames must not contribute."""
        translation = torch.zeros(2, 16, 3)
        goal = torch.zeros(2, 3)
        translation[:, -1, :] = 3.0
        far = float(object_goal_terminal_loss(translation, goal))
        # Perturbing every frame EXCEPT the last must leave the value alone.
        translation[:, :-1, :] = 99.0
        self.assertAlmostEqual(
            float(object_goal_terminal_loss(translation, goal)), far, places=6
        )

    def test_zero_at_the_goal(self):
        translation = torch.randn(3, 16, 3)
        goal = translation[:, -1, :].clone()
        self.assertAlmostEqual(
            float(object_goal_terminal_loss(translation, goal)), 0.0, places=9
        )

    def test_matches_squared_distance_times_batch(self):
        translation = torch.zeros(2, 8, 3)
        translation[:, -1, :] = torch.tensor([3.0, 4.0, 0.0])
        goal = torch.zeros(2, 3)
        # 3-4-5 triangle: squared distance 25 per sample, mean 25, times batch 2.
        self.assertAlmostEqual(
            float(object_goal_terminal_loss(translation, goal)), 50.0, places=5
        )

    def test_rejects_malformed_translation(self):
        with self.assertRaises(ValueError):
            object_goal_terminal_loss(torch.zeros(2, 3), torch.zeros(2, 3))


class DefaultInertnessTest(unittest.TestCase):
    """The sealed path must not shift: the term is off unless asked for."""

    def test_default_weight_is_zero(self):
        self.assertEqual(DEFAULT_OBJECT_GOAL_WEIGHT, 0.0)
        self.assertTrue(
            GuidanceSettings(enabled=True).uses_default_hand_decomposition
        )

    def test_nonzero_weight_leaves_the_author_default_path(self):
        """Otherwise the term would be silently discarded by the author branch."""
        self.assertFalse(
            GuidanceSettings(enabled=True, object_goal_weight=1.0)
            .uses_default_hand_decomposition
        )

    def test_default_settings_still_match_the_author_exactly(self):
        inputs = _inputs()
        self.assertEqual(
            float(author_full_hoi_loss(**inputs, settings=GuidanceSettings(enabled=True))),
            float(apply_hoi_guidance_loss(
                inputs["fk_joints"], inputs["object_vertices"],
                inputs["object_translation"], inputs["object_rotation"],
                inputs["contact"], None, None,
            )),
        )

    def test_zero_weight_ignores_a_supplied_goal(self):
        inputs = _inputs()
        baseline = float(
            author_full_hoi_loss(**inputs, settings=GuidanceSettings(enabled=True))
        )
        with_goal = float(author_full_hoi_loss(
            **inputs,
            settings=GuidanceSettings(enabled=True, contact_weight=3.0),
            object_goal=torch.randn(2, 3),
        ))
        without = float(author_full_hoi_loss(
            **inputs, settings=GuidanceSettings(enabled=True, contact_weight=3.0),
        ))
        self.assertAlmostEqual(with_goal, without, places=5)
        self.assertNotAlmostEqual(with_goal, baseline, places=5)


class NonZeroWeightTakesEffectTest(unittest.TestCase):
    """Guard against the P6 no-op: the weight must move loss AND gradient."""

    def test_missing_goal_raises_rather_than_silently_dropping(self):
        with self.assertRaises(ValueError):
            author_full_hoi_loss(
                **_inputs(),
                settings=GuidanceSettings(enabled=True, object_goal_weight=1.0),
                object_goal=None,
            )

    def test_weight_changes_the_loss(self):
        inputs = _inputs()
        goal = torch.randn(2, 3)
        values = []
        for weight in (0.0, 1.0, 10.0):
            values.append(float(author_full_hoi_loss(
                **inputs,
                settings=GuidanceSettings(
                    enabled=True, contact_weight=3.0, object_goal_weight=weight,
                ),
                object_goal=goal,
            )))
        self.assertNotAlmostEqual(values[0], values[1], places=4)
        self.assertNotAlmostEqual(values[1], values[2], places=4)

    def test_weight_scales_the_term_linearly(self):
        inputs = _inputs()
        goal = torch.randn(2, 3)

        def loss(weight):
            return float(author_full_hoi_loss(
                **inputs,
                settings=GuidanceSettings(
                    enabled=True, contact_weight=3.0, object_goal_weight=weight,
                ),
                object_goal=goal,
            ))

        term = float(object_goal_terminal_loss(inputs["object_translation"], goal))
        base = loss(0.0)
        for weight in (2.0, 5.0):
            self.assertAlmostEqual(
                loss(weight) - base, weight * term,
                delta=max(abs(weight * term) * 1e-5, 1e-4),
            )

    def test_gradient_reaches_the_object_translation(self):
        inputs = _inputs()
        goal = torch.randn(2, 3)
        leaf = inputs["object_translation"].detach().clone().requires_grad_(True)
        arguments = dict(inputs, object_translation=leaf)
        loss = author_full_hoi_loss(
            **arguments,
            settings=GuidanceSettings(
                enabled=True, contact_weight=3.0, object_goal_weight=5.0,
            ),
            object_goal=goal,
        )
        gradient = torch.autograd.grad(-loss, leaf)[0]
        # The terminal frame must carry gradient from this term.
        self.assertGreater(float(gradient[:, -1, :].norm()), 0.0)


class ObjectAlreadyReceivesGradientTest(unittest.TestCase):
    """Records a corrected claim: the author's hinge does NOT detach the object.

    I previously asserted repeatedly that guidance never moves the object. That
    holds only for the consistency term (guidance_loss.py:42-47); the contact
    hinge at :38 consumes non-detached ``obj_verts``, so object translation and
    rotation already receive guidance gradient.
    """

    def test_author_loss_gives_the_object_gradient(self):
        generator = torch.Generator().manual_seed(42)
        batch, frames, vertices = 2, 16, 64
        rest = torch.randn(batch, vertices, 3, generator=generator)
        translation = torch.randn(batch, frames, 3, generator=generator).requires_grad_(True)
        rotation, _ = torch.linalg.qr(
            torch.randn(batch, frames, 3, 3, generator=generator)
        )
        rotation = rotation.detach().requires_grad_(True)
        verts = torch.einsum("bvc,btdc->btvd", rest, rotation) + translation[:, :, None]
        loss = apply_hoi_guidance_loss(
            torch.randn(batch, frames, 24, 3, generator=generator),
            verts, translation, rotation,
            torch.ones(batch, frames, 4),  # every frame engaged
            None, None,
        )
        gradient_translation, gradient_rotation = torch.autograd.grad(
            loss, [translation, rotation], allow_unused=True,
        )
        self.assertIsNotNone(gradient_translation)
        self.assertGreater(float(gradient_translation.norm()), 0.0)
        self.assertIsNotNone(gradient_rotation)
        self.assertGreater(float(gradient_rotation.norm()), 0.0)


class GoalFrameAlignmentTest(unittest.TestCase):
    """The goal must be lifted to the frame the decode returns, not differenced raw."""

    def _frame(self, batch=2):
        generator = torch.Generator().manual_seed(7)
        rotation, _ = torch.linalg.qr(
            torch.randn(batch, 3, 3, generator=generator)
        )
        reference, _ = torch.linalg.qr(
            torch.randn(batch, 3, 3, generator=generator)
        )
        return WindowFrame(
            torch.randn(batch, 3, generator=generator),
            rotation.detach(),
            reference.detach(),
        )

    def test_local_to_global_round_trip(self):
        """Lifting a local goal must invert the decode's own transform."""
        frame = self._frame()
        local = torch.randn(2, 1, 3)
        world = WindowStateCodec.global_position(local, frame)
        # Invert: (world - origin) @ world_to_local^T recovers the local point.
        recovered = (world - frame.origin[:, None]) @ frame.world_to_local.transpose(-1, -2)
        self.assertTrue(torch.allclose(recovered, local, atol=1e-5))

    def test_raw_local_goal_differs_from_the_lifted_one(self):
        """If these agreed, the frame bug would be undetectable."""
        frame = self._frame()
        local = torch.randn(2, 1, 3)
        world = WindowStateCodec.global_position(local, frame).reshape(2, 3)
        self.assertFalse(
            torch.allclose(world, local.reshape(2, 3), atol=1e-3),
            msg="the lifted goal coincides with the local one; the test frame is degenerate",
        )

    def test_terminal_loss_is_zero_only_in_the_matching_frame(self):
        frame = self._frame()
        local_goal = torch.randn(2, 1, 3)
        world_goal = WindowStateCodec.global_position(local_goal, frame).reshape(2, 3)
        translation = torch.zeros(2, 4, 3)
        translation[:, -1, :] = world_goal
        self.assertAlmostEqual(
            float(object_goal_terminal_loss(translation, world_goal)), 0.0, places=6
        )
        self.assertGreater(
            float(object_goal_terminal_loss(translation, local_goal.reshape(2, 3))), 0.0
        )


class ValidationTest(unittest.TestCase):
    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, object_goal_weight=-1.0)

    def test_nonfinite_weight_raises(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GuidanceSettings(enabled=True, object_goal_weight=value)

    def test_config_round_trip(self):
        settings = GuidanceSettings.from_config({
            "enabled": True, "arm": "b", "contact_weight": 3.0,
            "object_goal_weight": 25.0,
        })
        self.assertEqual(settings.object_goal_weight, 25.0)
        self.assertEqual(settings.contact_weight, 3.0)
        self.assertFalse(settings.uses_default_hand_decomposition)


if __name__ == "__main__":
    unittest.main()
