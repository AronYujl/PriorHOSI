"""Pin the per-step guidance norm cap: inert when off, per-sample when on, B-only.

The cap is the single minimal B-side fix preregistered in
docs/plan/PHASE_1C_HSI.md 2026-08-23 SS-H, after the matched 2x2 tied 100% of the
physically impossible root accelerations to guidance.  Three properties have to hold or
the smoke that uses it means nothing:

  * off is an exact identity, so a capped build reproduces the sealed guided run;
  * the norm is reduced per sample, never once for the whole batch -- a batch-level
    branch keyed on sample 0 is exactly how layout neutrality was broken before;
  * only Sampler.p_sample applies it.  Sampler.cm_sample is the consistency path and the
    user's standing constraint is that C is neither modified nor retrained, so the AST
    guard here is what makes "C is untouched" a checked claim rather than a promise.

The cap arithmetic is tested through the real helper, not a re-implementation.
"""

import ast
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import cap_guidance_increment

SOURCE = (REPO / "code" / "models" / "infbagel.py").read_text()


def _increment(scale, batch=1, frames=16, channels=232, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, frames, channels, generator=generator) * scale


class CapArithmeticTests(unittest.TestCase):
    def test_none_is_the_identity_object(self):
        gradient = _increment(1.0)
        self.assertIs(cap_guidance_increment(gradient, None), gradient)

    def test_cap_above_the_norm_is_bitwise_identity(self):
        gradient = _increment(1.0)
        norm = float(gradient.flatten(1).norm(dim=1)[0])
        capped = cap_guidance_increment(gradient, norm * 2.0)
        self.assertTrue(torch.equal(capped, gradient))

    def test_cap_below_the_norm_scales_to_exactly_the_cap(self):
        gradient = _increment(1.0)
        norm = float(gradient.flatten(1).norm(dim=1)[0])
        cap = norm / 4.0
        capped = cap_guidance_increment(gradient, cap)
        self.assertAlmostEqual(float(capped.flatten(1).norm(dim=1)[0]), cap, places=4)

    def test_direction_is_preserved(self):
        gradient = _increment(1.0)
        norm = float(gradient.flatten(1).norm(dim=1)[0])
        capped = cap_guidance_increment(gradient, norm / 10.0)
        cosine = torch.nn.functional.cosine_similarity(
            capped.flatten(1), gradient.flatten(1), dim=1
        )
        self.assertAlmostEqual(float(cosine[0]), 1.0, places=5)

    def test_norm_is_reduced_per_sample_not_over_the_batch(self):
        # Sample 0 is far above the cap, sample 1 far below it.  Sample 1 must come back
        # bitwise unchanged: it is multiplied by exactly 1.0, not by the batch's factor.
        big = _increment(100.0, seed=1)
        small = _increment(0.001, seed=2)
        batched = torch.cat((big, small), dim=0)
        norms = batched.flatten(1).norm(dim=1)
        cap = float(norms[0]) / 10.0
        self.assertGreater(float(norms[0]), cap)
        self.assertLess(float(norms[1]), cap)

        capped = cap_guidance_increment(batched, cap)
        self.assertTrue(torch.equal(capped[1], batched[1]))
        self.assertAlmostEqual(float(capped[:1].flatten(1).norm(dim=1)[0]), cap, places=3)

    def test_single_sample_result_is_independent_of_the_batch_it_rode_in(self):
        # The layout-neutrality property itself: capping [big] alone and capping
        # [big, small] together must give the same row for big.
        big = _increment(100.0, seed=1)
        small = _increment(0.001, seed=2)
        cap = float(big.flatten(1).norm(dim=1)[0]) / 10.0
        alone = cap_guidance_increment(big, cap)
        together = cap_guidance_increment(torch.cat((big, small), dim=0), cap)
        self.assertTrue(torch.equal(alone[0], together[0]))

    def test_zero_increment_does_not_divide_by_zero(self):
        gradient = torch.zeros(1, 16, 232)
        capped = cap_guidance_increment(gradient, 1.0)
        self.assertTrue(torch.isfinite(capped).all())
        self.assertTrue(torch.equal(capped, gradient))


class CapCallSiteTests(unittest.TestCase):
    """Static guard: the diffusion path applies the cap, the consistency path does not."""

    def setUp(self):
        tree = ast.parse(SOURCE)
        sampler = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Sampler"
        )
        self.methods = {
            node.name: node
            for node in sampler.body
            if isinstance(node, ast.FunctionDef)
        }

    def _calls_cap(self, method):
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cap_guidance_increment"
            for node in ast.walk(self.methods[method])
        )

    def test_p_sample_applies_the_cap(self):
        self.assertIn("p_sample", self.methods)
        self.assertTrue(self._calls_cap("p_sample"))

    def test_cm_sample_does_not_apply_the_cap(self):
        self.assertIn("cm_sample", self.methods)
        self.assertFalse(
            self._calls_cap("cm_sample"),
            "the consistency path must stay untouched: C is neither modified nor retrained",
        )

    def test_cap_is_read_from_kwargs_in_init(self):
        init = self.methods["__init__"]
        self.assertIn("hsi_guidance_norm_cap", ast.dump(init))

    def test_only_the_two_known_autograd_grad_call_sites_exist(self):
        live = [
            line
            for line in SOURCE.splitlines()
            if "autograd.grad" in line and not line.strip().startswith("#")
        ]
        self.assertEqual(len(live), 2, live)


if __name__ == "__main__":
    unittest.main()
