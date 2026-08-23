"""Pin the two guidance DOSE knobs: off by default, B-only, and the decay's direction.

The 2026-08-23 norm-cap smoke measured the per-step increment distributions and killed the
magnitude hypothesis: B's increments are smaller than C's at every percentile up to p99.
What differs is the count -- 499 increments per window on the diffusion path against 15 on
the consistency path.  These two knobs attack that, and both must be inert by default.

The decay's direction is pinned explicitly because it is easy to state backwards:
alpha_cumprod rises toward 1 as t falls to 0, so (1 - alpha_cumprod) SUPPRESSES guidance on
the late low-noise steps and KEEPS it on the early high-noise ones.
"""

import ast
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import Sampler
from utils import extract

SOURCE = (REPO / "code" / "models" / "infbagel.py").read_text()


def _sampler(**kwargs):
    return Sampler(
        device="cpu", mask_ind=0, emb_f=None, batch_size=1, channel=232,
        auto_regre_num=1, timesteps=500, ddim_timesteps=25, cm_timesteps=16, **kwargs
    )


class DefaultsAreInertTests(unittest.TestCase):
    def test_both_knobs_default_off(self):
        sampler = _sampler()
        self.assertIsNone(sampler.hsi_guidance_dose_scale)
        self.assertIs(sampler.hsi_guidance_alpha_decay, False)
        self.assertIsNone(sampler.hsi_guidance_norm_cap)

    def test_knobs_are_independent(self):
        only_dose = _sampler(hsi_guidance_dose_scale=0.5)
        self.assertEqual(only_dose.hsi_guidance_dose_scale, 0.5)
        self.assertIs(only_dose.hsi_guidance_alpha_decay, False)
        only_decay = _sampler(hsi_guidance_alpha_decay=True)
        self.assertIsNone(only_decay.hsi_guidance_dose_scale)
        self.assertIs(only_decay.hsi_guidance_alpha_decay, True)

    def test_dose_scale_is_coerced_to_float(self):
        # a YAML string must not silently make the branch a no-op or a type error
        self.assertEqual(_sampler(hsi_guidance_dose_scale="0.25").hsi_guidance_dose_scale, 0.25)


class DecayDirectionTests(unittest.TestCase):
    """(1 - alpha_cumprod) must be ~0 on the last steps and ~1 on the first."""

    def setUp(self):
        self.sampler = _sampler()

    def _factor(self, t_value):
        t = torch.full((1,), t_value, dtype=torch.long)
        return float(1.0 - extract(self.sampler.alpha_cumprod, t, (1, 16, 232)))

    def test_late_low_noise_steps_are_suppressed(self):
        self.assertLess(self._factor(1), 0.01)

    def test_early_high_noise_steps_are_preserved(self):
        self.assertGreater(self._factor(499), 0.98)

    def test_factor_is_monotone_increasing_in_t(self):
        values = [self._factor(t) for t in (1, 50, 100, 250, 400, 499)]
        self.assertEqual(values, sorted(values))

    def test_factor_stays_in_the_unit_interval(self):
        for t in (0, 1, 100, 250, 499):
            self.assertGreaterEqual(self._factor(t), 0.0)
            self.assertLessEqual(self._factor(t), 1.0)


class DoseCallSiteTests(unittest.TestCase):
    """Static guard: both knobs live in p_sample only, so C stays untouched."""

    def setUp(self):
        tree = ast.parse(SOURCE)
        sampler = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef) and n.name == "Sampler")
        self.methods = {n.name: n for n in sampler.body if isinstance(n, ast.FunctionDef)}

    def _mentions(self, method, attribute):
        return any(
            isinstance(node, ast.Attribute) and node.attr == attribute
            for node in ast.walk(self.methods[method])
        )

    def test_p_sample_applies_both_knobs(self):
        self.assertTrue(self._mentions("p_sample", "hsi_guidance_dose_scale"))
        self.assertTrue(self._mentions("p_sample", "hsi_guidance_alpha_decay"))

    def test_cm_sample_applies_neither(self):
        for attribute in ("hsi_guidance_dose_scale", "hsi_guidance_alpha_decay",
                          "hsi_guidance_norm_cap"):
            self.assertFalse(
                self._mentions("cm_sample", attribute),
                "%s must not reach the consistency path: C is neither modified nor retrained"
                % attribute,
            )

    def test_no_training_method_reads_the_knobs(self):
        for method in ("p_losses", "consistency_loss"):
            for attribute in ("hsi_guidance_dose_scale", "hsi_guidance_alpha_decay",
                              "hsi_guidance_norm_cap"):
                self.assertFalse(self._mentions(method, attribute), (method, attribute))


if __name__ == "__main__":
    unittest.main()
