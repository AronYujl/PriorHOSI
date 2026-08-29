"""HOSI-test evaluator determinism and released-arithmetic identity.

Two properties, both measured rather than asserted by inspection:

G1  The per-episode object-vertex subsample draws NOTHING from the global RNG, so
    an episode's generated motion no longer depends on how many metric calls ran
    before it, and an episode evaluated alone matches the same episode evaluated
    inside the full sweep.

G2  ``occ_list_layout_repaired=False`` -- the default -- reproduces the released
    ``occ_list`` entry-0 layout bit for bit, and True is the transposed repair.
    Every existing checkpoint was trained under the released layout, so the
    default must not move.
"""

import hashlib
import os
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
os.environ.setdefault("ROOT_DIR", str(REPO))


def _subsample_seed_reference(seed, scene_name, test_idx):
    digest = hashlib.sha256(f'{seed}|{scene_name}|{test_idx}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big', signed=False) & ((1 << 63) - 1)


class SubsampleSeedTests(unittest.TestCase):
    """G1: the subsample is episode-keyed and global-RNG-free."""

    def setUp(self):
        import test_infbagel_hosi as evaluator
        self.evaluator = evaluator

    def test_seed_is_keyed_on_episode_identity_not_on_call_order(self):
        seed_fn = self.evaluator._subsample_seed
        a = seed_fn(42, "scene_a", 3)
        b = seed_fn(42, "scene_a", 3)
        self.assertEqual(a, b, "same episode must give the same seed")
        self.assertEqual(a, _subsample_seed_reference(42, "scene_a", 3))
        # distinct episodes and distinct run seeds separate
        self.assertNotEqual(a, seed_fn(42, "scene_a", 4))
        self.assertNotEqual(a, seed_fn(42, "scene_b", 3))
        self.assertNotEqual(a, seed_fn(43, "scene_a", 3))
        # usable as a torch seed
        for value in (a, seed_fn(0, "", 0), seed_fn(2 ** 31, "x" * 200, 10 ** 6)):
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 1 << 63)
            torch.Generator(device="cpu").manual_seed(value)

    def test_subsample_does_not_advance_the_global_rng(self):
        """The defect: a global draw here shifted every later episode's noise.

        Reproduces the evaluator's interleaving -- sample, then metric -- and
        requires the sampler's stream to be untouched by the metric.
        """
        nv, keep = 17996, 10475          # clothesstand, the smallest of the 13 objects

        def stream(with_metric_draw):
            torch.manual_seed(42)
            out = []
            for episode in range(4):
                out.append(torch.randn(1, 16, 232))          # stands in for sample_step
                if with_metric_draw:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(_subsample_seed_reference(42, "s", episode))
                    torch.randperm(nv, generator=generator)[:keep]
            return out

        for i, (a, b) in enumerate(zip(stream(False), stream(True))):
            self.assertTrue(
                torch.equal(a, b),
                f"episode {i} noise moved, so the metric still draws from the global RNG",
            )

    def test_a_global_draw_would_have_been_caught(self):
        """Negative control: the previous code must fail the check above."""
        nv, keep = 17996, 10475

        def stream(with_metric_draw):
            torch.manual_seed(42)
            out = []
            for _ in range(4):
                out.append(torch.randn(1, 16, 232))
                if with_metric_draw:
                    torch.randperm(nv)[:keep]                 # the defect
            return out

        clean, polluted = stream(False), stream(True)
        self.assertTrue(torch.equal(clean[0], polluted[0]), "episode 0 is before the draw")
        self.assertFalse(
            all(torch.equal(a, b) for a, b in zip(clean[1:], polluted[1:])),
            "the negative control did not reproduce the defect, so it proves nothing",
        )


class NoGlobalRngDrawsTests(unittest.TestCase):
    """G1, structurally: no stochastic call in the evaluator omits `generator=`.

    Checked on the AST, not by grep: a textual scan misses continuation lines, and
    all three of these calls are now wrapped across lines.
    """

    def test_every_stochastic_call_passes_a_generator(self):
        import ast

        source = (REPO / "code" / "test_infbagel_hosi.py").read_text()
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in ("randperm", "randn", "randint", "multinomial") or (
                name == "rand" and isinstance(node.func, ast.Attribute)
            ):
                if not any(kw.arg == "generator" for kw in node.keywords):
                    offenders.append((node.lineno, name))
        self.assertEqual(
            offenders, [],
            f"these draws still use the global RNG and will make episodes "
            f"order-dependent: {offenders}",
        )

    def test_the_three_known_draw_sites_are_still_present(self):
        """Guard the guard: if the draws move or vanish, the AST test above passes
        vacuously, so pin that there are still exactly three of them."""
        import ast

        source = (REPO / "code" / "test_infbagel_hosi.py").read_text()
        sites = [
            node.lineno for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "randperm"
        ]
        self.assertEqual(
            len(sites), 3,
            "expected 3 randperm sites (1 metric subsample + 2 object-point "
            f"conditioning draws), found {len(sites)} at {sites}",
        )


class OccListLayoutGateTests(unittest.TestCase):
    """G2: the default is the released layout, bit for bit."""

    def _sampler(self, **kwargs):
        from models.infbagel import Sampler
        return Sampler(
            device="cpu", mask_ind=0, emb_f=0, batch_size=2, channel=232,
            auto_regre_num=2, timesteps=500, ddim_timesteps=25, cm_timesteps=16,
            **kwargs,
        )

    def test_default_is_the_released_layout(self):
        sampler = self._sampler()
        self.assertFalse(sampler.occ_list_layout_repaired)
        occ = torch.randn(2, 32, 32, 32)
        entry0 = sampler._occ_list_entry0(occ)
        self.assertTrue(
            torch.equal(entry0, occ),
            "the default moved: it must reproduce the released occ_list[0] exactly",
        )

    def test_repaired_transposes_x_and_y(self):
        sampler = self._sampler(occ_list_layout_repaired=True)
        self.assertTrue(sampler.occ_list_layout_repaired)
        occ = torch.randn(2, 32, 32, 32)
        entry0 = sampler._occ_list_entry0(occ)
        self.assertTrue(torch.equal(entry0, occ.permute(0, 2, 1, 3)))
        self.assertFalse(torch.equal(entry0, occ), "a cubic grid must still differ")

    def test_repaired_entry0_matches_the_occ_temp_entries(self):
        """The point of the repair: entry 0 in the same layout as entries 1..N.

        ``_compute_occ`` permutes the occ_temp entries and the returned ``occ``, so
        under the repair entry 0 must use that same permutation.
        """
        sampler = self._sampler(occ_list_layout_repaired=True)
        occ = torch.randn(2, 32, 32, 32)
        self.assertTrue(
            torch.equal(sampler._occ_list_entry0(occ), occ.permute(0, 2, 1, 3)),
            "entry 0 must match the permutation the occ_temp entries already use",
        )

    def test_gate_accepts_the_omegaconf_false(self):
        for value in (False, None, 0):
            self.assertFalse(self._sampler(occ_list_layout_repaired=value).occ_list_layout_repaired)
        for value in (True, 1):
            self.assertTrue(self._sampler(occ_list_layout_repaired=value).occ_list_layout_repaired)


if __name__ == "__main__":
    unittest.main()
