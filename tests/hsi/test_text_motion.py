"""Metric identities and paper-cohort semantics for native artifact readout."""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from priors.hsi.text_motion import (
    GEOMETRY_KEYS, frechet_samples, geometry_groups, generation_protocol,
    paired_mean_ratios, ratio_gate,
)


def test_fid_matches_full_covariance_formula():
    rng = np.random.RandomState(42)
    x, y = rng.randn(32, 7), rng.randn(32, 7) * 2 + 0.3
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    expected = np.square(x.mean(0) - y.mean(0)).sum() + np.trace(cx + cy - 2 * sqrtm(cx @ cy))
    actual = frechet_samples(torch.from_numpy(x), torch.from_numpy(y))
    np.testing.assert_allclose(float(actual), expected, atol=1e-10)


def test_rank_deficient_fid_translation_and_scale():
    x = torch.tensor(np.random.RandomState(42).randn(8, 20), dtype=torch.float64)
    shift = torch.arange(20, dtype=torch.float64) / 10
    expected_scale = x.mean(0).square().sum() + x.var(0).sum()
    actual = frechet_samples(torch.stack((x, x)), torch.stack((x + shift, 2 * x)))
    torch.testing.assert_close(actual, torch.stack((shift.square().sum(), expected_scale)), atol=1e-10, rtol=1e-10)


def test_groups_use_caption_and_inclusive_five_cm_threshold():
    keys = set().union(*GEOMETRY_KEYS.values())
    metrics = {name: {key: 1.0 for key in keys} for name in ("walk", "sit", "lie")}
    metrics["sit"]["last_dist"] = 0.05
    metrics["lie"]["last_dist"] = 0.051
    groups = geometry_groups(metrics, {"walk": "walk", "sit": "sit down", "lie": "lie down"})
    assert list(groups["locomotion"]) == ["walk"]
    assert groups["interactive"]["sit"]["success_last_5cm"] == 1.0
    assert groups["interactive"]["lie"]["success_last_5cm"] == 0.0


def test_generation_protocol_uses_each_artifacts_sampler():
    for mode, steps in (("diffusion", 500), ("consistency", 16), ("ddim", 25)):
        payload = {"sample_type": mode, "guided": True, "seed": 42,
                   "timing": {"sampler_steps_per_window": steps}}
        result = generation_protocol(payload)
        assert result["sample_type"] == mode and result["sampler_steps"] == steps


def test_paired_ratio_keeps_exact_multiplicative_effect_for_every_resample():
    x = torch.tensor([[1., 4.], [2., 1.], [8., 3.], [5., 9.]], dtype=torch.float64)
    scale = torch.tensor([0.8, 1.2], dtype=torch.float64)
    point, interval = paired_mean_ratios(x, x * scale, replicates=500)
    torch.testing.assert_close(point, scale, atol=1e-12, rtol=0)
    torch.testing.assert_close(interval, scale.expand(2, -1), atol=1e-12, rtol=0)


def test_ratio_gate_distinguishes_equivalence_failure_and_uncertainty():
    assert ratio_gate(1, [0.95, 1.05], False)["status"] == "PASS"
    assert ratio_gate(1, [0.95, 1.05], True)["status"] == "PASS"
    assert ratio_gate(1.1, [1.06, 1.2], False)["status"] == "FAIL"
    assert ratio_gate(0.9, [0.8, 0.94], True)["status"] == "FAIL"
    assert ratio_gate(1, [0.9, 1.1], False)["status"] == "INCONCLUSIVE"
