"""Metric identities and paper-cohort semantics for native artifact readout."""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from priors.hsi.text_motion import GEOMETRY_KEYS, frechet_samples, geometry_groups


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
