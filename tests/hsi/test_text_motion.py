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


def _cohort_fixture(tmp_path):
    import json
    from types import SimpleNamespace
    from unittest import mock

    ids = ["scene:0", "scene:1", "scene:2", "scene:3"]
    captions = ["walk", "sit", "sit", "lie"]
    vectors = np.asarray([[0., 0.], [1., 2.], [2., 1.], [3., 3.]], dtype=np.float32)
    truth_dir = tmp_path / "truth"
    directories = {truth_dir: [SimpleNamespace(sequence_id=name, caption=caption,
                                              text=vector, motion=vector)
                              for name, caption, vector in zip(ids, captions, vectors)]}
    inputs = {}
    for label, shift in (("control", 0.), ("enhanced", 1.), ("permuted", 2.)):
        source = tmp_path / (label + ".json")
        shard = tmp_path / label / "evaluation" / "per_sequence_metrics.json"
        directories[shard.parent.parent / "motion"] = [
            SimpleNamespace(sequence_id=name, caption=caption, text=vector,
                            motion=vector + np.asarray([shift, 0.], dtype=np.float32))
            for name, caption, vector in zip(ids, captions, vectors)
        ]
        # Shard file order is deliberately opposite to native sequence order.
        directories[shard.parent.parent / "motion"].reverse()
        source.write_text(json.dumps({
            "metrics": {name: {"pen_value": 0.1, "goal_orientation_err_rad": None,
                               "per_window": {"nested": 1.}} for name in reversed(ids)},
            "merged_from": [str(shard)], "sample_type": "ddim", "guided": False,
            "seed": 42, "timing": {"sampler_steps_per_window": 25},
        }))
        inputs[label] = str(source)
    legacy = SimpleNamespace(
        load_directory=mock.Mock(side_effect=lambda directory, _: directories[directory]),
        embed=mock.Mock(side_effect=lambda items, *_: (
            np.stack([item.text for item in items]), np.stack([item.motion for item in items]))),
    )
    cfg = SimpleNamespace(
        cohort_output=str(tmp_path / "readout"), device="cpu", cohort_inputs=inputs,
        cohort_truth_dir=str(truth_dir), table3_bootstrap_batch=256,
        table3_encoder_checkpoint="frozen-encoder", table3_mean="mean", table3_std="std",
    )
    return cfg, legacy, directories


def test_cohort_readout_keeps_real_labels_pairs_fid_and_exports_scalar_metrics(tmp_path):
    import json
    from unittest import mock
    from priors.hsi.text_motion import cohort_text_motion_readout

    cfg, legacy, _ = _cohort_fixture(tmp_path)
    with mock.patch("priors.hsi.text_motion._legacy_encoder", return_value=(legacy, None, {}, ())):
        path = cohort_text_motion_readout(cfg)
    summary = json.loads(path.read_text())
    assert list(summary["arms"]) == ["control", "enhanced", "permuted"]
    assert all(not record["guided"] for record in summary["protocol"]["generation"].values())
    assert summary["groups"]["all"]["sequence_count"] == 4
    assert summary["groups"]["interactive"]["sequence_count"] == 3
    assert summary["groups"]["all"]["retrieval_gallery32"]["status"] == "unavailable"
    assert summary["groups"]["interactive"]["retrieval_gallery32"]["distinct_caption_count"] == 2
    for label, expected in (("control", 0.), ("enhanced", 1.), ("permuted", 4.)):
        np.testing.assert_allclose(summary["arms"][label]["groups"]["all"]["FID"]["mean"],
                                   expected, atol=1e-12)
    paired = next(row for row in summary["fid_pairwise"]
                  if row["a"] == "control" and row["b"] == "enhanced" and row["group"] == "all")
    np.testing.assert_allclose(paired["ci95"], [1., 1.], atol=1e-12)
    record = json.loads((path.parent / "enhanced" / "per_sequence_metrics.json").read_text())
    assert not record["guided"]
    assert all(row["MM-Dist"] == 1. for row in record["metrics"].values())
    assert all(row["pen_value"] == 0.1 for row in record["metrics"].values())
    assert all(row["goal_orientation_err_rad"] is None for row in record["metrics"].values())
    assert all("per_window" not in row for row in record["metrics"].values())
    with np.load(path.parent / "enhanced" / "embeddings.npz") as arrays:
        assert arrays["fid_bootstrap_all"].shape == (2000,)
        assert arrays["sequence_ids"].tolist() == summary["sequence_ids"]
    assert legacy.embed.call_count == 4  # One shared GT and three prediction cohorts.


def test_cohort_readout_requires_native_caption_alignment(tmp_path):
    import pytest
    from unittest import mock
    from priors.hsi.text_motion import cohort_text_motion_readout

    cfg, legacy, directories = _cohort_fixture(tmp_path)
    motion_dir = tmp_path / "control" / "motion"
    directories[motion_dir][0].caption = "wrong task"
    with mock.patch("priors.hsi.text_motion._legacy_encoder", return_value=(legacy, None, {}, ())):
        with pytest.raises(AssertionError):
            cohort_text_motion_readout(cfg)
