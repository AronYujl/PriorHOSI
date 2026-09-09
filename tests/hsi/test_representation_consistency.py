"""Representation mismatch separates body posture, global root, and seam regions."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from priors.hsi.representation_consistency import POSITION_JOINTS_28, position_fk_metrics


def test_common_rigid_transform_preserves_all_distances():
    torch.manual_seed(42)
    direct, fk = torch.randn(12, 28, 3), torch.randn(12, 28, 3)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = torch.tensor([2., 3., 4.])
    before, _ = position_fk_metrics(direct, fk, [2])
    after, _ = position_fk_metrics(direct @ rotation + shift, fk @ rotation + shift, [2])
    for name in before:
        torch.testing.assert_close(torch.tensor(before[name]), torch.tensor(after[name]))


def test_root_translation_and_pose_error_are_separate():
    direct = torch.zeros(12, 28, 3)
    fk = direct + torch.tensor([3., 0., 0.])
    result, _ = position_fk_metrics(direct, fk, [2])
    assert result["root_relative_body_m"] == 0
    assert result["root_error_m"] == result["body_mpjpe_m"] == 3
    fk = direct.clone()
    fk[6:12, 1:22, 0] = 2
    result, _ = position_fk_metrics(direct, fk, [2])
    assert result["root_error_m"] == 0
    assert result["boundary_root_relative_body_m"] == 2
    assert result["interior_root_relative_body_m"] == 0
    assert result["root_relative_body_m"] == 1


def test_direct_position_mapping_and_single_window_region():
    assert POSITION_JOINTS_28 == list(range(22)) + [23, 24, 25, 34, 40, 49]
    direct = torch.zeros(12, 28, 3)
    result, _ = position_fk_metrics(direct, direct, [])
    assert result["boundary_root_relative_body_m"] is None
    assert result["interior_root_relative_body_m"] == 0
