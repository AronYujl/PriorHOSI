"""Representation mismatch separates body posture, global root, and seam regions."""

import sys
from pathlib import Path

import torch
import pytest
import pytorch3d.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from priors.hsi.representation_consistency import POSITION_JOINTS_28, position_fk_metrics
from utils import interpolate_joints, interp_jrot, quaternion_slerp


@pytest.mark.parametrize("angle", [0.001, 1.2])
@pytest.mark.parametrize("sign", [1., -1.])
def test_quaternion_interpolation_advances_from_first_to_second_rotation(angle, sign):
    start = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    end = transforms.axis_angle_to_quaternion(
        torch.tensor([angle, 0., 0.], dtype=torch.float64)
    ) * sign
    for fraction in (0., .25, .5, 1.):
        actual = quaternion_slerp(start, end, fraction)
        expected = transforms.axis_angle_to_quaternion(
            torch.tensor([angle * fraction, 0., 0.], dtype=torch.float64)
        )
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def test_identical_and_antipodal_quaternions_hold_the_rotation():
    pose = transforms.axis_angle_to_quaternion(torch.tensor([.2, -.4, .1]))
    for sign in (1., -1.):
        for fraction in (0., .25, .5, 1.):
            torch.testing.assert_close(quaternion_slerp(pose, sign * pose, fraction), pose)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_positions_and_rotations_share_the_native_clock_and_final_hold(device):
    position = torch.tensor([0., 2., 6.], dtype=torch.float64, device=device)
    joints = torch.zeros(3, 28, 3, dtype=torch.float64, device=device)
    joints[:, :, 0] = position[:, None]
    axis_angle = torch.zeros(3, 22, 3, dtype=torch.float64, device=device)
    axis_angle[:, :, 0] = position[:, None] * .1
    rotation = transforms.axis_angle_to_quaternion(axis_angle)
    expected_position = torch.tensor(
        [0., 2./3, 4./3, 2., 10./3, 14./3, 6., 6., 6.],
        dtype=torch.float64, device=device,
    )
    actual_joints = interpolate_joints(joints, 3).reshape(9, 28, 3)
    actual_rotation = interp_jrot(rotation, 3)
    expected_axis = torch.zeros(9, 22, 3, dtype=torch.float64, device=device)
    expected_axis[:, :, 0] = expected_position[:, None] * .1
    torch.testing.assert_close(actual_joints[:, :, 0], expected_position[:, None].expand(-1, 28))
    torch.testing.assert_close(actual_rotation, transforms.axis_angle_to_quaternion(expected_axis))
    torch.testing.assert_close(actual_joints[::3], joints)
    torch.testing.assert_close(actual_rotation[::3], rotation)
    torch.testing.assert_close(actual_rotation[-3:], rotation[-1:].expand(3, -1, -1), atol=0, rtol=0)
    assert actual_joints.device == actual_rotation.device == joints.device
    assert actual_joints.dtype == actual_rotation.dtype == joints.dtype


def test_scale_one_is_exact_identity_and_single_frame_holds():
    joints = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    rotation = transforms.axis_angle_to_quaternion(
        torch.tensor([[[.001, .3, -.2]]], dtype=torch.float64)
    )
    assert interpolate_joints(joints, 1) is joints
    assert interp_jrot(rotation, 1) is rotation
    torch.testing.assert_close(interpolate_joints(joints, 3), joints.expand(3, -1), atol=0, rtol=0)
    torch.testing.assert_close(interp_jrot(rotation, 3), rotation.expand(3, -1, -1), atol=0, rtol=0)


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
