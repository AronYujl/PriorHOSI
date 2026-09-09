"""Coordinate and temporal contracts at the external motion boundary."""

import sys
from pathlib import Path

import torch
from pytorch3d import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))

from mixer.inbetween import interpolate_rotations, heading
from mixer.inbetween_external import native_fk, retarget_positions, bone_lengths


def test_rotation_interpolation_crosses_heading_wrap_on_short_arc():
    first = transforms.axis_angle_to_matrix(torch.tensor([[0., 179*torch.pi/180, 0.]]))
    last = transforms.axis_angle_to_matrix(torch.tensor([[0., -179*torch.pi/180, 0.]]))
    output = interpolate_rotations(first, last, torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(output[0], first)
    torch.testing.assert_close(output[-1], last, atol=1e-6, rtol=1e-6)
    forward = output[1, 0] @ torch.tensor([0., 0., 1.])
    assert forward[2] < -.999


def test_shape_retarget_keeps_root_and_recovers_original_bone_directions():
    torch.manual_seed(42)
    neutral = torch.randn(2, 22, 3)
    rotation = transforms.axis_angle_to_matrix(torch.randn(2, 7, 22, 3)*.15)
    roots = torch.randn(2, 7, 3)
    original = native_fk(rotation, neutral, roots)
    lengths = bone_lengths(neutral)
    other_lengths = lengths*torch.linspace(.7, 1.3, 22)
    changed = retarget_positions(original, other_lengths[:, None], roots)
    recovered = retarget_positions(changed, lengths[:, None], roots)
    torch.testing.assert_close(recovered, original, atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(changed[:, :, 0], roots)


def test_heading_canonicalization_preserves_world_up():
    joints = torch.zeros(22, 3)
    joints[2, 0], joints[17, 0] = -1., -2.
    joints[1, 0], joints[16, 0] = 1., 2.
    rot = transforms.axis_angle_to_matrix(torch.tensor([0., .8, 0.]))
    moved = (rot @ joints.T).T
    torch.testing.assert_close(heading(moved), torch.tensor(.8))
