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


def test_contact_intervals_keep_boundaries_and_remove_only_short_runs():
    from mixer.inbetween_contact import contact_intervals
    contacts = torch.zeros(12, 2, dtype=torch.bool)
    contacts[:3, 0] = True
    contacts[6:8, 0] = True
    contacts[8:, 1] = True
    kept, intervals = contact_intervals(contacts, 3)
    assert kept[:3, 0].all() and kept[8:, 1].all()
    assert not kept[6:8, 0].any()
    assert [(r['start'], r['stop'], r['retained']) for r in intervals] == [(0, 3, True), (6, 8, False), (8, 12, True)]
    assert contacts[6:8, 0].all()


def test_contact_correction_changes_only_free_frames_and_carries_objects():
    from mixer.inbetween_contact import fixed_context_motion
    source = dict(pose=torch.randn(61, 22, 3), translation=torch.randn(61, 3), object_translation=torch.randn(61, 3))
    free = torch.randn(41, 22, 3, requires_grad=True)
    root = torch.randn(41, 3, requires_grad=True)
    result = fixed_context_motion(source, free, root)
    for key in ('pose', 'translation'):
        assert torch.equal(result[key][:10], source[key][:10])
        assert torch.equal(result[key][51:], source[key][51:])
    assert result['object_translation'] is source['object_translation']
    (result['pose'].sum()+result['translation'].sum()).backward()
    torch.testing.assert_close(free.grad, torch.ones_like(free))
    torch.testing.assert_close(root.grad, torch.ones_like(root))


def test_fixed_contact_speed_still_counts_a_sliding_foot_after_it_is_raised():
    from mixer.inbetween_contact import contact_measures
    vertices = torch.zeros(61, 4, 3)
    vertices[:, :, 0] = torch.arange(61)[:, None]/30
    patches = torch.arange(4)[:, None]
    mask = torch.ones(61, 4, dtype=torch.bool)
    grounded = contact_measures(vertices, patches, mask)
    vertices[:, :, 1] = .2
    raised = contact_measures(vertices, patches, mask)
    assert abs(grounded['fixed_contact_speed_m_s']-1.) < 1e-5
    assert grounded['fixed_contact_speed_m_s'] == raised['fixed_contact_speed_m_s']
    assert raised['fixed_contact_height_abs_m'] > .19
