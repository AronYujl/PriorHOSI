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


def test_small_angle_native_interpolation_keeps_body_and_object_keyframes():
    import numpy as np
    from utils import interp_jrot, interp_object, quaternion_slerp
    angles = torch.tensor([[0., .4, 0.], [0., .4005, 0.], [0., .401, 0.]])
    quaternions = transforms.axis_angle_to_quaternion(angles)
    first = quaternion_slerp(quaternions[0], quaternions[1], 0.)
    last = quaternion_slerp(quaternions[0], quaternions[1], 1.)
    torch.testing.assert_close(first, quaternions[0], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(last, quaternions[1], atol=1e-7, rtol=1e-7)
    early = transforms.quaternion_to_axis_angle(quaternion_slerp(quaternions[0],quaternions[1],.25))[1]
    late = transforms.quaternion_to_axis_angle(quaternion_slerp(quaternions[0],quaternions[1],.75))[1]
    assert angles[0,1] < early < late < angles[1,1]
    body = interp_jrot(quaternions[:,None].expand(-1,22,-1),3)
    expected = transforms.quaternion_to_matrix(quaternions)
    torch.testing.assert_close(transforms.quaternion_to_matrix(body[::3,0]), expected, atol=1e-7, rtol=1e-7)
    position = np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]])
    translated, rotated = interp_object(position, expected.numpy().reshape(3,9),3)
    np.testing.assert_allclose(translated[::3],position,atol=1e-7)
    np.testing.assert_allclose(rotated[::3].reshape(3,3,3),expected.numpy(),atol=1e-7)


def test_bridge_prediction_batch_reader_leaves_scalar_metadata_out(tmp_path):
    import numpy as np
    from mixer.source_bridge import PREDICTION_FIELDS, prediction_row
    data = dict(local_rot_mats=np.zeros((2,61,22,3,3),dtype=np.float32),
        root_positions=np.zeros((2,61,3),dtype=np.float32), target_joints=np.zeros((2,61,22,3),dtype=np.float32),
        foot_contacts=np.zeros((2,61,4),dtype=np.bool_), fps=np.array(30), scale=np.ones(2))
    data['root_positions'][1] = 7.
    path = tmp_path/'prediction.npz'
    np.savez(path,**data)
    with np.load(path) as arrays:
        sample = prediction_row(arrays,1,'cpu')
    assert set(sample) == set(PREDICTION_FIELDS)
    assert sample['root_positions'].shape == (61,3)
    assert bool((sample['root_positions'] == 7.).all())


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


def test_native_surface_gradient_matches_rigid_translation(monkeypatch):
    import utils
    from mixer.inbetween_contact import differentiable_body
    monkeypatch.setattr(utils, 'SMPL_DIR', str(Path(__file__).resolve().parents[2]/'smpl_models'))
    device = torch.device('cuda:6')
    model = utils.create_smplx_model('male', device).eval().requires_grad_(False)
    pose = torch.zeros(1, 22, 3, device=device, requires_grad=True)
    translation = torch.zeros(1, 3, device=device, requires_grad=True)
    motion = dict(pose=pose, translation=translation, betas=torch.zeros(16, device=device), gender='male')
    vertices, joints = differentiable_body(motion, model)
    gradient = torch.autograd.grad(vertices[..., 1].mean(), (translation, pose), retain_graph=True)
    torch.testing.assert_close(gradient[0], torch.tensor([[0., 1., 0.]], device=device))
    assert torch.isfinite(gradient[1]).all() and gradient[1].abs().max() > 1e-4
    joint_gradient = torch.autograd.grad(joints[..., 2].mean(), translation)[0]
    torch.testing.assert_close(joint_gradient, torch.tensor([[0., 0., 1.]], device=device))


def test_acquisition_bridge_roundtrip_preserves_both_native_contexts(monkeypatch):
    import utils
    from mixer.source_bridge import bridge_condition
    from mixer.surface_edit import decode_body
    monkeypatch.setattr(utils, 'SMPL_DIR', str(Path(__file__).resolve().parents[2]/'smpl_models'))
    device = torch.device('cuda:1')
    model = utils.create_smplx_model('female', device).eval().requires_grad_(False)
    pose = torch.zeros(17, 22, 3, device=device)
    pose[:, 0, 1] = .6
    pose[:, 18, 2] = .2
    source = dict(pose=pose, translation=torch.tensor([2., .1, -3.], device=device).repeat(17, 1),
        betas=torch.linspace(-.2, .2, 16, device=device), gender='female')
    source['translation'][:, 0] += torch.arange(17, device=device)*.01
    source['verts'], source['joints'] = decode_body(source, model)
    target = dict(source, pose=pose[:10].clone(), translation=source['translation'][-1:].repeat(10, 1))
    target['pose'][:, 0, 1] = .9
    target['pose'][:, 18, 2] = .7
    condition, canonical, origin, arrays = bridge_condition(source, target,
        torch.tensor([2.5, .5, -3.], device=device), torch.eye(3, device=device), model)
    for key in ('pose', 'translation'):
        assert torch.equal(condition[key][:10], source[key][-10:])
        assert torch.equal(condition[key][51:], target[key])
    assert arrays['known'].sum() == 20
    assert not arrays['known'][10:51].any()
    rebuilt = native_fk(arrays['local_rot_mats'][None], arrays['neutral_joints'][None],
        arrays['root_positions'][None])[0] @ canonical+origin
    torch.testing.assert_close(rebuilt, condition['joints'][:, :22], atol=1e-5, rtol=0)
    torch.testing.assert_close(condition['object_translation'],
        torch.tensor([2.5, .5, -3.], device=device).repeat(61, 1))


def test_acquisition_requires_each_source_contacting_hand_at_the_suffix():
    from mixer.source_bridge import acquisition_measures
    condition = dict(joints=torch.ones(61, 28, 3), object_translation=torch.zeros(61, 3),
        object_rotation=torch.eye(3).repeat(61, 1, 1))
    condition['joints'][51:, [24, 26]] = torch.tensor([.01, 0., 0.])
    motion = dict(condition, joints=condition['joints'].clone())
    motion['joints'][51:, 26] = 1.
    partial = acquisition_measures(motion, condition, torch.zeros(1, 3))
    assert partial['source_contacting_hands'] == [True, True]
    assert not partial['suffix_contact_recovered']
    motion['joints'][51:, 26] = condition['joints'][51:, 26]
    complete = acquisition_measures(motion, condition, torch.zeros(1, 3))
    assert complete['suffix_contact_recovered']
    assert not complete['contact_at_bridge_start']
    assert complete['first_matching_contact_frame'] == 51


def test_bridge_rotation_seam_uses_short_arc_at_both_boundaries():
    from mixer.source_bridge import rotation_seam_measures
    pose = torch.zeros(61, 22, 3)
    pose[:10, 0, 1] = 179*torch.pi/180
    pose[10:51, 0, 1] = -179*torch.pi/180
    pose[51:, 0, 1] = 178*torch.pi/180
    result = rotation_seam_measures(dict(pose=pose))
    assert abs(result['entry_root_rotation_jump_deg']-2.) < 1e-4
    assert abs(result['exit_rotation_jump_max_deg']-3.) < 1e-4


def test_loaded_bridge_object_sdf_supports_native_metrics_correction_and_source_geometry(tmp_path):
    import json
    import numpy as np
    from mixer.source_bridge import load_bridge_object_sdf
    from mixer.inbetween_contact import object_distances
    from mixer.standing_transition import object_measures
    from mixer.multitask_geometry import geometry_measures
    folder = tmp_path/'data/object/rest_object_sdf_256_npy_files'
    folder.mkdir(parents=True)
    axis = np.linspace(-1, 1, 9, dtype=np.float32)
    x, _, _ = np.meshgrid(axis, axis, axis, indexing='ij')
    np.save(folder/'box.npy', x)
    info = dict(centroid=[0., 0., 0.], extents=[2., 2., 2.])
    (folder/'box.json').write_text(json.dumps(info))
    sdf, info = load_bridge_object_sdf(tmp_path, 'box', 'cpu')
    vertices = torch.tensor([[[-.02, .1, 0.]]]).repeat(61, 1, 1).requires_grad_(True)
    motion = dict(verts=vertices, object_translation=torch.zeros(61, 3),
        object_rotation=torch.eye(3).repeat(61, 1, 1))
    signed = object_distances(vertices[10:51], motion, sdf, info)
    torch.testing.assert_close(signed, torch.full((41, 1), -.02), atol=1e-7, rtol=0)
    gradient, = torch.autograd.grad(signed.sum(), vertices)
    torch.testing.assert_close(gradient[10:51, 0], torch.tensor([1., 0., 0.]).repeat(41, 1))
    native = object_measures(motion, sdf, info)
    source = geometry_measures(vertices, torch.ones_like(sdf[:, None]), info, sdf[:, None], info,
        motion['object_translation'][:, None], motion['object_rotation'])
    assert abs(native['human_object_penetration_max_m']-.02) < 1e-7
    assert abs(source['object_penetration_max_m']-.02) < 1e-7
