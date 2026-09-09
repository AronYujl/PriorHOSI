"""Source provenance, physical boundary semantics and persistent chain state."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'code'))

from mixer.multitask import (
    attach_task_reference, object_boundary_measures, source_record, source_type, successor_condition,
    transition_edge, validate_episode,
)
from mixer.multitask_handoff import native_budget_frames, motion_slice, source_handoff_checks
from mixer.source_eligibility import align_motion, direction_guard, select_lingo_pool


def test_no_hand_annotation_does_not_make_a_held_prop_action_static():
    assert source_type('drink from cup with right hand') is None
    assert source_type('brush teeth with toothbrush in right hand') is None
    assert source_type('sit down on office chair') == 'static_object_interaction'
    assert source_type('walk') == 'locomotion'


def test_source_terminal_refers_to_sequence_end_not_first_window_end():
    language = dict(ori_sequence_idx=[0], start_idx=[120], end_idx=[168],
                    text=[['walk']], end_range=[302])
    record = source_record('LINGO', 0, language, np.array([100]), np.array([302]), '010', 'test')
    assert record['initial_context_frames'] == list(range(120, 130))
    assert record['terminal_context_frames'] == list(range(292, 302))
    assert record['source_terminal_frame'] == 301
    assert record['source_sequence_start'] == 100
    assert record['source_start_frame'] == 120
    assert not record['source_is_complete_recomposed_gt']
    task_source = attach_task_reference(record, dict(test_frames=[0, 1, 15]))
    assert task_source['source_terminal_frame'] == 301
    assert task_source['task_reference_frame'] == 165
    assert task_source['task_reference_context_frames'] == list(range(156, 166))


def test_suspended_stationary_object_requires_placement_before_release():
    joints = torch.zeros(10, 28, 3)
    rotation = torch.eye(3).repeat(10, 1, 1)
    translation = torch.tensor([[0., .5, 0.]]).repeat(10, 1)
    vertices = torch.tensor([[-.1, -.1, 0.], [.1, .1, 0.]])
    result = object_boundary_measures(joints, translation, rotation, vertices)
    assert result['hands_released']
    assert not result['supported_slow']
    assert result['required_exit_action'] == 'place_then_release'


def test_grounded_object_keeps_release_semantics_when_hand_still_touches():
    joints = torch.zeros(10, 28, 3)
    translation = torch.zeros(10, 3)
    rotation = torch.eye(3).repeat(10, 1, 1)
    vertices = torch.tensor([[0., 0., 0.], [.1, .1, 0.]])
    result = object_boundary_measures(joints, translation, rotation, vertices)
    assert result['supported_slow']
    assert not result['hands_released']
    assert result['required_exit_action'] == 'release'


def test_successor_uses_achieved_body_and_object_state():
    segment = dict(data_idx=7, start_location=[100., 0., 100.], initialization='source')
    history = dict(joints=torch.ones(10, 28, 3)*2)
    objects = {'box': dict(position=torch.tensor([3., .4, 2.]))}
    condition = successor_condition(segment, history, objects)
    assert condition['initial_history'] is history
    assert condition['persistent_objects'] is objects
    assert condition['data_idx'] == 7
    torch.testing.assert_close(condition['start_location'], torch.tensor([2., 0., 2.]))
    assert segment['start_location'] == [100., 0., 100.]


def test_native_budget_includes_initial_history_and_excludes_held_padding():
    # Eight 16-frame windows share two coarse samples at every boundary.
    coarse_frames = 16 + 7*14
    observed = (coarse_frames-1)*3+1
    assert native_budget_frames(11.2) == observed == 340
    assert native_budget_frames(9.8) == 298
    assert observed-native_budget_frames(9.8) == 42
    assert native_budget_frames(193/30) == 197


def test_cutoff_context_preserves_body_identity_and_actual_object_track():
    motion = dict(pose=torch.zeros(340, 22, 3), joints=torch.ones(340, 28, 3),
        object_translation=torch.arange(340).float()[:, None].expand(-1, 3),
        betas=torch.arange(16).float(), gender='male')
    prefix = motion_slice(motion, 0, native_budget_frames(9.8))
    context = motion_slice(prefix, -10, None)
    assert context['object_translation'][:, 0].tolist() == list(range(288, 298))
    assert context['betas'] is motion['betas']
    assert context['gender'] == 'male'
    assert len(motion['pose']) == 340


def handoff_fixture():
    from types import SimpleNamespace
    thresholds = SimpleNamespace(object_floor_m=.05, object_speed_m_s=.1,
        object_angular_speed_rad_s=.5, tilt_deg=25, root_height_m=.65, foot_height_m=.08)
    scene = dict(scene_penetration_mean_m=0., scene_penetration_max_m=0., scene_outside_fraction=0.)
    values = dict(pelvis_goal_error_m=.03, object_goal_error_m=.03, object_floor_max_m=.02,
        object_speed_max_m_s=.05, object_angular_speed_max_rad_s=.3, tilt_max_deg=3.,
        root_height_min_m=.9, foot_height_max_m=.04, object_scene_geometry=scene,
        body_geometry=dict(scene, object_penetration_max_m=0., floor_penetration_max_m=0.))
    return values, thresholds


@pytest.mark.parametrize('metric,value,failed', [
    ('object_speed_max_m_s', .6, 'object_translation_slow'),
    ('object_angular_speed_max_rad_s', .6, 'object_rotation_slow'),
    ('object_floor_max_m', .2, 'object_supported'),
    ('foot_height_max_m', .09, 'body_foot_support'),
    ('pelvis_goal_error_m', .11, 'pelvis_at_goal'),
])
def test_goal_arrival_alone_does_not_authorize_a_handoff(metric, value, failed):
    values, thresholds = handoff_fixture()
    assert all(source_handoff_checks(values, thresholds, .1).values())
    values[metric] = value
    checks = source_handoff_checks(values, thresholds, .1)
    assert [key for key, passed in checks.items() if not passed] == [failed]


def test_legacy_padding_cannot_supply_zero_terminal_object_velocity():
    from mixer.standing_transition import observed_motion
    moving = torch.arange(14).float()[:, None].expand(-1, 3)*.02
    raw = dict(pose=torch.zeros(16, 22, 3), object_translation=torch.cat((moving, moving[-1:].expand(2, -1))),
        betas=torch.arange(16).float())
    observed = observed_motion(raw)
    assert len(observed['pose']) == 14
    assert (raw['object_translation'][-1]-raw['object_translation'][-2]).norm() == 0
    assert (observed['object_translation'][-1]-observed['object_translation'][-2]).norm()*30 > .1
    assert observed['betas'] is raw['betas']


def test_source_direction_guards_keep_omomo_terminal_and_initial_rules_separate():
    terminal = dict(torso_tilt_max_deg=4., pelvis_height_min_m=.9,
        foot_floor_distance_max_m=.03, supported_slow=True, hands_released=True,
        hand_object_min_m=.3)
    assert all(direction_guard('omomo_to_lingo', terminal, .08).values())
    terminal['hands_released'] = False
    assert not all(direction_guard('omomo_to_lingo', terminal, .08).values())
    initial = dict(hand_object_min_m=.081)
    assert all(direction_guard('lingo_to_omomo', initial, .08).values())
    initial['hand_object_min_m'] = .079
    assert not all(direction_guard('lingo_to_omomo', initial, .08).values())


def test_source_alignment_targets_the_omomo_boundary_without_changing_source_length():
    motion = dict(
        joints=torch.zeros(6, 28, 3),
        verts=torch.zeros(6, 12, 3),
        pose=torch.zeros(6, 22, 3),
        translation=torch.zeros(6, 3),
    )
    motion['joints'][:, 0, 0] = torch.arange(6).float()
    aligned, record = align_motion(motion, 0, torch.tensor([2., 0., 3.]), torch.tensor(0.))
    torch.testing.assert_close(aligned['joints'][0, 0], torch.tensor([2., 0., 3.]))
    assert len(aligned['joints']) == 6
    assert record['target_root'] == [2., 0., 3.]


def test_source_pool_is_balanced_by_action_type_and_ordered_by_data_idx():
    records = [
        dict(source_id='walk-9', data_idx=9, task_type='locomotion'),
        dict(source_id='walk-2', data_idx=2, task_type='locomotion'),
        dict(source_id='sit-8', data_idx=8, task_type='static_object_interaction'),
        dict(source_id='sit-1', data_idx=1, task_type='static_object_interaction'),
        dict(source_id='sit-0', data_idx=0, task_type='static_object_interaction'),
    ]
    pool = select_lingo_pool(records, 1)
    assert [(row['task_type'], row['data_idx']) for row in pool] == [
        ('static_object_interaction', 0), ('locomotion', 2)
    ]


def episode_fixture():
    sources = {'lingo-3': dict(source_dataset='LINGO', data_idx=3),
               'omomo-3': dict(source_dataset='OMOMO', data_idx=3)}
    segments = [dict(segment_id='walk', task_type='locomotion', source_id='lingo-3',
        source_dataset='LINGO', data_idx=3, scene_name='scene', initialization='source'),
        dict(segment_id='carry', task_type='hoi', source_id='omomo-3',
        source_dataset='OMOMO', data_idx=3, scene_name='scene', initialization='previous_generated_history')]
    return dict(scene_name='scene', segments=segments,
        transitions=[transition_edge('walk', 'carry', 'approach_and_grasp', 'persist')]), sources


def test_corpus_qualified_indices_can_have_the_same_numeric_data_idx():
    episode, sources = episode_fixture()
    assert validate_episode(episode, sources)
    episode['segments'][1]['source_dataset'] = 'LINGO'
    with pytest.raises(ValueError, match='resolve together'):
        validate_episode(episode, sources)


@pytest.mark.parametrize('change,message', [
    ('reset', 'inherit generated history'), ('scene', 'persistent target scene'),
    ('edge', 'one transition'), ('order', 'transition order'),
    ('exclude_transition', 'includes transitions'), ('no_contact_target', 'contact target'),
])
def test_continuous_episode_contract_rejects_broken_composition(change, message):
    episode, sources = episode_fixture()
    if change == 'reset':
        episode['segments'][1]['initialization'] = 'source'
    elif change == 'scene':
        episode['segments'][1]['scene_name'] = 'another_scene'
    elif change == 'edge':
        episode['transitions'] = []
    elif change == 'order':
        episode['transitions'][0]['source_segment'] = 'carry'
    elif change == 'exclude_transition':
        episode['transitions'][0]['evaluate_transition_frames'] = False
    else:
        episode['segments'][0].update(task_type='static_object_interaction', contact_target=None)
    with pytest.raises(ValueError, match=message):
        validate_episode(episode, sources)


def test_sdf_query_preserves_xyz_axes_and_metric_units():
    from mixer.multitask_geometry import signed_query
    axis = torch.linspace(-1, 1, 9)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    sdf = (x+2*y+3*z)[None, None]
    info = dict(centroid=[3., 4., 5.], extents=[4., 2., 3.])
    points = torch.tensor([[3.2, 4.4, 5.6], [3., 4., 5.], [6., 4., 5.]])
    signed, outside = signed_query(points, sdf, info)
    torch.testing.assert_close(signed[:2], torch.tensor([2.8, 0.]), atol=2e-6, rtol=0)
    assert outside.tolist() == [False, False, True]


def test_path_clearance_includes_the_persisted_object():
    from mixer.multitask_geometry import clear_straight_paths
    axis = torch.linspace(-1, 1, 33)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    sphere = (x*x+y*y+z*z).sqrt()-.2
    obj_info = dict(centroid=[0., 0., 0.], extents=[2., 2., 2.])
    scene = torch.ones(1, 1, 33, 33, 33)
    scene_info = dict(centroid=[0., 1., 0.], extents=[8., 8., 8.])
    start = torch.tensor([-1., 0., 0.])
    end = torch.tensor([[1., 0., 0.]])
    clear, length = clear_straight_paths(start, end, scene, scene_info,
        sphere[None, None], obj_info, torch.tensor([0., .75, 0.]), torch.eye(3))
    assert not clear.item()
    torch.testing.assert_close(length, torch.tensor([2.]))
    clear, _ = clear_straight_paths(start, end, scene, scene_info,
        sphere[None, None], obj_info, torch.tensor([0., .75, 2.]), torch.eye(3))
    assert clear.item()


def test_native_translation_rotation_uses_the_rest_pelvis_pivot():
    from mixer.multitask_geometry import transformed_motion
    from mixer.surface_edit import yaw_matrix
    pelvis_offset = torch.tensor([.1, -.2, .3])
    translation = torch.tensor([[1., 2., 3.]])
    joints = torch.zeros(1, 28, 3)+translation[:, None]+pelvis_offset
    motion = dict(translation=translation, joints=joints, verts=joints.clone(), pose=torch.zeros(1, 22, 3))
    rotation = yaw_matrix(torch.tensor([torch.pi/2]))[0]
    shift = torch.tensor([4., 0., 5.])
    moved = transformed_motion(motion, rotation, shift)
    torch.testing.assert_close(moved['translation']+pelvis_offset, moved['joints'][:, 0])
    torch.testing.assert_close(moved['joints'], joints @ rotation.T+shift)


def test_terminal_witness_reconstructs_both_given_goals_independently_of_entry_path():
    from mixer.multitask_geometry import terminal_goal_rotation
    from mixer.surface_edit import yaw_matrix
    source_pelvis = torch.tensor([2., .9, -3.])
    source_object = torch.tensor([2.5, .4, -2.7])
    desired_rotation = yaw_matrix(torch.tensor([1.1]))[0]
    target_pelvis = torch.tensor([-1., .9, 2.])
    target_object = desired_rotation @ (source_object-source_pelvis)+target_pelvis
    task = dict(start_location=[2., 0., -2.], pelvis_goal=[-1., 0., 2.], object_goal=target_object.tolist())
    rotation = terminal_goal_rotation(source_pelvis, source_object, task)
    torch.testing.assert_close(rotation, desired_rotation)
    shift = target_pelvis-rotation @ source_pelvis
    torch.testing.assert_close(rotation @ source_object+shift, target_object)


def test_endpoint_grounding_preserves_each_context_velocity_and_grounds_both():
    from mixer.multitask_geometry import ground_endpoint_contexts
    vertices = torch.zeros(20, 4, 3)
    vertices[:10, :, 1] = -.02
    vertices[10:, :, 1] = -.12
    vertices[:, :, 0] = torch.arange(20)[:, None]/30
    motion = dict(verts=vertices, joints=vertices.clone(), translation=vertices[:, 0].clone(),
                  pose=torch.randn(20, 22, 3))
    grounded = ground_endpoint_contexts(motion)
    torch.testing.assert_close(grounded['verts'][..., 1], torch.zeros(20, 4))
    for lo, hi in [(0, 10), (10, 20)]:
        torch.testing.assert_close(grounded['joints'][lo+1:hi]-grounded['joints'][lo:hi-1],
                                   motion['joints'][lo+1:hi]-motion['joints'][lo:hi-1])
    assert grounded['pose'] is motion['pose']
    torch.testing.assert_close(motion['verts'][:10, :, 1], torch.full((10, 4), -.02))


def test_seating_support_distinguishes_a_seat_from_a_vertical_wall():
    from mixer.multitask_geometry import seating_support_mask
    axis = torch.linspace(-1, 1, 33)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    info = dict(centroid=[0., 0., 0.], extents=[2., 2., 2.])
    patches = torch.tensor([[[-.02, -.01, -.1], [-.02, -.01, .1]]])
    assert seating_support_mask(patches, y[None, None], info).item()
    assert not seating_support_mask(patches, x[None, None], info).item()
