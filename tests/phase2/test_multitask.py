"""Source provenance, physical boundary semantics and persistent chain state."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'code'))

from mixer.multitask import (
    object_boundary_measures, source_record, source_type, successor_condition,
    transition_edge, validate_episode,
)


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
