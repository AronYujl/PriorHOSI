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
from mixer.source_eligibility import (
    GRASPED_ENTRY, align_motion, direction_guard, ground_source_motion, object_support,
    select_lingo_pool, source_support, static_first_context, task_object_transform,
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
    task_source = attach_task_reference(record, dict(test_frames=[0, 1, 15]))
    assert task_source['source_terminal_frame'] == 301
    assert task_source['task_reference_frame'] == 165
    assert task_source['task_reference_context_frames'] == list(range(156, 166))


def dataset_standing_fixture(frames=10):
    from types import SimpleNamespace
    joints = torch.zeros(frames, 28, 3)
    joints[:, 0, 1] = .95
    joints[:, 12, 1] = 1.5
    joints[:, [1, 2], 1] = .9
    joints[:, [4, 5], 1] = .5
    joints[:, [7, 8, 10, 11], 1] = .05
    joints[:, [1, 4, 7, 10], 0] = -.1
    joints[:, [2, 5, 8, 11], 0] = .1
    settings = SimpleNamespace(standing_tilt_deg=25., standing_pelvis_height_m=.7,
        standing_knee_flexion_deg=45., standing_speed_m_s=.5, foot_support_m=.08,
        standing_context_frames=10, minimum_frames=49, maximum_frames=600,
        minimum_walk_distance_m=.5, cut_options_per_source=8)
    return joints, settings


def test_dataset_join_stance_accepts_contact_and_rejects_a_crouch():
    from mixer.source_eligibility import standing_context
    joints, settings = dataset_standing_fixture()
    joints[:, [24, 25, 26, 27]] = torch.tensor([0., .5, .1])
    assert standing_context(joints, settings)['passes']
    joints[:, 0, 1] = .6
    assert not standing_context(joints, settings)['passes']


@pytest.mark.parametrize('direction', ['lingo_to_omomo', 'omomo_to_lingo'])
def test_dataset_search_finds_interior_standing_cut_in_long_motion(direction):
    from mixer.source_eligibility import source_cut_options
    joints, settings = dataset_standing_fixture(500)
    joints[:220, :, 0] += torch.arange(220)[:, None]*.03
    joints[220:, :, 0] += 219*.03
    joints[230:, :, 1] -= .4
    if direction == 'omomo_to_lingo':
        joints = joints.flip(0).clone()
    span = dict(source_start_frame=1000, source_stop_frame=1500,
        actions=[dict(start=1000, stop=1500, text='walk', action_type='locomotion')])
    choices = source_cut_options(span, joints, direction, settings)
    assert choices and all(c['internal_cut'] for c in choices)
    if direction == 'lingo_to_omomo':
        assert choices[0]['source_frame_interval'][1] == 1230
    else:
        assert choices[0]['source_frame_interval'][0] == 1270
    assert all(c['walk_distance_m'] >= .5 for c in choices)


def test_dataset_source_spans_keep_seated_activity_and_split_props_and_gaps():
    from mixer.source_eligibility import DATASET_ACTIONS, labelled_spans
    items = [(0,100,'walk'), (100,200,'sit down on chair'), (200,300,'stand up from seat'),
             (300,400,'pick up cup with right hand'), (400,500,'walk'), (510,600,'walk')]
    rows = [dict(start=a, stop=b, text=t, action_type=DATASET_ACTIONS.get(t), scene='010',
        family='010', data_idx=i, sequence=i) for i,(a,b,t) in enumerate(items)]
    spans = labelled_spans(rows)
    assert [(s['source_start_frame'],s['source_stop_frame']) for s in spans] == [(0,300),(400,500),(510,600)]
    assert [a['text'] for a in spans[0]['actions']] == ['walk','sit down on chair','stand up from seat']
    for text in ['drink from cup with right hand','play guitar with both hands','sit down on yoga ball']:
        assert text not in DATASET_ACTIONS


def test_dataset_full_interval_geometry_catches_collision_between_clear_endpoints():
    from types import SimpleNamespace
    from mixer.source_eligibility import full_source_geometry, slice_source
    axis = torch.linspace(-1,1,17)
    x, _, _ = torch.meshgrid(axis,axis,axis,indexing='ij')
    info = dict(centroid=[0.,0.,0.],extents=[2.,2.,2.])
    scene = (x[None,None],info)
    obj_sdf = torch.ones(1,1,17,17,17)
    vertices = torch.tensor([[[.25,.1,0.]], [[-.25,.1,0.]], [[.25,.1,0.]]])
    motion = dict(verts=vertices,joints=vertices)
    threshold = SimpleNamespace(scene_mean_penetration_m=.001,scene_max_penetration_m=.01,
        object_max_penetration_m=.05,floor_max_penetration_m=.01)
    rest = torch.tensor([[.8,.1,0.]])
    args = (scene,obj_sdf,info,rest,torch.zeros(3),torch.eye(3),threshold)
    assert full_source_geometry(slice_source(motion,0,1),*args)['passes']
    assert full_source_geometry(slice_source(motion,2,3),*args)['passes']
    assert not full_source_geometry(motion,*args)['passes']


def test_dataset_stitch_preserves_both_sources_and_moving_object_contexts():
    from mixer.source_bridge import stitch_dataset_motion
    keys = ('pose','translation','joints','object_translation','object_rotation')
    first = {k:torch.arange(20).float()[:,None] for k in keys}
    second = {k:torch.arange(100,130).float()[:,None] for k in keys}
    bridge = {k:torch.arange(200,261).float()[:,None] for k in keys}
    first['betas'] = torch.arange(16).float()
    full = stitch_dataset_motion(first,bridge,second)
    for key in keys:
        assert full[key].shape[0] == 91
        assert torch.equal(full[key][:20],first[key])
        assert torch.equal(full[key][20:61],bridge[key][10:51])
        assert torch.equal(full[key][61:],second[key])
    assert full['betas'] is first['betas']


def test_dataset_inference_payload_contains_only_initial_motion_and_task_goals():
    from mixer.source_bridge import inference_episode
    first = {key:torch.arange(30).float()[:,None] for key in ('pose','translation','object_translation','object_rotation')}
    base = dict(text='walk',pelvis_goal=[1.,0.,2.],object_goal=None,frame_count=30)
    episode = dict(episode_id='example',scene_name='scene',body_identity=dict(gender='male',betas=[0.]*16),
        persistent_objects=[dict(object_id='box',geometry='box.ply')], segments=[
            dict(base,segment_id='lingo',source_dataset='LINGO',source_id='lingo-private',task_type='locomotion',contact_targets=[]),
            dict(base,segment_id='omomo',source_dataset='OMOMO',source_id='omomo-private',task_type='hoi',
                object_goal=[1.,.5,2.],object_rotation_goal=torch.eye(3).tolist())])
    task = inference_episode(episode,first)
    assert task['frame_count'] == 101
    assert task['segments'][1]['start_frame'] == 71
    assert len(task['initial_context']['pose']) == 10
    assert task['initial_context']['pose'][-1] == [9.]
    assert task['segments'][1]['object_goal'] == [1.,.5,2.]
    import json
    encoded = json.dumps(task)
    for field in ('source_id','source_frame_interval','witness','construction','omomo-private','lingo-private'):
        assert field not in encoded


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


def test_expanded_pool_covers_families_and_respects_full_clip_duration():
    rows = [dict(source_id=f'lingo-{i}', data_idx=i, source_scene_family=family,
        task_type='locomotion', text='walk', source_duration_s=duration)
        for i,family,duration in [(0,'a',2.), (1,'a',3.), (2,'a',4.), (3,'b',5.), (4,'c',11.), (5,'c',6.)]]
    selected = select_lingo_pool(rows, 3, balance_families=True, max_duration_s=10.)
    assert [r['source_id'] for r in selected] == ['lingo-0', 'lingo-3', 'lingo-5']


def test_pair_search_alternates_action_types_and_prefers_unused_sources():
    from collections import Counter
    from mixer.source_eligibility import ordered_pair_pool
    rows = [dict(source_id=f'lingo-{i}', data_idx=i, source_scene_family=family, task_type=kind)
        for i,family,kind in [(0,'a','locomotion'), (1,'b','locomotion'),
                              (2,'c','static_object_interaction'), (3,'d','static_object_interaction')]]
    ordered = ordered_pair_pool(rows, Counter({'lingo-0':7}), Counter({'a':7}))
    assert [r['source_id'] for r in ordered] == ['lingo-1','lingo-2','lingo-0','lingo-3']


def test_pilot_selection_uses_source_coverage_and_keeps_failed_model_scores_irrelevant():
    from mixer.source_eligibility import select_execution_pilot
    sources = {f'lingo-{i}':dict(source_id=f'lingo-{i}', source_scene_family=str(i)) for i in range(3)}
    episodes = [dict(episode_id=str(i), direction=GRASPED_ENTRY, scene_name=scene,
        persistent_objects=[dict(object_id='box')], segments=[dict(source_id=source)],
        model_score=score) for i,scene,source,score in
        [(0,'a','lingo-0',.9),(1,'a','lingo-1',.8),(2,'b','lingo-0',.7),(3,'b','lingo-2',0.)]]
    assert select_execution_pilot(episodes, sources, 2) == ['0','3']
    for row in episodes:
        row['model_score'] = 1-row['model_score']
    assert select_execution_pilot(episodes, sources, 2) == ['0','3']


def test_actual_history_budget_keeps_partial_window_and_every_existing_frame():
    from mixer.multitask_execution import TRACK_KEYS, append_bridge, append_window, window_plan
    assert window_plan(10, 109) == [42, 42, 15]
    assert window_plan(10, 52) == [42]
    history = {key:torch.arange(10).float()[:,None] for key in TRACK_KEYS}
    history['betas'] = torch.zeros(16)
    initial = {key:value.clone() for key,value in history.items()}
    for count in window_plan(10, 109):
        window = {key:torch.arange(len(history['pose'])-4, len(history['pose'])+42).float()[:,None] for key in TRACK_KEYS}
        history = append_window(history, window, count)
    for key in TRACK_KEYS:
        torch.testing.assert_close(history[key][:10], initial[key])
        assert history[key][:,0].tolist() == list(range(109))
    bridge = {key:torch.arange(99,160).float()[:,None] for key in TRACK_KEYS}
    combined = append_bridge(history, bridge)
    for key in TRACK_KEYS:
        assert combined[key][:,0].tolist() == list(range(160))
    assert combined['betas'] is history['betas']


@pytest.mark.parametrize('change,failed', [
    (dict(goal_error=.11),'goal'), (dict(support=False),'body_support'),
    (dict(geometry=False),'geometry'), (dict(history_error=.001),'native_history'),
    (dict(budget_ok=False),'frame_budget'), (dict(object_fixed=False),'persistent_object')])
def test_actual_predecessor_failure_blocks_a_source_feasible_successor(change, failed):
    from mixer.multitask_execution import blocked, stage_checks
    values = dict(finite=True, goal_error=.03, geometry=True, support=True,
        history_error=1e-6, budget_ok=True, object_fixed=True)
    values.update(change)
    checks = stage_checks(**values)
    reasons = [name for name, passed in checks.items() if not passed]
    assert reasons == [failed]
    record = blocked('kimodo','hsi',reasons)
    assert record['status'] == 'blocked_by_predecessor'
    assert record['metrics'] is None and record['generated_frames'] == 0


def test_actual_contact_trajectory_keeps_native_coarse_history_samples():
    from mixer.multitask_execution import contact_track
    coarse = torch.arange(64).reshape(16,4).float()
    native = contact_track(coarse)
    assert native.shape == (46,4)
    torch.testing.assert_close(native[::3], coarse)


def test_first_window_reuse_requires_the_actual_input_history_and_progress(tmp_path):
    from mixer.multitask_execution import TRACK_KEYS, cached_hsi_window
    history = {key:torch.arange(10).float()[:,None] for key in TRACK_KEYS}
    path = tmp_path/'first_window.pt'
    torch.save(dict(actual_input=history, world={'points':torch.ones(16,28,3)}, clean=torch.zeros(1,16,232),
        audit=dict(progress=[6,54,96],generation_seconds=60.,hsi_forward_calls=1000)),path)
    _, audit, _ = cached_hsi_window(path,history,(6,54,96))
    assert audit['generation_seconds'] == 0 and audit['cached_generation_seconds'] == 60
    assert audit['hsi_forward_calls'] == 0
    with pytest.raises(ValueError,match='different progress'):
        cached_hsi_window(path,history,(48,96,96))
    history['object_translation'] = history['object_translation']+.01
    with pytest.raises(ValueError,match='different actual history'):
        cached_hsi_window(path,history,(6,54,96))


def test_native_history_preserves_moving_object_pose_contact_and_physical_goal(monkeypatch):
    from types import SimpleNamespace
    import utils
    from datasets.infbagel import InfBaGelDataset
    from mixer.body_projection import native_rest_offsets
    from mixer.kinematic_composition import _PARENTS_22
    from mixer.standing_transition import make_history, native_local_goal
    from mixer.surface_edit import decode_body
    from test_infbagel_hosi import decode_sample_window
    from pytorch3d import transforms
    monkeypatch.setattr(utils, 'SMPL_DIR', str(Path(__file__).resolve().parents[2]/'smpl_models'))
    device = torch.device('cuda:1')
    model = utils.create_smplx_model('male', device).eval().requires_grad_(False)
    dataset = InfBaGelDataset.__new__(InfBaGelDataset)
    dataset.min_torch = torch.tensor([-5.,-5.,-5.], device=device)
    dataset.max_torch = -dataset.min_torch
    dataset.obj_min_torch, dataset.obj_max_torch = dataset.min_torch, dataset.max_torch
    dataset.ori_sequence_idx = [0]
    dataset.parents_22 = np.array(_PARENTS_22)
    dataset.transl = np.array([[.03, -.07, .02]], dtype=np.float32)
    pose = torch.zeros(10,22,3,device=device)
    pose[:,0,1] = torch.linspace(.2,.4,10,device=device)
    motion = dict(pose=pose, translation=torch.tensor([1.,1.,2.],device=device).repeat(10,1),
        betas=torch.zeros(16,device=device), gender='male',
        object_translation=torch.arange(30,device=device).reshape(10,3).float()/50,
        object_rotation=transforms.axis_angle_to_matrix(pose[:,0]),
        contact=torch.arange(40,device=device).reshape(10,4).float()/40)
    _, motion['joints'] = decode_body(motion, model)
    cfg = SimpleNamespace(device=str(device), dataset=SimpleNamespace(nb_joints=28), batch_size=1, max_window_size=16)
    task = dict(data_idx=0)
    clean, mat, reference, error = make_history(cfg,dataset,motion,task,model,
        object_reference=motion['object_rotation'][-4], preserve_object_history=True, contact_history=motion['contact'])
    assert error < 1e-5
    decoded = decode_sample_window(cfg,clean,dataset,mat)
    torch.testing.assert_close(decoded['obj_trans_orig'][0,:2], motion['object_translation'][[-4,-1]], atol=1e-6, rtol=1e-6)
    rotations = decoded['object_rot_mat'].reshape(16,3,3) @ reference[0]
    torch.testing.assert_close(rotations[:2], motion['object_rotation'][[-4,-1]], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(clean[0,:2,228:232], motion['contact'][[-4,-1]])
    goal = motion['joints'][-1,0].tolist()
    local = native_local_goal(dataset,motion,task,model,mat,goal,planar=False)
    physical = local @ mat[0,:3,:3].T+mat[:,:3,3]+native_rest_offsets(model,motion['betas'])[0]+torch.as_tensor(dataset.transl[0],device=device)
    torch.testing.assert_close(physical[0], motion['joints'][-1,0], atol=1e-6, rtol=1e-6)

    from mixer.multitask_execution import sample_hoi_window
    dataset.scene_name = ['subject_box_sequence']
    dataset.obj_rest_verts = {'box':torch.tensor([[0.,0.,0.],[.1,.2,.3]],device=device)}
    cfg.seed = 42
    task.update(object_name='box', pelvis_goal=goal, object_goal=[.8,.3,.6])
    captured = {}
    class Sampler:
        def p_sample_loop(self, fixed, *args, **kwargs):
            captured['fixed'] = fixed.clone()
            return [clean.clone()], []
    class Codec:
        def recompute_bps(self, vertices, reference):
            captured['reference'] = reference.clone()
            return vertices.new_zeros(1,1024,3)
    world, audit, _, snapshot = sample_hoi_window(cfg,Sampler(),dataset,motion,task,model,
        torch.zeros(1,768,device=device),Codec(),0,3)
    torch.testing.assert_close(captured['fixed'][0,:2,228:], motion['contact'][[-4,-1]])
    torch.testing.assert_close(captured['reference'][0], motion['object_rotation'][-4])
    torch.testing.assert_close(world['object_rotation_world'][:2].to(device),motion['object_rotation'][[-4,-1]],atol=1e-6,rtol=1e-6)
    assert audit['object_history_max_error_m'] < 1e-6
    assert audit['progress'] == [0,48,132]
    torch.testing.assert_close(snapshot['conditioned_history'].to(device),captured['fixed'])


@pytest.mark.parametrize('contact_frames,accepted', [
    ([0, 1, 2, 3, 4], True), ([1, 2, 3, 4, 5], False),
    ([0], False), ([0, 2, 4, 6, 8], True),
])
def test_grasped_entry_requires_initial_and_sustained_source_contact(contact_frames, accepted):
    joints = torch.ones(10, 28, 3)
    joints[contact_frames, 24] = torch.tensor([.01, 0., 0.])
    boundary = object_boundary_measures(joints, torch.zeros(10, 3),
        torch.eye(3).repeat(10, 1, 1), torch.zeros(1, 3))
    assert all(direction_guard(GRASPED_ENTRY, boundary, .08).values()) == accepted
    assert not all(direction_guard('lingo_to_omomo', boundary, .08).values())


def test_contact_audit_distinguishes_eye_proximity_from_each_hand():
    from smplx.joint_names import JOINT_NAMES
    from utils import SMPLX_JOINTS_28
    names = [JOINT_NAMES[index] for index in SMPLX_JOINTS_28]
    joints = torch.ones(10, 28, 3)
    eyes = [names.index(name) for name in ('left_eye_smplhf', 'right_eye_smplhf')]
    joints[:, eyes] = 0.
    translation, rotation, vertices = torch.zeros(10, 3), torch.eye(3).repeat(10, 1, 1), torch.zeros(1, 3)
    eyes_only = object_boundary_measures(joints, translation, rotation, vertices)
    assert not eyes_only['first_frame_hand_contact']
    assert eyes_only['hands_released']
    joints[:, names.index('left_middle1')] = 0.
    left_only = object_boundary_measures(joints, translation, rotation, vertices)
    assert left_only['first_frame_hand_distances_m'][0] == 0.
    assert left_only['first_frame_hand_distances_m'][1] > 1.
    assert left_only['hand_contact_frame_fraction'] == 1.


def test_source_grounding_and_horizontal_alignment_preserve_standup_height_and_velocity():
    joints = torch.zeros(14, 28, 3)
    joints[..., 1] = torch.linspace(.55, .98, 14)[:, None]
    joints[:, 1, 0], joints[:, 2, 0] = .1, -.1
    vertices = joints[:, :8].clone()
    vertices[:, 0, 1] = -.03
    motion = dict(joints=joints, verts=vertices, pose=torch.zeros(14, 22, 3),
        translation=joints[:, 0].clone())
    grounded = ground_source_motion(motion)
    aligned, placement = align_motion(grounded, -1, torch.tensor([2., 1.41, 3.]), torch.tensor(.7))
    assert placement['translation_m'][1] == 0
    torch.testing.assert_close(aligned['joints'][..., 1], joints[..., 1]+.03)
    torch.testing.assert_close(aligned['joints'][-1, 0, [0, 2]], torch.tensor([2., 3.]))
    torch.testing.assert_close(aligned['joints'].diff(dim=0)[..., 1], joints.diff(dim=0)[..., 1])
    torch.testing.assert_close(motion['verts'][:, 0, 1], torch.full((14,), -.03))


@pytest.mark.parametrize('goal', [[3., 0., 1.], [1., 0., 3.], [-1., 0., 1.], [1., 0., -1.]])
def test_omomo_entry_placement_matches_native_hosi_heading_and_object_transform(goal):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    source_rotation = Rotation.from_euler('zxy', [.1, .2, .7])
    pose = torch.zeros(10, 22, 3)
    pose[:, 0] = torch.tensor(source_rotation.as_rotvec(), dtype=torch.float32)
    roots = torch.tensor([2., .9, -3.]).repeat(10, 1)
    roots[:, 0] += torch.arange(10)*.01
    joints = roots[:, None].repeat(1, 28, 1)
    motion = dict(pose=pose, translation=roots-torch.tensor([.02, -.1, .03]),
        joints=joints, verts=joints.clone())
    corpus = SimpleNamespace(object_translation=(roots+torch.tensor([.4, -.5, .2])).numpy(),
        object_rotation=np.repeat(np.eye(3)[None], 10, axis=0))
    task = dict(start_location=[1., 0., 1.], pelvis_goal=goal)
    moved, position, rotation = task_object_transform(corpus,
        dict(initial_context_frames=list(range(10))), task, GRASPED_ENTRY, motion)
    delta = np.array(goal)-np.array(task['start_location'])
    target_yaw = np.arctan2(-delta[2], delta[0])+np.pi/2
    expected = Rotation.from_euler('y', target_yaw).as_matrix() @ Rotation.from_euler('zxy', [0., 0., -.7]).as_matrix()
    expected = torch.tensor(expected, dtype=torch.float32)
    target_root = torch.tensor([1., .9, 1.])
    shift = target_root-expected @ roots[0]
    torch.testing.assert_close(moved['joints'], joints @ expected.T+shift, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(position, torch.from_numpy(corpus.object_translation[0]) @ expected.T+shift)
    torch.testing.assert_close(rotation, expected)
    torch.testing.assert_close(moved['joints'][..., 1], joints[..., 1])


def test_source_support_checks_every_seated_context_and_every_foot_frame():
    from types import SimpleNamespace
    axis = torch.linspace(-1, 1, 33)
    _, y, _ = torch.meshgrid(axis, axis, axis, indexing='ij')
    scene = (y[None, None]-.5, dict(centroid=[0., 0., 0.], extents=[2., 2., 2.]))
    joints = torch.zeros(14, 28, 3)
    joints[:, 0, 1] = .68
    joints[:, [7, 8, 10, 11], 1] = .03
    joints[10:, 0, 1] = 1.
    vertices = torch.zeros(14, 8, 3)
    vertices[:, :4, 1] = .5
    vertices[10:, :4, 1] = .82
    vertices[:, [0, 1], 0], vertices[:, [2, 3], 0] = -.1, .1
    weights = torch.zeros(8, 22)
    weights[:2, 1], weights[2:4, 2], weights[4:, 7] = 1., 1., 1.
    motion = dict(joints=joints, verts=vertices, pose=torch.zeros(14, 22, 3))
    record = dict(task_type='static_object_interaction', text='stand up from seat')
    model = SimpleNamespace(lbs_weights=weights)
    assert source_support(motion, record, model, scene)['passes']
    vertices[4, :4, 1] += .1
    assert not source_support(motion, record, model, scene)['seat_supported']
    vertices[4, :4, 1] -= .1
    joints[12, [7, 8, 10, 11], 1] = .09
    assert not source_support(motion, record, model, scene)['feet_supported']


def test_persistent_object_needs_floor_or_upward_facing_scene_support():
    axis = torch.linspace(-1, 1, 33)
    x, y, _ = torch.meshgrid(axis, axis, axis, indexing='ij')
    info = dict(centroid=[0., 0., 0.], extents=[2., 2., 2.])
    vertices = torch.tensor([[-.01, .5, 0.], [.01, .5, 0.]])
    assert object_support(vertices, (y[None, None]-.5, info))['supported']
    assert not object_support(vertices, (x[None, None], info))['supported']
    assert not object_support(vertices, (torch.ones_like(y)[None, None], info))['supported']
    vertices[:, 1] = 0
    assert object_support(vertices, (torch.ones_like(y)[None, None], info))['supported']


def test_static_grasp_target_preserves_the_first_pose_and_original_moving_context():
    source = dict(pose=torch.arange(10*22*3).reshape(10, 22, 3).float(),
        object_translation=torch.arange(30).reshape(10, 3).float(),
        object_rotation=torch.eye(3).repeat(10, 1, 1), betas=torch.arange(16).float())
    target = static_first_context(source)
    for key in ('pose', 'object_translation', 'object_rotation'):
        assert torch.equal(target[key], source[key][:1].expand_as(source[key]))
    assert target['betas'] is source['betas']
    assert source['object_translation'][1, 0] == 3


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
