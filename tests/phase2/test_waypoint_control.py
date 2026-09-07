"""Physical waypoint and retained-output contracts, independent of experiment IDs."""
import numpy as np
import pytest
import torch

from mixer.candidate_selection import retained_endpoint_features, saved_endpoint_features, select_candidates
from mixer.waypoint_control import waypoint_variants, local_waypoint, coordinate_difference, validate_model_trace
from utils import interpolate_joints, interp_object, transform_points


def proposals(path=None, waypoint=None, bounds=None, occupancy=None, **kwargs):
    return waypoint_variants(torch.tensor([.5,0.,0.]) if waypoint is None else waypoint,
        [[0.,0.],[.5,0.],[1.,0.]] if path is None else path, torch.tensor([0.,1.,0.]),
        torch.tensor([-2.,-2.,-2.,2.,2.,2.]) if bounds is None else bounds,
        (lambda p:torch.zeros(p.shape[:-1])) if occupancy is None else occupancy, **kwargs)


def test_signed_waypoints_use_path_normal_and_preserve_rng_and_height():
    before=torch.random.get_rng_state().clone()
    result=proposals()
    assert result['applicable'] and result['normal']==[0.,1.]
    assert result['variants']['Wplus']['waypoint_world']==pytest.approx([.5,0.,.1])
    assert result['variants']['Wminus']['waypoint_world']==pytest.approx([.5,0.,-.1])
    assert result['variants']['Wplus']['query'][1]==1.
    assert torch.equal(before,torch.random.get_rng_state())


@pytest.mark.parametrize('arguments,reason',[
    ({'navigation':False},'no_intermediate_navigation'),
    ({'path':[[0.,0.]]},'degenerate_path'),
    ({'waypoint':torch.tensor([1.,0.,0.])},'original_waypoint_is_path_endpoint'),
    ({'path':[[.5,0.],[.5,0.],[1.,0.]]},'zero_path_tangent'),
    ({'bounds':torch.tensor([-2.,-2.,-.05,2.,2.,.05])},'Wplus:outside_scene'),
    ({'occupancy':lambda p:torch.ones(p.shape[:-1])},'Wplus:occupied_offset_at_initial_pelvis_height'),
])
def test_inapplicable_waypoints_keep_explicit_reason(arguments,reason):
    result=proposals(**arguments)
    assert not result['applicable'] and reason in result['reasons']


def test_world_to_local_waypoint_keeps_initial_frame_fixed():
    mat=torch.tensor([[[0.,0.,1.,3.],[0.,1.,0.,0.],[-1.,0.,0.,2.],[0.,0.,0.,1.]]])
    original=mat.clone();world=torch.tensor([3.2,0.,2.5])
    local=local_waypoint(world,mat)
    assert torch.allclose(transform_points(local.reshape(1,1,3),mat).reshape(3),world)
    assert torch.equal(mat,original)
    assert torch.allclose(local,torch.tensor([[-.5,0.,.2]]),atol=1e-6)


def test_native_object_3d_rejects_horizontal_only_pass():
    human=torch.zeros(2,28,3);human[-1,0,1]=1.
    obj=torch.tensor([[0.,0.,0.],[.0642,-.0777,0.]])
    original=human.clone();task=dict(pelvis_goal=[0.,0.,0.],object_goal=[0.,0.,0.])
    result=retained_endpoint_features(human,obj,task,2)
    assert obj[-1,[0,2]].norm()<.1 and result['object_endpoint_m']>.1
    assert result['root_endpoint_m']==0 and not result['completed']
    assert torch.equal(human,original)


@pytest.mark.parametrize('distance,complete',[(.099,True),(.1,False),(.101,False)])
def test_native_completion_strict_boundary(distance,complete):
    result=retained_endpoint_features(torch.zeros(1,28,3),torch.tensor([[distance,0.,0.]]),
        dict(pelvis_goal=[0.,0.,0.],object_goal=[0.,0.,0.]),1)
    assert result['completed']==complete


def test_actual_retained_frame_matches_native_interpolation_and_tail():
    coarse=np.array([[0.,0.,0.],[.09,0.,0.],[.18,0.,0.]],dtype=np.float32)
    rotation=np.tile(np.eye(3).reshape(1,9),(3,1))
    objects,_=interp_object(coarse,rotation,3)
    human=interpolate_joints(torch.tensor(coarse).repeat(1,28),3).reshape(9,28,3)
    task=dict(pelvis_goal=[0.,0.,0.],object_goal=[0.,0.,0.])
    truncated=retained_endpoint_features(human,torch.tensor(objects).float(),task,5)
    assert truncated['endpoint_frame']==4 and truncated['object_endpoint_m']==pytest.approx(.12)
    complete=retained_endpoint_features(human,torch.tensor(objects).float(),task,9)
    assert complete['object_endpoint_m']==pytest.approx(.18)
    assert torch.equal(human[-1],human[-3])
    saved=dict(stitched=dict(object_translation_world=coarse,object_rotation_world=rotation),
               evaluated_joints_world=human[:5],interp_s=3)
    assert saved_endpoint_features(saved,task,'cpu')==truncated
    with pytest.raises(ValueError,match='retained frame'):
        retained_endpoint_features(human,torch.tensor(objects),task,10)


def test_new_endpoint_features_feed_same_selector_protection_without_native_report():
    common=dict(valid=True,energy=.001,root_endpoint_m=.0,contact_distance_m=.01,
                contact_fraction=.9,support_speed_m_per_s=.1)
    features={0:dict(common,object_endpoint_m=.09),1:dict(common,object_endpoint_m=.1008)}
    scores={i:dict(full=-i,static=-i,mismatch=-i) for i in features}
    result=select_candidates(features,scores,dict(budget_relative=.1,budget_absolute=.0067))
    assert result['acceptable']==[0] and result['selected']['CF']==0
    assert result['rejections'][1]==['object_endpoint_m']


def test_coordinate_rms_is_per_coordinate_and_excludes_history():
    first=dict(human=torch.zeros(1,16,24,3),object_translation_world=torch.zeros(1,16,3),
               local_human=torch.zeros(1,16,21,3))
    second={k:v.clone() for k,v in first.items()}
    second['human'][:,2:,:,0]=.03;second['human'][:,:2]=100
    second['object_translation_world'][:,2:,2]=.06
    result=coordinate_difference(first,second)
    assert result['human_rms_mm']==pytest.approx(30/3**.5)
    assert result['object_rms_mm']==pytest.approx(60/3**.5)
    assert result['local_human_rms_mm']==0


def test_model_trace_counts_diffusion_and_existing_B1_reference_calls():
    goal=torch.tensor([[.1,.8,.2]])
    trace=dict(calls=516,goals=torch.tensor([[.1,0.,.2,0.,0.,0.,1.,2.,3.]]))
    record=dict(geometry_edit=dict(hoi_teacher_calls=16))
    assert validate_model_trace(trace,goal,record)==dict(diffusion_calls=500,B1_reference_calls=16)
    trace['calls']=500
    with pytest.raises(AssertionError,match='observed 500, expected 516'):
        validate_model_trace(trace,goal,record)
