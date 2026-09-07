"""Counterfactual state, retained-time and diagnostic data-boundary contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import pytorch3d.transforms as transforms

from mixer.continuation_outcomes import (
    exact_tree, branch_horizon, retained_ranges, require_training_source,
    pair_and_classify, _concat_world, metric_deltas)
from mixer.candidate_selection import move_tree
from test_infbagel_hosi import decode_sample_window, prepare_next_window


@pytest.mark.parametrize('start,total,expected,terminal', [
    (0,8,[1,2],False),(5,8,[6,7],True),(6,8,[7],True),(7,8,[],True)])
def test_budget_and_original_terminal(start,total,expected,terminal):
    assert branch_horizon(start,total)==(expected,terminal)


def test_native_retained_ranges_have_no_overlapping_history_or_extra_tail():
    assert retained_ranges(2,8,3)==dict(current=(0,14),next_1=(14,28),next_2=(28,42))
    assert retained_ranges(5,8,3)==dict(current=(0,14),next_1=(14,28),next_2=(28,44))
    assert retained_ranges(7,8,1)==dict(current=(0,16))


def test_stitch_uses_only_real_commits_and_each_history_once():
    windows=[dict(world={'points_world':torch.arange(i*14,i*14+16).reshape(1,16,1)}) for i in range(3)]
    value,ranges=_concat_world(dict(source={'window':0,'total_windows':3},windows=windows))
    assert torch.equal(value['points_world'].flatten(),torch.arange(44))
    assert ranges['next_2']==(28,44)


def test_physical_identity_includes_object_reference_and_typed_conditions():
    value=dict(history=torch.zeros(1,2,232),reference=torch.eye(3),progress=torch.tensor([42]),scene=3)
    assert exact_tree(value,copy.deepcopy(value))
    for key in ('history','reference','progress'):
        other=copy.deepcopy(value);other[key].reshape(-1)[0]+=1
        assert not exact_tree(value,other)
    other=copy.deepcopy(value);other['progress']=other['progress'].float()
    assert not exact_tree(value,other)


def test_restored_tensors_are_independent():
    original=dict(history=torch.ones(2,232),context=dict(object_reference=torch.eye(3)))
    first=move_tree(original,'cpu');second=move_tree(original,'cpu')
    first['history'].zero_();first['context']['object_reference'].fill_(5)
    assert exact_tree(second,original)


def test_test_source_cannot_be_passed_to_future_training_consumer():
    with pytest.raises(ValueError,match='excluded from training'):
        require_training_source(dict(training_allowed=False))
    assert require_training_source(dict(training_allowed=True))['training_allowed']


class IdentityDataset:
    def normalize_torch(self,x,is_object=False):return x
    def denormalize_torch(self,x,is_object=False):return x


def test_shared_native_history_rebases_heading_without_changing_world_or_object_reference():
    from constants import rest_pelvis
    cfg=SimpleNamespace(batch_size=1,max_window_size=16,auto_regre_num=2,device='cpu',seed=42,
                        dataset=SimpleNamespace(nb_joints=28))
    dataset=IdentityDataset()
    clean=torch.zeros(1,16,232)
    xyz=clean[:,:,:84].reshape(1,16,28,3)
    xyz[:,:,:3]=torch.tensor(np.asarray(rest_pelvis)).float()
    xyz[...,0]+=torch.arange(16)[None,:,None]*.02
    clean[:,:,84:216]=transforms.matrix_to_rotation_6d(torch.eye(3)).repeat(1,16,22)
    clean[:,:,216]=torch.arange(16)*.03
    clean[:,:,217]=.8
    clean[:,:,219:228]=torch.eye(3).reshape(1,1,9)
    clean[:,:,228:232]=torch.tensor([.9,.6,0.,1.])
    yaw=transforms.axis_angle_to_matrix(torch.tensor([0.,.7,0.]))
    mat=torch.eye(4)[None];mat[0,:3,:3]=yaw;mat[0,:3,3]=torch.tensor([2.,0.,3.])
    before=decode_sample_window(cfg,clean,dataset,mat)
    saved=copy.deepcopy(before)
    ref=transforms.axis_angle_to_matrix(torch.tensor([.1,.2,.3]))[None]
    prefix=transforms.axis_angle_to_matrix(torch.tensor([0.,-.4,0.]))
    geometry={'box':torch.arange(1200*3).reshape(1200,3).float()*.0001}
    frame,fixed,points=prepare_next_window(cfg,dataset,1,'scene',2,{0:'sub_box_01'},geometry,ref,prefix,
        before['points_orig'],before['obj_trans_orig'],before['object_rot_mat'],before['global_rot_6d'],before['contact_label'])
    replay=clean.clone();replay[:,:2]=fixed
    after=decode_sample_window(cfg,replay,dataset,frame)
    for key in before:
        assert torch.allclose(after[key][:,:2],before[key][:,-2:],atol=2e-6),key
    assert torch.equal(fixed[:,:,219:228],before['object_rot_mat'][:,-2:])
    assert exact_tree(before,saved)
    # The footprint belongs to the first history frame, with the same fixed BPS reference.
    expected=(prefix@ref[0])@geometry['box'].T+before['obj_trans_orig'][0,-2,:,None]
    assert points.shape==(1,1024,3)
    assert torch.cdist(points[0],expected.T).amin(-1).max()<.002


@pytest.fixture
def limits():
    return json.loads(Path('experiments/protocols/p2_continuation_outcomes_s42_20260907.json').read_text())['labels']


def state_metrics():
    hand=dict(hand=0,fixed_active=True,active_surface_mean_m=.02,anchor_vs_W0_m=0.,
              coverage_5cm=.9,predicted_coverage=1.)
    m=dict(status='ok',hands=[hand,dict(hand,hand=1)],support_speed_m_per_s=.1,
           world_joint_speed_m_per_s=.2,human_goal_error_cm=5.,object_goal_error_3D_cm=6.,
           native_surface_HS_s_mean=4.,native_surface_OS_s_mean=20.,human_scene_RMS_cm=1.,
           object_scene_RMS_cm=2.,human_occupied_fraction=.02,object_occupied_fraction=.03)
    row=dict(cached_local_guard=dict(accepted=True),slices={n:copy.deepcopy(m) for n in
             ('current','next_1','next_2','cumulative')},delta_vs_W0={},failure=None,
             termination='budget_censored',terminal_observed=False)
    return dict(contact_reference={'active_hands':[True,True]},branches={a:copy.deepcopy(row) for a in ('W0','Wplus')})


def test_immediate_per_hand_damage_cannot_be_hidden_by_other_hand(limits):
    result=state_metrics()
    for m in result['branches']['Wplus']['slices'].values():m['hands'][0]['coverage_5cm']=.5
    pair_and_classify(result,limits)
    d=result['branches']['Wplus']['diagnostics']
    assert d['immediate_damage'] and not d['delayed_damage']
    assert d['category']=='immediate_damage'
    assert 'hand0:coverage_5cm' in d['failures']['current']


def test_delayed_and_recovery_are_separate_flags(limits):
    result=state_metrics()
    result['branches']['Wplus']['slices']['next_1']['support_speed_m_per_s']=.2
    pair_and_classify(result,limits)
    d=result['branches']['Wplus']['diagnostics']
    assert not d['immediate_damage'] and d['delayed_damage'] and d['recovery']
    assert d['category']=='recovery'


def test_short_horizon_benefit_keeps_censor_and_original_local_guard(limits):
    result=state_metrics();m=result['branches']['Wplus']['slices']['cumulative']
    m['native_surface_HS_s_mean']=3.;m['native_surface_OS_s_mean']=19.
    pair_and_classify(result,limits)
    d=result['branches']['Wplus']['diagnostics']
    assert d['beneficial_compatible'] and d['budget_censored']
    assert d['terminal_inference']=='ambiguous_or_censored'
    assert result['beneficial_compatible_actions']==['Wplus']
    result['branches']['Wplus']['cached_local_guard']['accepted']=False
    pair_and_classify(result,limits)
    assert result['no_useful_alternative']


def test_n_a_contact_and_predicted_release_keep_ambiguity(limits):
    result=state_metrics();result['contact_reference']['active_hands']=[False,False]
    for row in result['branches'].values():
        for m in row['slices'].values():
            for h in m['hands']:h.update(fixed_active=False,active_surface_mean_m=None,anchor_vs_W0_m=None)
    pair_and_classify(result,limits)
    assert result['branches']['Wplus']['diagnostics']['category']=='ambiguous_or_censored'
    result=state_metrics();result['branches']['Wplus']['slices']['next_2']['hands'][0]['predicted_coverage']=.5
    pair_and_classify(result,limits)
    assert result['branches']['Wplus']['diagnostics']['ambiguities']


def test_failed_branch_is_not_a_feasible_alternative(limits):
    result=state_metrics();result['branches']['Wplus']['failure']={'type':'nonfinite'}
    pair_and_classify(result,limits)
    assert not result['branches']['Wplus']['diagnostics']['beneficial_compatible']
    assert result['branches']['Wplus']['diagnostics']['category']=='ambiguous_or_censored'


def test_three_dimensional_object_progress_and_scene_subterms_stay_separate(limits):
    result=state_metrics()
    result['branches']['Wplus']['slices']['current']['object_goal_error_3D_cm']=8.
    result['branches']['Wplus']['slices']['cumulative']['native_surface_HS_s_mean']=3.
    result['branches']['Wplus']['slices']['cumulative']['native_surface_OS_s_mean']=21.
    pair_and_classify(result,limits)
    d=result['branches']['Wplus']['diagnostics']
    assert 'object_goal_error_3D_cm' in d['failures']['current']
    assert not d['scene_benefit']


def test_differences_preserve_n_a_and_skip_status_booleans():
    assert metric_deltas({'a':2.,'b':None,'flag':True},{'a':1.,'b':1.,'flag':False})=={'a':1.}


def test_budget_refuses_unregistered_longer_horizon():
    with pytest.raises(ValueError,match='at most two'):
        branch_horizon(0,9,3)


def test_pair_uses_same_prefix_reference_when_one_branch_is_shorter(limits):
    result=state_metrics();row=result['branches']['Wplus']
    row['paired_reference']=copy.deepcopy(row['slices'])
    # W0's full observed mean differs from its shorter common-prefix mean.
    result['branches']['W0']['slices']['cumulative']['native_surface_HS_s_mean']=2.
    row['slices']['cumulative']['native_surface_HS_s_mean']=3.
    pair_and_classify(result,limits)
    assert row['delta_vs_W0']['cumulative']['native_surface_HS_s_mean']==-1.


@pytest.fixture
def fake_rollout(monkeypatch):
    import hydra
    import test_infbagel_hosi as native
    import mixer.continuation_outcomes as module
    config=SimpleNamespace(device='cpu',sampler=SimpleNamespace(pelvis={}))
    context=dict(mat=torch.eye(4)[None],seq_name_dict={0:'sub_box_0'},
                 obj_rot_mat_ref=torch.eye(3)[None],obj_rot_mat_prefix=torch.eye(3),
                 seq_length=torch.tensor([300]),obj_bps_data=torch.zeros(1,1,1024,3),
                 text_emb=torch.zeros(1,1,768))
    class Sampler:
        def __init__(self):
            self.inner_hoi=SimpleNamespace(sample_calls=0)
            self.scene_editor=SimpleNamespace(motion_records=[],records=[])
            self._window_context=copy.deepcopy(context)
        def set_dataset_and_model(self,dataset,hoi,hsi_model):self.model=hoi
    instances=[]
    def construct(cfg):
        sampler=Sampler();instances.append(sampler);return sampler
    def world(cfg,dataset,clean,ctx):
        output=dict(points_orig=clean.clone(),obj_trans_orig=clean[:,:,216:219].clone(),
                    object_rot_mat=clean[:,:,219:228].clone(),global_rot_6d=clean[:,:,84:216].clone(),
                    contact_label=clean[:,:,228:].clone())
        return dict(points_world=clean.clone()),output
    def prepare(cfg,dataset,step,scene,index,names,rest,ref,prefix,points,*args):
        return torch.eye(4)[None],points[:,-2:].clone(),None
    def step(cfg,index,mat,fixed,sampler,*args):
        generator=torch.Generator().manual_seed(torch.initial_seed()+sampler.inner_hoi.sample_calls*1000003)
        sampler.inner_hoi.sample_calls+=1
        sampler._window_context['private_scene_cache']={'window':index}
        clean=torch.randn(1,16,232,generator=generator)+fixed.mean()
        clean[:,:2]=fixed
        sampler.model(clean)
        sampler.scene_editor.motion_records.append({'edited':clean})
        sampler.scene_editor.records.append({'window':index})
        return world(cfg,None,clean,context)[1]
    monkeypatch.setattr(hydra.utils,'instantiate',construct)
    monkeypatch.setattr(module,'_world_record',world)
    monkeypatch.setattr(native,'prepare_next_window',prepare)
    monkeypatch.setattr(native,'sample_step',step)
    monkeypatch.setattr(native,'get_guidance_from_json',lambda *a:{})
    import astar
    monkeypatch.setattr(astar,'get_path',lambda *a:np.array([[0.,0.],[1.,0.]]))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    dataset=SimpleNamespace(obj_rest_verts={},text=[['carry box']])
    record=dict(rng={'episode_seed':42},window=1,total_windows=6,selected='W0',scene='scene',task=0)
    snapshot=dict(edited=torch.zeros(1,16,232),rest_offsets=torch.zeros(1,16,24,3),replay_context=context)
    return config,dataset,record,dict(snapshot=snapshot,editor={}),instances


def test_branch_order_repeat_rng_counters_and_scene_cache_isolated(fake_rollout,tmp_path):
    from mixer.continuation_outcomes import generate_branch
    from mixer.single_side_waypoint import random_snapshot
    cfg,dataset,record,cached,instances=fake_rollout
    task=dict(data_idx=0,start_location=[0,0,0],pelvis_goal=[1,0,0])
    outputs={}
    before=random_snapshot(torch.device('cpu'))
    for order,arms in enumerate([('Wplus','Wminus'),('Wminus','Wplus')]):
        for arm in arms:
            out=tmp_path/f'{order}-{arm}';out.mkdir()
            cache=copy.deepcopy(cached);cache['snapshot']['edited']+=1 if arm=='Wplus' else 2
            result=generate_branch(cfg,dataset,torch.nn.Identity(),torch.nn.Identity(),record,
                {'test_idx':0},cache,task,arm,out,{'generation':{'max_extra_windows':2}})
            assert result['failure'] is None,result['failure']
            assert result['costs']['HOI_calls']==2
            assert result['costs']['generated_windows']==2
            assert result['termination']=='budget_censored'
            outputs[order,arm]=result['windows'][-1]['clean']
    for arm in ('Wplus','Wminus'):assert torch.equal(outputs[0,arm],outputs[1,arm])
    assert not torch.equal(outputs[0,'Wplus'],outputs[0,'Wminus'])
    assert all(s.inner_hoi.sample_calls==4 for s in instances)
    assert len({id(s._window_context['private_scene_cache']) for s in instances})==4
    after=random_snapshot(torch.device('cpu'))
    assert before[0]==after[0] and np.array_equal(before[1][1],after[1][1]) and torch.equal(before[2],after[2])


def test_failed_branch_preserves_current_and_failure_cost(fake_rollout,tmp_path,monkeypatch):
    from mixer.continuation_outcomes import generate_branch
    import test_infbagel_hosi as native
    cfg,dataset,record,cached,_=fake_rollout
    def fail(*a,**k):raise FloatingPointError('measured nonfinite')
    monkeypatch.setattr(native,'sample_step',fail)
    result=generate_branch(cfg,dataset,torch.nn.Identity(),torch.nn.Identity(),record,{'test_idx':0},cached,
        dict(data_idx=0,start_location=[0,0,0],pelvis_goal=[1,0,0]),'Wplus',tmp_path,
        {'generation':{'max_extra_windows':2}})
    assert result['failure']['type']=='FloatingPointError'
    assert result['termination']=='nonfinite' and not result['terminal_observed']
    assert result['costs']['attempted_windows']==1 and result['costs']['generated_windows']==0
    restored=torch.load(tmp_path/'Wplus.pt',weights_only=False)
    assert len(restored['windows'])==1 and restored['failure']==result['failure']
