"""Contracts of complete-candidate sampling, scoring and selection."""
from types import SimpleNamespace

import pytest
import torch

from mixer.candidate_selection import (
    CandidatePoolSampler, aggregate_scores, mismatch_dynamic, prediction_frames,
    reconstruction_error, score_window, select_candidates,
)
from mixer.composed_sampler import HOSIComposedSampler


def test_scoring_prediction_frames_cover_global_future_once():
    all_frames=[]
    for i in range(3):
        valid,frames=prediction_frames(i,16)
        assert valid.sum()==14
        all_frames+=frames.tolist()
    assert all_frames==list(range(2,44))
    valid,frames=prediction_frames(2,16,torch.arange(16)<9)
    assert frames.tolist()==list(range(30,37))
    with pytest.raises(ValueError,match='zero valid'):
        prediction_frames(0,16,torch.zeros(16,dtype=torch.bool))


def test_frame_weighted_aggregation_and_duplicate_rejection():
    windows=[dict(full=1.,static=2.,mismatch=3.,coordinates=84,global_frames=[2]),
             dict(full=3.,static=4.,mismatch=5.,coordinates=252,global_frames=[3,4,5])]
    assert aggregate_scores(windows)==dict(full=2.5,static=3.5,mismatch=4.5)
    with pytest.raises(ValueError,match='repeated'):
        aggregate_scores([windows[0],windows[0]])
    with pytest.raises(ValueError,match='zero'):
        aggregate_scores([])


def test_position_score_ignores_history_object_and_padding():
    x=torch.zeros(1,16,232);pred=x.clone()
    pred[:,:2]=100;pred[:,2:9,:84]=2;pred[:,9:]=float('nan');pred[:,2:9,216:]=float('nan')
    valid,_=prediction_frames(0,16,torch.arange(16)<9)
    assert reconstruction_error(pred,x,valid)==4
    pred[:,4,13]=float('inf')
    with pytest.raises(FloatingPointError):reconstruction_error(pred,x,valid)


def common_args():
    common=[torch.zeros(1) for _ in range(17)]
    common[13]=torch.ones(1,dtype=torch.bool)
    common[15]=torch.arange(4*32**3).reshape(4,32,32,32).float()
    return tuple(common)


def test_mismatch_preserves_all_static_conditions_and_object_view_values():
    common=common_args();wrong=mismatch_dynamic(common)
    assert all(a is b for i,(a,b) in enumerate(zip(common,wrong)) if i!=15)
    assert torch.equal(wrong[15][:1],common[15][:1])
    assert torch.equal(wrong[15][1:],torch.roll(common[15][1:],16,2))
    assert torch.equal(torch.sort(wrong[15].flatten()).values,torch.sort(common[15].flatten()).values)


class FakeScorer:
    def __init__(self):
        self.calls=[];self.queries=[]
        self.hsi_sampler=SimpleNamespace(q_sample=lambda x,t,n:x+.2*n,student_model=self.model)
    def model(self,view,*common,is_sample=True,is_uncondition=False):
        self.calls.append((view.clone(),common,is_uncondition))
        assert not common[13].any()
        return view + (1 if is_uncondition else 2)
    def _hsi_predict_pair(self,view,common):
        return self.model(view,*common),self.model(view,*common,is_uncondition=True)
    def _hsi_model_arguments(self,current,previous,t,context):
        self.queries.append((current.clone(),previous.clone()))
        # Geometry preparation may use random subsampling, owned by a private stream.
        torch.rand(3)
        return common_args()


def test_score_pairs_clean_geometry_private_noise_and_raw_heads():
    sampler=FakeScorer();clean=torch.arange(16*232).float().reshape(1,16,232)*.001
    before=clean.clone();state=torch.random.get_rng_state().clone()
    options=dict(levels=[50,157],repetitions=2)
    first=score_window(sampler,clean,{},371,0,options)
    assert torch.equal(state,torch.random.get_rng_state())
    assert torch.equal(before,clean)
    assert all(torch.equal(q,clean) for q in sampler.queries[0])
    for i in range(0,12,3):
        full,static,wrong=sampler.calls[i:i+3]
        assert torch.equal(full[0],static[0]) and torch.equal(full[0],wrong[0])
        assert all(a is b for a,b in zip(full[1],static[1]))
        assert not full[2] and static[2]
        assert torch.equal(full[0][:,:2,:216],clean[:,:2,:216])
        assert not full[0][:,:2,216:].any()
    score_window(sampler,clean+1,{},17,1,options)
    assert first==score_window(sampler,clean,{},371,0,options)
    assert first['hsi_calls']==12 and first['coordinates']==14*84


@pytest.mark.parametrize('offset',[0,100000000,200000000,300000000])
def test_candidate_seed_is_scoped_and_zero_keeps_original(monkeypatch,offset):
    def original(self,*args,**kwargs):
        g=torch.Generator().manual_seed(torch.initial_seed()+1000003)
        return torch.rand(3,generator=g),torch.initial_seed()
    monkeypatch.setattr(HOSIComposedSampler,'p_sample_loop',original)
    sampler=object.__new__(CandidatePoolSampler)
    sampler.candidate_seed_offset=offset;sampler.record_replay_context=False
    torch.manual_seed(413);state=torch.random.get_rng_state().clone()
    result,seed=sampler.p_sample_loop(torch.zeros(1))
    assert seed==413+offset
    assert torch.equal(state,torch.random.get_rng_state()) and torch.initial_seed()==413
    expected=torch.rand(3,generator=torch.Generator().manual_seed(413+offset+1000003))
    assert torch.equal(result,expected)


def feature(**kwargs):
    return dict(dict(valid=True,energy=.01,root_endpoint_m=.02,object_endpoint_m=.04,
        contact_distance_m=.03,contact_fraction=.8,support_speed_m_per_s=.1),**kwargs)


POLICY=dict(budget_relative=.1,budget_absolute=.006666666666666667)


def test_whole_candidate_budget_selection_ties_and_order():
    features={0:feature(),1:feature(energy=.005),2:feature(energy=.03)}
    scores={0:dict(full=0.,static=1.,mismatch=2.),1:dict(full=2.,static=0.,mismatch=1.),
            2:dict(full=-1.,static=-1.,mismatch=-1.)}
    expected=select_candidates(features,scores,POLICY)
    assert expected['selected']==dict(C0=0,CG=1,CS=1,CF=0,CM=1)
    assert expected['budget_set']==[0,1]
    assert select_candidates(dict(reversed(list(features.items()))),scores,POLICY)==expected
    scores[1]['full']=0.
    assert select_candidates(features,scores,POLICY)['selected']['CF']==0


@pytest.mark.parametrize('change,reason',[
    ({'valid':False},'invalid_motion_or_editor_guard'),
    ({'root_endpoint_m':.10},'root_endpoint_m'),
    ({'object_endpoint_m':.11},'object_endpoint_m'),
    ({'contact_distance_m':.041},'contact_distance'),
    ({'contact_fraction':.74},'contact_fraction'),
    ({'support_speed_m_per_s':.111},'support_speed'),
])
def test_common_protections_exclude_candidate_before_hsi(change,reason):
    f={0:feature(),1:feature(**change)};s={i:dict(full=float(-i),static=float(-i),mismatch=float(-i)) for i in f}
    result=select_candidates(f,s,POLICY)
    assert result['acceptable']==[0] and reason in result['rejections'][1]
    assert set(result['selected'].values())=={0}


def test_no_acceptable_fallback_and_nonfinite_score_failure():
    f={0:feature(valid=False),1:feature(valid=False)}
    result=select_candidates(f,{},POLICY)
    assert result['fallback']=='no_acceptable_candidate' and set(result['selected'].values())=={0}
    with pytest.raises(FloatingPointError):
        select_candidates({0:feature()},{0:dict(full=float('nan'),static=0,mismatch=0)},POLICY)
