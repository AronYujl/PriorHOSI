import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mixer.single_side_waypoint import (independent_variants, geometry_energy,
    window_choices, random_snapshot, restore_random, SingleSideWaypointSampler, fork_sampler)
from mixer.candidate_selection import CandidatePoolSampler
from mixer.waypoint_control import waypoint_variants


def inputs(occupancy):
    return (torch.tensor([.5,0.,0.]), [[0.,0.],[.5,0.],[1.,0.]],torch.tensor([0.,1.,0.]),
            torch.tensor([-2.,-2.,-2.,2.,2.,2.]),occupancy)


def test_one_invalid_side_keeps_other_and_legacy_is_unchanged():
    args=inputs(lambda p:p[...,2]>0)
    assert not waypoint_variants(*args)['applicable']
    result=independent_variants(*args)
    assert result['valid_sides']==['Wminus']
    assert result['variants']['Wminus']['waypoint_world']==pytest.approx([.5,0.,-.1])


def test_endpoint_has_no_offset():
    args=list(inputs(lambda p:p[...,2]>0));args[0]=torch.tensor([1.,0.,0.])
    assert independent_variants(*args)['valid_sides']==[]


def test_fixed_energy_and_budget_ties_and_independent_acceptance():
    metrics={a:dict(human_scene_RMS_cm=h,object_scene_RMS_cm=o) for a,h,o in
             [('W0',1.,2.),('Wplus',1.,1.),('Wminus',0.,0.)]}
    assert geometry_energy(metrics['W0'])==pytest.approx(5/75)
    checks={'Wplus':{'accepted':True},'Wminus':{'accepted':False}}
    config=dict(budget_relative=.1,budget_absolute=1/150)
    d=window_choices(metrics,checks,config)
    assert d['accepted']==['W0','Wplus'] and d['G']=='Wplus'
    assert d['budget_set']==['Wplus'] and d['H']=='Wplus'
    metrics['Wplus']=metrics['W0'].copy()
    scores={'W0':{'full':1.},'Wplus':{'full':1.}}
    assert window_choices(metrics,checks,config,scores)['H']=='W0'
    scores['Wplus']['full']=.5
    assert window_choices(metrics,checks,config,scores)['H']=='Wplus'


def test_no_qualified_side_returns_W0_even_with_nonzero_scene_energy():
    metrics={a:dict(human_scene_RMS_cm=10.,object_scene_RMS_cm=10.) for a in ('W0','Wplus')}
    d=window_choices(metrics,{'Wplus':{'accepted':False}},dict(budget_relative=.1,budget_absolute=.0067))
    assert d['G']==d['H']=='W0' and d['energies']['W0']>0


def test_random_replay_restores_all_cpu_sources():
    device=torch.device('cpu');snapshot=random_snapshot(device)
    one=(random.random(),np.random.rand(),torch.randn(4))
    restore_random(snapshot,device)
    two=(random.random(),np.random.rand(),torch.randn(4))
    assert one[:2]==two[:2] and torch.equal(one[2],two[2])


def test_disabled_policy_calls_existing_sampler_directly(monkeypatch):
    sampler=SingleSideWaypointSampler.__new__(SingleSideWaypointSampler)
    sampler.waypoint_policy='off';marker=object()
    monkeypatch.setattr(CandidatePoolSampler,'p_sample_loop',lambda self,*a,**k:marker)
    assert sampler.p_sample_loop()==marker


def test_fork_isolates_mutable_counters_caches_and_recorders():
    inner=SimpleNamespace(audit={'n':0},guidance_audit={'n':0},guidance_rest_vertex_cache={},
                          local_bps_digest=SimpleNamespace(copy=lambda:object()),sample_calls=0)
    parent=SingleSideWaypointSampler.__new__(SingleSideWaypointSampler)
    parent.hoi_adapter=SimpleNamespace(inner=inner)
    parent.hsi_sampler=SimpleNamespace(batch_size=1)
    parent.scene_editor=SimpleNamespace(records=[{'old':True}],motion_records=[{'old':True}])
    branch=fork_sampler(parent)
    branch.inner_hoi.sample_calls+=1;branch.inner_hoi.audit['n']+=1
    branch.inner_hoi.guidance_audit['n']+=1;branch.inner_hoi.guidance_rest_vertex_cache['a']=1
    branch.scene_editor.records.append({'new':True});branch.hsi_sampler.batch_size=2
    assert inner.sample_calls==0 and inner.audit=={'n':0} and inner.guidance_audit=={'n':0}
    assert inner.guidance_rest_vertex_cache=={} and parent.hsi_sampler.batch_size==1
    assert parent.scene_editor.records==[{'old':True}]
