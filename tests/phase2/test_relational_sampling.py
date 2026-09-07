"""Executed relationship state, native geometry, and posterior intervention contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch3d import transforms

from mixer.relational_sampling import (RelationMemory, RelationalGuidance, relation_energy, relation_geometry)
from mixer.waypoint_control import decode_motion


@pytest.fixture
def settings():
    return json.loads(Path('experiments/protocols/p2_relational_sampling_s42_20260907.json').read_text())['relation']


def observed(distance=.01, contact=1., drift=0., location=0.):
    relative=torch.full((1,2,2,3),location)
    relative[:,1,:,0]+=drift
    return dict(relative=relative,distance=torch.full((1,2,2),distance),contact=torch.full((1,2,2),contact))


@pytest.mark.parametrize('distance,contact,drift',[(.06,1.,0.),(.01,.9,0.),(.01,1.,.03)])
def test_trust_requires_geometry_contact_and_temporal_consistency(settings,distance,contact,drift):
    m=RelationMemory('C2',settings);m.observe(observed(distance,contact,drift))
    assert not m.active.any()


def test_local_refresh_and_persistent_anchor_share_engagement_rules(settings):
    local=RelationMemory('C1',settings);persistent=RelationMemory('C2',settings)
    for m in (local,persistent):m.observe(observed());m.observe(observed(location=.01))
    assert torch.equal(local.active,persistent.active)
    assert torch.equal(local.anchor,torch.full((1,2,3),.01))
    assert torch.equal(persistent.anchor,torch.zeros(1,2,3))


def test_contact_prediction_alone_does_not_release(settings):
    m=RelationMemory('C2',settings);m.observe(observed());a=m.anchor.clone()
    event=m.observe(observed(contact=0.))
    assert m.active.all() and torch.equal(m.anchor,a)
    assert not torch.tensor(event['explicit_release']).any()


def test_combined_lost_geometry_and_contact_is_unknown_and_reacquisition_ambiguous(settings):
    m=RelationMemory('C2',settings);m.observe(observed())
    event=m.observe(observed(distance=.11,contact=0.))
    assert torch.tensor(event['suspended_ambiguous']).all() and not m.active.any()
    assert not torch.tensor(event['explicit_release']).any()
    event=m.observe(observed(location=.05))
    assert torch.tensor(event['regrasp_ambiguous']).all()
    assert torch.equal(m.anchor,torch.full((1,2,3),.05))


def test_explicit_release_is_hand_specific(settings):
    m=RelationMemory('C2',settings);m.observe(observed())
    m.observe(observed(),explicit_release=torch.tensor([[True,False]]))
    assert m.active.tolist()==[[False,True]]


def test_candidate_contact_cannot_disable_energy_and_history_is_excluded(settings):
    m=RelationMemory('C2',settings);m.observe(observed())
    geom=observed(distance=.15,contact=0.,location=.1)
    geom={k:v.repeat(1,8,1,1) if v.ndim==4 else v.repeat(1,8,1) for k,v in geom.items()}
    first=relation_energy(geom,m,settings)[0]
    geom['contact'].fill_(1);geom['relative'][:,:2]=10.;geom['distance'][:,:2]=10.
    assert torch.equal(relation_energy(geom,m,settings)[0],first)
    assert first>0


class IdentityDataset:
    def denormalize_torch(self,x,is_object=False):return x


def fixture_geometry():
    torch.manual_seed(42)
    clean=torch.zeros(1,16,232)
    clean[...,84:216]=transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(torch.randn(1,16,22,3)*.1)).flatten(-2)
    clean[...,219:228]=transforms.axis_angle_to_matrix(torch.randn(1,16,3)*.1).flatten(-2)
    clean[...,216:219]=torch.tensor([.5,.8,.1])
    clean[...,228:230]=1
    offsets=torch.randn(24,3)*.1
    mat=torch.eye(4)[None];mat[:,:3,:3]=transforms.axis_angle_to_matrix(torch.tensor([[0.,.7,0.]]))
    mat[:,:3,3]=torch.tensor([2.,0.,3.])
    context=dict(mat=mat,obj_rot_mat_prefix=transforms.axis_angle_to_matrix(torch.tensor([0.,-.3,0.])),
                 obj_rot_mat_ref=transforms.axis_angle_to_matrix(torch.tensor([[.1,.2,.3]])))
    rest=torch.randn(100,3)*.2
    return clean,offsets,context,rest


def test_differentiable_geometry_matches_existing_native_fk_and_full_surface():
    clean,offsets,context,rest=fixture_geometry();dataset=IdentityDataset()
    g=relation_geometry(clean,dataset,offsets,context,rest)
    native=decode_motion(clean,dataset,offsets,context,rest[None])
    relative=(native['object_rotation_world'][:,:,None].transpose(-1,-2) @
              (native['human'][...,22:24,:]-native['object_translation_world'][:,:,None])[...,None]).squeeze(-1)
    distance=torch.cdist(native['human'][...,22:24,:].flatten(0,1),native['object_surface'].flatten(0,1)).amin(-1).reshape(1,16,2)
    assert torch.allclose(g['human'],native['human'],atol=2e-6)
    assert torch.allclose(g['relative'],relative,atol=2e-6)
    assert torch.allclose(g['distance'],distance,atol=2e-6)


def test_object_local_relation_survives_common_world_frame_change():
    clean,offsets,context,rest=fixture_geometry();dataset=IdentityDataset()
    first=relation_geometry(clean,dataset,offsets,context,rest)
    changed=copy.deepcopy(context)
    common=transforms.axis_angle_to_matrix(torch.tensor([0.,1.2,0.]))
    changed['mat'][:,:3,:3]=common@context['mat'][:,:3,:3]
    changed['mat'][:,:3,3]=(common@context['mat'][:,:3,3,None]).squeeze(-1)+torch.tensor([4.,0.,-2.])
    changed['obj_rot_mat_prefix']=common@context['obj_rot_mat_prefix']
    second=relation_geometry(clean,dataset,offsets,changed,rest)
    assert torch.allclose(first['relative'],second['relative'],atol=2e-6)
    assert torch.allclose(first['distance'],second['distance'],atol=2e-6)


def test_guidance_has_finite_nonzero_joint_gradient_and_preserves_history(settings):
    clean,offsets,context,rest=fixture_geometry()
    guide=RelationalGuidance('C2',settings);guide.memory.observe(observed())
    guide.dataset=IdentityDataset();guide.context=context;guide.offsets=offsets;guide.rest=rest
    guide.variance=torch.ones(500)*1e-5;guide.has_active=True;guide.telemetry=[]
    posterior=clean.clone();fixed=clean[:,:2].clone()
    result=guide.apply(posterior,clean,fixed,9)
    assert torch.isfinite(result).all()
    assert torch.equal(result[:,:2],fixed)
    assert not torch.equal(result[:,2:,84:216],clean[:,2:,84:216])
    assert not torch.equal(result[:,2:,216:219],clean[:,2:,216:219])
    assert torch.equal(result[...,228:232],clean[...,228:232])
    assert guide.telemetry[0][-1]==0
    assert guide.apply(posterior,clean,fixed,10) is posterior


def test_unknown_state_applies_zero_updates(settings):
    guide=RelationalGuidance('C2',settings);guide.has_active=False
    x=torch.zeros(1,16,232)
    assert guide.apply(x,x,x[:,:2],9) is x


def test_quality_gate_keeps_anchor_and_intermediate_progress_as_proxies():
    from mixer.relational_sampling import direct_quality
    limits=json.loads(Path('experiments/protocols/p2_continuation_outcomes_s42_20260907.json').read_text())['labels']
    row=dict(hands=[dict(fixed_active=True,surface_mean_m=.01,coverage_5cm=1.,anchor_vs_W0_m=0.)]*2,
             support_speed_m_per_s=.02,world_joint_speed_m_per_s=.5,human_goal_error_cm=1.,object_goal_error_3D_cm=1.)
    changed=copy.deepcopy(row)
    changed['human_goal_error_cm']=100.;changed['object_goal_error_3D_cm']=100.
    for h in changed['hands']:h['anchor_vs_W0_m']=.5
    assert direct_quality(changed,row,limits)==[]
    changed['hands'][0]['coverage_5cm']=.8
    assert 'hand0:coverage' in direct_quality(changed,row,limits)


def test_hydra_relational_job_declares_runtime_options(monkeypatch):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    monkeypatch.setenv('ROOT_DIR',str(Path.cwd()))
    with initialize_config_dir(config_dir=str(Path('code/config').resolve()),version_base=None):
        cfg=compose(config_name='config_sample_hosi_relational_sampling',overrides=[
            'relational_sampling.enabled=true','relational_sampling.scene=test',
            'relational_sampling.state_ids=[state-000]','dataset.load_object_payload=false','dataset.vis=true'])
    resolved=OmegaConf.to_container(cfg,resolve=True)
    assert resolved['relational_sampling']['enabled']
    assert not resolved['continuation']['enabled']
    assert resolved['sampler']['pelvis']['_target_']=='mixer.candidate_selection.CandidatePoolSampler'
