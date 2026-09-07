"""Physical time, fixed references and optional shared post-editor solver."""
import pytest
import torch
from pytorch3d import transforms
from tests.phase2.test_conditional_repair import fixture
from mixer.conditional_repair import ConstrainedPoseFit, locked_encode
from mixer.temporal_preservation import (TemporalPreservation, TemporalObjective,
    physical_derivatives, root_local_features)
from mixer.scene_evidence import local_armijo


def time_data():
    return torch.arange(16,dtype=torch.float64)[None]*.1, torch.ones(1,16,dtype=torch.bool), torch.ones(1,15,dtype=torch.bool)


def objective_fixture(parameters=None):
    g,c=fixture()
    fit=ConstrainedPoseFit(g,g,c['scene_flag'],foot_guard_mode='quality')
    p=fit.zero if parameters is None else parameters
    return fit,TemporalObjective(fit,p,*time_data(),.2,2.)


def test_analytic_physical_derivatives_and_time_scaling():
    t,valid,links=time_data()
    for power,velocity,acceleration in [(0,torch.zeros(14),0),(1,torch.ones(14),0),(2,2*t[0,2:]-.1,2)]:
        x=t.pow(power)[...,None,None].expand(-1,-1,21,3)
        v,a,vm,am=physical_derivatives(x,t,valid,links)
        torch.testing.assert_close(v[0,:,0,0],velocity.double(),atol=1e-12,rtol=0)
        torch.testing.assert_close(a,torch.full_like(a,acceleration),atol=1e-12,rtol=0)
        vv,aa,_,_=physical_derivatives(x,2*t,valid,links)
        torch.testing.assert_close(vv,v/2);torch.testing.assert_close(aa,a/4)
        assert vm.sum()==am.sum()==14
    with pytest.raises(ValueError,match='equally spaced'):
        physical_derivatives(x,t.square(),valid,links)


def test_missing_edges_history_boundary_and_empty_masks():
    t,v,l=time_data();x=torch.zeros(1,16,21,3,dtype=torch.float64)
    x[:,0]=.01;x[:,1]=.02
    vel,acc,vm,am=physical_derivatives(x,t,v,l)
    assert vel[0,0,0,0].item()==pytest.approx(-.2)
    torch.testing.assert_close(acc[0,0,0,0],torch.tensor(-3.,dtype=torch.float64))
    l[:,1]=False
    _,_,vm,am=physical_derivatives(x,t,v,l)
    assert not vm[0,0] and not am[0,:2].any()
    fit,_=objective_fixture();v.zero_()
    module=TemporalPreservation(enabled=True,velocity_scale_m_per_s=.2,acceleration_scale_m_per_s2=2.)
    out,p,r=module.apply(fit,fit.geometry.base,fit.zero,t,v,l)
    assert out is fit.geometry.base and r['reason']=='empty_stencils'


@pytest.mark.parametrize('options',[dict(enabled=False),dict(enabled=True,lambda_v=0,lambda_a=0)])
def test_exact_bypass_needs_no_fk_metadata_or_rng(options):
    module=TemporalPreservation(**options);x=torch.randn(1,16,232);p=torch.randn(1,16,67)
    rng=torch.get_rng_state().clone();out,q,r=module.apply(None,x,p)
    assert out is x and q is p and torch.equal(rng,torch.get_rng_state())


def test_source_identity_and_keep_center_gradients_under_outer_no_grad():
    fit,obj=objective_fixture()
    p=fit.zero.clone().requires_grad_()
    terms=obj.terms(p)
    assert terms['velocity_error_m2_per_s2'].item()==0
    assert terms['acceleration_error_m2_per_s4'].item()==0
    assert torch.autograd.grad(obj(p).sum(),p)[0].abs().max()==0
    p=fit.zero.clone();p[:,2:,4+3*11]=.2
    center=TemporalObjective(fit,p,*time_data(),.2,2.)
    q=p.clone().requires_grad_()
    assert torch.autograd.grad(center.terms(q)['keep'].sum(),q)[0].abs().max()==0
    # Source differs from proposal: gradient at zero must survive differentiable decode.
    obj.source_features=obj.source_features.clone();obj.source_features[:,2:,15,0]+=.01
    obj.reference=obj.derivatives(obj.source_features)
    grad=torch.autograd.grad(obj(fit.zero.requires_grad_()).sum(),fit.zero)[0]
    assert torch.isfinite(grad).all() and grad.abs().max()>0
    assert grad[:,:2].abs().max()==0 and grad[...,:4].isfinite().all()
    module=TemporalPreservation(enabled=True,velocity_scale_m_per_s=.2,acceleration_scale_m_per_s2=2.,iterations=2)
    with torch.no_grad():
        out,q,r=module.apply(fit,locked_encode(fit.geometry,p),p,*time_data())
    assert r['reason']=='optimized' and r['changed']
    assert q[:,:2].abs().max()==0 and q[...,:4].abs().max()==0
    assert torch.equal(out[:,:2],fit.geometry.base[:,:2])
    for a,b in [(0,3),(84,90),(216,232)]:assert torch.equal(out[...,a:b],fit.geometry.base[...,a:b])
    assert all(v.all() for v in fit.guards(q).values())
    assert all(trial['value'][0]<step['value'][0] for step in r['steps'] for trial in step['trials'] if trial['accepted'][0])


def test_root_local_features_invariant_to_joint_world_rigid_transform():
    fit,obj=objective_fixture();g=fit.geometry;state=g.decode(fit.zero)
    before=root_local_features(g,state)
    rotation=transforms.axis_angle_to_matrix(torch.tensor([.3,-.5,.2]))
    translation=torch.tensor([1.,2.,3.])
    state['human']=(rotation@state['human'][...,None]).squeeze(-1)+translation
    g.world_rotation=rotation@g.world_rotation
    torch.testing.assert_close(root_local_features(g,state),before,atol=6e-7,rtol=1e-5)


def test_projected_armijo_uses_original_gradient_dot_actual_direction():
    p=torch.tensor([[[1.]]])
    out,r=local_armijo(p,lambda p:p.square().flatten(1).sum(1),lambda p:torch.tensor([True]),
        initial_step=.1,gradient_transform=lambda p,g:g*.1,projected_slope=True)
    assert r['directional_slope'][0]==pytest.approx(-.4)
    assert r['accepted']==[True] and out.item()<1


def test_original_proposal_total_budget_and_anchors_survive_second_stage():
    fit,_=objective_fixture();g=fit.geometry
    p=fit.zero.clone();p[:,2:,4+3*11]=.2
    incoming=locked_encode(g,p)
    anchors=fit.objective.hand_anchor.clone();proposal=fit.proposal['human'].clone();energy=fit.proposal_foot_energy.clone()
    module=TemporalPreservation(enabled=True,velocity_scale_m_per_s=.2,acceleration_scale_m_per_s2=2.,iterations=2)
    module.apply(fit,incoming,p,*time_data())
    assert torch.equal(anchors,fit.objective.hand_anchor)
    assert torch.equal(proposal,fit.proposal['human']) and torch.equal(energy,fit.proposal_foot_energy)
    fit.invalid_proposal=True
    out,q,r=module.apply(fit,incoming,p,*time_data())
    assert out is incoming and q is p and r['reason']=='infeasible_incoming'
