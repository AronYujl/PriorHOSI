"""Contact-vector Jacobians, nullspace math and matched local editor semantics."""
import json
import pytest
import torch

from tests.phase2.test_relational import _geometry
from tests.phase2.test_scene_evidence import QuerySampler
from mixer.relational import RelationalObjective
from mixer.relation_projection import (contact_jacobian, nullspace_projection,
    project_dp_gradient, parameter_dp_proxy)
from mixer.scene_evidence import SceneEvidenceEditor, local_armijo


def setup_geometry():
    geometry, _ = _geometry()
    geometry.base[..., 228:230] = 1.
    geometry.dataset.get_nearest_free_voxel = lambda p, f: (torch.zeros(p.shape[:-1],device=p.device,dtype=torch.bool), p)
    p = geometry.base.new_zeros(1,16,67)
    objective = RelationalObjective(geometry, torch.zeros(1,dtype=torch.long),
                                    geometry.decode(p)['human'][:,2:])
    return geometry, objective, p


@pytest.mark.parametrize('nonzero',[False, True])
def test_residual_contact_objective_and_jacobian_directional_difference(nonzero):
    geometry, objective, p = setup_geometry()
    objective.contact[:,3:6,1] = False
    if nonzero:
        p = torch.randn_like(p)*.15
    jac, _ = contact_jacobian(objective, p)
    direction = torch.randn_like(p); direction[:,:2] = 0
    step = .002
    plus = objective.contact_residual(geometry.decode(p+step*direction))
    minus = objective.contact_residual(geometry.decode(p-step*direction))
    fd = ((plus-minus)/(2*step)*objective.contact[...,None]).flatten(-2)
    predicted = (jac @ direction[:,2:, :, None]).squeeze(-1)
    torch.testing.assert_close(predicted,fd,atol=1e-4,rtol=.005)
    residual = objective.contact_residual(geometry.decode(p))
    expected = (residual.square().sum(-1)*objective.contact).sum()/objective.contact.sum()/(3*.05**2)
    terms,_=objective.evaluate(geometry.decode(p))
    torch.testing.assert_close(terms['contact'][0],expected)
    if not nonzero:
        assert residual.abs().max()<2e-6
        assert jac.norm()>0.01


def test_frame_blocks_match_full_window_jacobian_at_nonzero_state():
    geometry,objective,p=setup_geometry()
    p=torch.randn_like(p)*.1
    # One active hand/frame makes the independent full Jacobian compact.
    objective.contact[:]=False;objective.contact[:,7,0]=True
    jac,_=contact_jacobian(objective,p)
    def residual(a):
        return objective.contact_residual(geometry.decode(a))[objective.contact].flatten()
    full=torch.autograd.functional.jacobian(residual,p)
    torch.testing.assert_close(jac[0,7,:3],full[:,0,9],atol=1e-7,rtol=1e-6)
    full[:,:,9]=0
    assert full.count_nonzero()==0
    direction=torch.randn_like(p)
    block,_=project_dp_gradient(objective,p,direction,jac)
    # Independent full-window projection on all future columns.
    full=torch.autograd.functional.jacobian(residual,p)[:,0,2:].flatten(1)
    expected,_=nullspace_projection(full,direction[:,2:].flatten())
    torch.testing.assert_close(block[:,2:].flatten(),expected,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_svd_projection_rank_idempotence_orthogonality_and_degenerate_inputs(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    generator=torch.Generator(device=device).manual_seed(42)
    jac=torch.randn(14,6,67,device=device,generator=generator)
    jac[:,5]=jac[:,4]
    direction=torch.randn(14,67,device=device,generator=generator)
    projected,record=nullspace_projection(jac,direction)
    assert record['rank']==[5]*14
    assert record['normalized_residual']<1e-7
    assert (jac.double()@projected.double()[...,None]).abs().max()<3e-6
    second,_=nullspace_projection(jac,projected)
    torch.testing.assert_close(second,projected,atol=3e-7,rtol=1e-5)
    assert abs(float(((direction-projected)*projected).sum()))<2e-5
    assert projected.norm()<=direction.norm()
    zero,record=nullspace_projection(jac,torch.zeros_like(direction))
    assert not zero.any() and record['reason']=='zero_direction'
    unchanged,record=nullspace_projection(torch.zeros_like(jac),direction)
    assert torch.equal(unchanged,direction) and record['reason']=='rank_zero'
    jac[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):nullspace_projection(jac,direction)


def test_common_source_motion_survives_and_anchors_stay_frozen():
    geometry,objective,p=setup_geometry()
    anchor=objective.hand_anchor.clone()
    direction=torch.zeros_like(p);direction[:,2:,:4]=torch.randn_like(direction[:,2:,:4])
    projected,record=project_dp_gradient(objective,p,direction)
    torch.testing.assert_close(projected,direction,atol=1e-5,rtol=1e-5)
    assert (projected-direction).norm()/direction.norm()<1e-5
    for a in (p+.1,p+.3):
        jac,residual=contact_jacobian(objective,a)
        assert residual.norm()>0
        assert torch.equal(anchor,objective.hand_anchor)
    source_jac,_=contact_jacobian(objective,p)
    assert not torch.equal(source_jac,jac)
    objective.contact[:]=False
    projected,record=project_dp_gradient(objective,p,direction)
    assert torch.equal(projected,direction) and record['reason']=='no_active_contact'


def test_parameter_proxy_scale_stop_gradient_and_armijo():
    p=torch.zeros(1,4,2)
    gradient=torch.tensor([[[1.,-2.]]]).expand_as(p).clone().requires_grad_(True)
    a=p.clone().requires_grad_(True)
    value=parameter_dp_proxy(gradient,a,p,26.)
    actual,=torch.autograd.grad(value.sum(),a)
    assert torch.equal(actual,26*gradient)
    assert gradient.grad is None
    def energy(a):return parameter_dp_proxy(gradient,a,p,26.)+.5*a.square().flatten(1).sum(1)
    result,trace=local_armijo(p,energy,lambda a:torch.ones(1,dtype=torch.bool))
    torch.testing.assert_close(result,-26*gradient)
    assert trace['accepted']==[True] and len(trace['trials'])==1


@pytest.mark.parametrize('coefficient',[0.,26.])
def test_editor_default_identity_and_projected_history_rng_and_reference(coefficient):
    geometry,objective,p=setup_geometry()
    sampler=QuerySampler();sampler.dataset=geometry.dataset
    sampler.dataset.scene_grid_torch=torch.tensor([-100.,-100.,-100.,100.,100.,100.,10.,10.,10.])
    context=dict(mat=geometry.mat,obj_rot_mat_prefix=geometry.prefix,obj_rot_mat_ref=geometry.reference,
        scene_flag=torch.zeros(1,dtype=torch.long),seq_name_dict={0:'sub10_cube_0'},
        obj_rest_verts={'cube':geometry.object_points[0]})
    common=dict(enabled=True,noise_levels=(250,100),lambda_dp=coefficient)
    state=torch.get_rng_state().clone()
    original=SceneEvidenceEditor(**common)
    baseline=original.edit(sampler,geometry.base,{},None,context,geometry.offsets,42)
    for project in (False,True):
        editor=SceneEvidenceEditor(**common,dp_proxy='parameter',relation_projection=project)
        output=editor.edit(sampler,geometry.base,{},None,context,geometry.offsets,42)
        assert torch.equal(torch.get_rng_state(),state)
        assert torch.equal(output[:,:2],geometry.base[:,:2])
        assert torch.equal(output[...,228:],geometry.base[...,228:])
        assert torch.isfinite(output).all()
        if coefficient==0:
            assert torch.equal(output,baseline)
            assert editor.records[0]['hsi_teacher_calls']==0
            assert all('projection' not in i for i in editor.records[0]['iterations'])
        elif project:
            assert all(i['projection']['active_rows']==84 for i in editor.records[0]['iterations'])
        json.dumps(editor.audit_dict(),allow_nan=False)


def test_local_gpu_bootstrap_matches_existing_numpy_pairing_and_resamples():
    from mixer.scene_calibration import paired_local_metrics
    from tools.paired_bootstrap import paired_bootstrap
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    first={str(i):{'value':float(i*i)} for i in reversed(range(24))}
    second={str(i):{'value':float(i*i-i*.3+1)} for i in range(24)}
    actual=paired_local_metrics(first,second,'cuda:0')['value']
    # Independent NumPy computation under the established sorted task identity.
    import numpy as np
    names=sorted(first)
    delta=np.array([second[n]['value']-first[n]['value'] for n in names])
    index=np.random.default_rng(42).integers(0,24,size=(10000,24))
    assert actual['delta']==pytest.approx(delta.mean(),abs=1e-12)
    assert actual['ci']==pytest.approx(np.percentile(delta[index].mean(1),[2.5,97.5]),abs=1e-12)
