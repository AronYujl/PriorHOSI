"""Physical invariants and native-surface derivatives for offline motion edits."""
import sys
from pathlib import Path

import pytest
import torch
from pytorch3d import transforms

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from mixer.surface_edit import (spline_basis,edit_envelope,yaw_matrix,
    shared_object_transform,object_frame_hands,terminal_displacements,
    terminal_envelope,terminal_acceptance,SurfaceProblem)


@pytest.mark.parametrize('length',[48,90,342])
def test_cubic_field_partition_and_exact_initial_terminal_locks(length):
    basis=spline_basis(length,'cpu')
    torch.testing.assert_close(basis.sum(1),torch.ones(length))
    assert (basis>=0).all()
    envelope=edit_envelope(length,'cpu')
    assert torch.equal(envelope[:6],torch.zeros(6))
    assert torch.equal(envelope[-3:],torch.zeros(3))
    control=torch.ones(basis.shape[1],3,requires_grad=True)
    field=(basis@control).tanh()*envelope[:,None]
    field.sum().backward()
    assert control.grad.abs().sum()>0
    assert field.abs().max()<1


def test_common_world_transform_preserves_object_frame_hands_under_nontrivial_rotation():
    torch.manual_seed(42)
    human=torch.randn(5,28,3)
    position=torch.randn(5,3)
    rotation=transforms.axis_angle_to_matrix(torch.randn(5,3))
    shift=torch.randn(5,3)*.1; shift[:,1]=0
    yaw=torch.linspace(-.4,.3,5)
    pivot=human[:,0]
    r=yaw_matrix(yaw)
    moved=(r[:,None]@(human-pivot[:,None])[...,None]).squeeze(-1)+pivot[:,None]+shift[:,None]
    obj,rot=shared_object_transform(pivot,position,rotation,shift,yaw)
    torch.testing.assert_close(object_frame_hands(moved,obj,rot),object_frame_hands(human,position,rotation),atol=1e-6,rtol=1e-6)
    independent=obj.clone();independent[:,0]+=.03
    assert (object_frame_hands(moved,independent,rot)-object_frame_hands(human,position,rotation)).norm(dim=-1).min()>.029


def test_signed_surface_axes_scale_and_gradient_match_analytic_world_plane():
    problem=SurfaceProblem.__new__(SurfaceProblem)
    xyz=torch.linspace(-1,1,9)
    # grid_sample query is permuted z,y,x: array axis0 therefore represents x.
    problem.sdf=xyz[:,None,None].expand(9,9,9)[None].clone()
    problem.centroid=torch.zeros(1,3)
    problem.extents=torch.tensor([[4.,3.,2.]])
    points=torch.tensor([[[-.3,.2,.5],[-.1,-.2,.6]]],requires_grad=True)
    distances=problem.signed(points)
    torch.testing.assert_close(distances,points[...,0],atol=1e-7,rtol=1e-6)
    loss=(-distances).clamp_min(0).mean();loss.backward()
    torch.testing.assert_close(points.grad[...,0],torch.full((1,2),-.5))
    torch.testing.assert_close(points.grad[...,1:],torch.zeros(1,2,2))


def test_terminal_uses_planar_pelvis_and_3d_object_capped_smooth_correction():
    joints=torch.zeros(90,28,3);joints[:,0,1]=1.1
    obj=torch.zeros(90,3)
    h,o=terminal_displacements(joints,obj,dict(pelvis_goal=[.11,0,0],object_goal=[0,.2,0]))
    torch.testing.assert_close(h[-1],torch.tensor([.03,0,0]))
    torch.testing.assert_close(o[-1],torch.tensor([0,.05,0]))
    assert torch.equal(h[:6],torch.zeros_like(h[:6]))
    assert torch.equal(o[-3:],o[-1:].expand(3,3))
    assert torch.equal(terminal_envelope(90,'cpu')[:58],torch.zeros(58))


def test_terminal_acceptance_keeps_all_native_constraints_and_does_not_upgrade_success():
    base=dict(completed=False,contact_percent=.7,foot_sliding=.1,
        scene_human_penetration_s_mean=2.,scene_obj_penetration_s_mean=5.)
    candidate=dict(base,completed=True)
    assert terminal_acceptance(base,candidate)==(True,[])
    for key,value in [('contact_percent',.67),('foot_sliding',.13),
                      ('scene_human_penetration_s_mean',2.1),('scene_obj_penetration_s_mean',5.1)]:
        assert not terminal_acceptance(base,dict(candidate,**{key:value}))[0]
    assert not terminal_acceptance(candidate,candidate)[0]


def test_chunked_support_and_field_derivatives_count_each_frame_pair_once():
    problem=SurfaceProblem.__new__(SurfaceProblem)
    problem.length=53
    problem.foot_mask=torch.ones(53,4,dtype=torch.bool)
    problem.foot_pairs=torch.ones(52,4,dtype=torch.bool)
    problem.position_count=212;problem.velocity_count=208;problem.hand_count=0
    source_vertices=torch.ones(53,3,3)
    problem.source=dict(joints=torch.zeros(53,28,3),verts=source_vertices,
        object_translation=torch.ones(53,3),object_rotation=torch.eye(3).repeat(53,1,1))
    problem.objects=lambda p,r:p[:,None].expand(-1,3,-1)
    problem.signed=lambda p:p[...,0]*0+1
    problem.outside=lambda p:p[...,0]*0
    def chunk(parameters,lo,hi):
        time=torch.arange(lo,hi).float()
        field=time[:,None]*parameters[None]
        joints=torch.zeros(hi-lo,28,3)+field[:,None,:1]
        return dict(verts=source_vertices[lo:hi]+field[:,None,:1],joints=joints,
            object_translation=problem.source['object_translation'][lo:hi],
            object_rotation=problem.source['object_rotation'][lo:hi],normalized=field)
    problem.chunk=chunk
    p=torch.tensor([.0001],requires_grad=True)
    actual=problem.objective(p,backward=True)
    derivative=p.grad.clone()
    q=p.detach().clone().requires_grad_(True)
    times=torch.arange(53).float()
    expected=(times*q).square().mean()/.02**2+.2*(30*q).square().sum()/.03**2
    expected=expected+.05*(times*q).square().mean()+.05*q.square().sum()
    expected.backward()
    assert sum(actual.values())==pytest.approx(float(expected),rel=1e-6)
    torch.testing.assert_close(derivative,q.grad,rtol=1e-6,atol=1e-5)


def test_pose_editor_preserves_native_hand_relations_and_locked_frames_with_leg_changes(monkeypatch):
    torch.manual_seed(42)
    offsets=torch.randn(28,3)*.2;offsets[0]=0
    def body(pose,translation,betas,gender,**kwargs):
        local=offsets[None].expand(len(pose),-1,-1).clone()
        local[:,7:9]+=pose[:,1:3]*.1
        rotation=transforms.axis_angle_to_matrix(pose[:,0])
        joints=(rotation[:,None]@local[...,None]).squeeze(-1)+translation[:,None]
        return joints,joints
    monkeypatch.setattr('utils.run_smplx_model',body)
    pose=torch.randn(48,22,3)*.1
    translation=torch.randn(48,3)
    vertices,joints=body(pose,translation,None,None)
    source=dict(pose=pose,translation=translation,verts=vertices,joints=joints,
        betas=torch.zeros(16),gender='neutral',object_translation=translation+torch.tensor([.3,.1,.2]),
        object_rotation=transforms.axis_angle_to_matrix(torch.randn(48,3)*.2))
    problem=SurfaceProblem(source,torch.nn.Linear(1,1),torch.randn(5,3),torch.ones(5,5,5),
        dict(centroid=[0,0,0],extents=[10,10,10]),0,{},'relation_10')
    parameters=problem.parameters()
    with torch.no_grad():
        parameters[:,0]=.4;parameters[:,2]=.3;parameters[:,6]=.5
    result=problem.chunk(parameters,0,48)
    for key in ('pose','translation','verts','joints','object_translation','object_rotation'):
        assert torch.equal(result[key][:6],source[key][:6])
        assert torch.equal(result[key][-3:],source[key][-3:])
    torch.testing.assert_close(object_frame_hands(result['joints'],result['object_translation'],result['object_rotation']),
        object_frame_hands(source['joints'],source['object_translation'],source['object_rotation']),atol=1e-6,rtol=1e-6)
    result['verts'].sum().backward()
    assert torch.isfinite(parameters.grad).all()
    assert parameters.grad[:,:3].abs().sum()>0
