"""Physical invariants and native-surface derivatives for offline motion edits."""
import sys
from pathlib import Path

import pytest
import json
import numpy as np
import torch
from pytorch3d import transforms

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from mixer.surface_edit import (spline_basis,edit_envelope,yaw_matrix,
    shared_object_transform,object_frame_hands,terminal_displacements,
    terminal_envelope,terminal_acceptance,SurfaceProblem)


def test_native_object_sdf_multi_suffix_asset_identity(tmp_path):
    from mixer.surface_edit import load_object_sdf
    np.save(tmp_path/'clothesstand.ply.npy',np.arange(8).reshape(2,2,2))
    (tmp_path/'clothesstand.ply.json').write_text(json.dumps(dict(centroid=[1,2,3],extents=[4,4,4])))
    data,info=load_object_sdf(tmp_path,'clothesstand')
    assert data.shape==(2,2,2) and data[1,1,1]==7
    assert info['centroid']==[1,2,3]


def test_native_pose_decode_is_independent_of_input_tensor_device():
    from utils import native_body_pose
    class Kinematics:
        def quat_ik_torch(self,rot):
            return torch.cat((rot[:,:1],rot[:,:1].transpose(-1,-2)@rot[:,1:]),1)
    torch.manual_seed(42)
    rotations=transforms.axis_angle_to_matrix(torch.randn(16,22,3))
    encoded=transforms.matrix_to_rotation_6d(rotations)
    expected=native_body_pose(encoded,Kinematics())
    assert expected.shape==(48,22,3) and expected.device.type=='cpu'
    assert torch.isfinite(expected).all()
    if torch.cuda.is_available():
        actual=native_body_pose(encoded.cuda(),Kinematics())
        assert torch.equal(actual,expected)


def test_native_hand_activity_uses_direct_distance_at_large_world_coordinates():
    from mixer.surface_edit import native_hand_distances
    joints=torch.ones(1,28,3)*100
    joints[0,24,0]+=.049;joints[0,26,0]+=.051
    vertices=torch.ones(1,30,3)*100
    vertices[:,1:,0]+=torch.arange(1,30)
    distances=native_hand_distances(joints,vertices)
    assert torch.equal(distances<.05,torch.tensor([[True,False]]))


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
    problem.motion_target=None
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


def test_terminal_stage_preserves_rejected_state_and_completed_identity(monkeypatch):
    from mixer.surface_edit import apply_terminal_repair
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    source=dict(pose=torch.zeros(60,22,3),translation=torch.zeros(60,3),
        verts=torch.zeros(60,28,3),joints=torch.zeros(60,28,3),
        object_translation=torch.ones(60,3)*.1,object_rotation=torch.eye(3).repeat(60,1,1))
    calls=[]
    monkeypatch.setattr('mixer.surface_edit.decode_body',lambda state,model:(state['verts']+1,state['joints']))
    class Problem:
        def __init__(self,base,model,obj,sdf,info,floor,task,arm):
            assert arm=='terminal'
            assert torch.equal(base['verts'],source['verts']+1)
            self.base=base
        def solve(self,steps):
            calls.append(steps)
            candidate=dict(self.base,joints=self.base['joints']+.01)
            return candidate,dict(trace=[dict(iteration=20)],steps=steps,best_iteration=20,
                optimization_seconds=0.,parameters=torch.ones(2,3))
    monkeypatch.setattr('mixer.surface_edit.SurfaceProblem',Problem)
    before=dict(completed=False,feet_height=0.,contact_percent=.7,foot_sliding=.1,
                scene_human_penetration_s_mean=1.,scene_obj_penetration_s_mean=2.)
    result,row,candidate,parameters=apply_terminal_repair(source,before,None,None,None,None,{},
        lambda state:dict(before,completed=True))
    assert row['accepted'] and row['attempted'] and row['metrics']['completed']
    assert torch.equal(result['joints'],candidate['joints']) and torch.equal(parameters,torch.ones(2,3))
    result,row,candidate,_=apply_terminal_repair(source,before,None,None,None,None,{},
        lambda state:dict(before,completed=True,scene_human_penetration_s_mean=1.1))
    assert not row['accepted'] and row['rejection_reasons']==['scene_human_penetration_s_mean']
    assert row['metrics']==before and row['candidate_metrics']['completed']
    assert torch.equal(result['joints'],source['joints']) and not torch.equal(candidate['joints'],source['joints'])
    completed=dict(before,completed=True)
    def unexpected(state):raise AssertionError('completed identity must reuse its metrics')
    result,row,_,parameters=apply_terminal_repair(source,completed,None,None,None,None,{},unexpected)
    assert row['accepted'] and not row['attempted'] and row['metrics']==completed
    assert parameters is None and calls==[20,20]
    assert torch.equal(result['pose'],source['pose']) and torch.equal(result['object_translation'],source['object_translation'])


def test_chunked_support_and_field_derivatives_count_each_frame_pair_once():
    problem=SurfaceProblem.__new__(SurfaceProblem)
    problem.motion_target=None
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


def test_native_completion_enters_task_and_scene_paired_analysis(tmp_path,monkeypatch):
    from mixer.surface_edit import summarize_surface
    from tools.paired_bootstrap import discover_metrics
    def paired(first,second,device):
        names=sorted(first)
        metrics=discover_metrics(first,second,names)['analyzed']
        return {k:dict(delta=sum(second[n][k]-first[n][k] for n in names)/len(names),
                       n=len(names)) for k in metrics}
    monkeypatch.setattr('mixer.scene_calibration.paired_local_metrics',paired)
    tasks=[]
    for ordinal in range(2):
        before=dict(completed=bool(ordinal),contact_percent=.7,foot_sliding=.1,
                    scene_human_penetration_s_mean=2.,scene_obj_penetration_s_mean=5.)
        after=dict(before,completed=True)
        audit=dict(mean_joint_change_mm=0.)
        row=dict(task=ordinal,scene=f'scene-{ordinal}',object='box',arms=dict(
            source=dict(metrics=before,audit=audit),
            terminal=dict(metrics=after,input_metrics=before,audit=audit,accepted=True)))
        dest=tmp_path/f'lanes/one/task-{ordinal:03d}'
        dest.mkdir(parents=True)
        (dest/'complete.json').write_text(json.dumps(row))
        tasks.append(dict(canonical_ordinal=ordinal))
    manifest=tmp_path/'tasks.json';manifest.write_text(json.dumps(dict(tasks=tasks)))
    result=summarize_surface(tmp_path,manifest,device='cpu')
    for unit in ('task','scene'):
        comparison=result['contrasts'][unit]['terminal-minus-terminal_input']
        assert comparison['completed']==dict(delta=.5,n=2)
    assert result['terminal_attempts']==result['terminal_recovered_tasks']==1


def test_hsi_root_target_uses_world_metres_and_relative_yaw_only():
    import math
    from mixer.hsi_motion_target import root_heading_delta
    class Dataset:
        def denormalize_torch(self,value):return value*2+7
    source=torch.zeros(1,16,232)
    r=transforms.axis_angle_to_matrix(torch.tensor([.2,.3,.1]))
    source[...,84:90]=transforms.matrix_to_rotation_6d(r)
    prediction=source.clone()
    prediction[...,:3]=torch.tensor([.04,.3,.02])
    prediction[...,84:90]=transforms.matrix_to_rotation_6d(yaw_matrix(torch.tensor([.12]))[0]@r)
    prediction[...,216:]=1000
    mat=torch.eye(4)[None];mat[:,:3,:3]=yaw_matrix(torch.tensor([math.pi/2]));mat[:,:3,3]=3
    delta=root_heading_delta(prediction,source,Dataset(),mat)
    torch.testing.assert_close(delta,torch.tensor([.04,-.08,.12]).expand(1,16,3),atol=1e-6,rtol=1e-5)


def test_hsi_heading_draw_average_respects_angle_wrap():
    import math
    from mixer.hsi_motion_target import mean_targets
    values=torch.tensor([[[.1,.2,math.radians(179)]],[[.3,.4,math.radians(-179)]]])
    result=mean_targets(values)
    torch.testing.assert_close(result[0,:2],torch.tensor([.2,.3]))
    assert abs(float(result[0,2]))==pytest.approx(math.pi)


def test_hsi_target_interpolation_follows_native_clock_bounds_and_locks():
    import math
    from mixer.hsi_motion_target import native_target
    coarse=torch.arange(20).float()[:,None]*torch.tensor([[.002,.003,.001]])
    target=native_target(coarse)
    expected=(coarse[10]*2/3+coarse[11]/3)*edit_envelope(60,'cpu')[31]
    torch.testing.assert_close(target[31],expected)
    assert torch.equal(target[:6],torch.zeros(6,3))
    assert torch.equal(target[-3:],torch.zeros(3,3))
    extreme=native_target(torch.full((20,3),10.))
    assert bool((extreme.abs()<=torch.tensor([.2,.2,math.radians(20)])).all())


def test_wrong_hsi_scene_rotates_entire_world_about_same_start():
    import math
    from mixer.hsi_motion_target import rotated_scene_context
    rotation=yaw_matrix(torch.tensor([math.pi/2]))[0]
    pivot=torch.tensor([2.,0.,-3.]);point=torch.tensor([.2,1.,.4,1.])
    for offset in ([1.,0.,2.],[-1.,0.,5.]):
        mat=torch.eye(4)[None];mat[:,:3,3]=torch.tensor(offset)
        context=dict(mat=mat,obj_rot_mat_prefix=torch.eye(3)[None],
                     pelvis_goal=torch.tensor([[.8,0,0]]),static_occ_cache=dict(goal=1))
        before=mat.clone();wrong=rotated_scene_context(context,pivot)
        actual=(wrong['mat'][0]@point)[:3]
        expected=pivot+rotation@((mat[0]@point)[:3]-pivot)
        torch.testing.assert_close(actual,expected)
        assert torch.equal(context['mat'],before)
        assert wrong['pelvis_goal'] is context['pelvis_goal']
        assert wrong['static_occ_cache']=={} and context['static_occ_cache']==dict(goal=1)
        torch.testing.assert_close(wrong['obj_rot_mat_prefix'][0],rotation)


def test_hsi_target_loss_has_physical_scale_and_chunk_independent_gradient():
    import math
    from mixer.hsi_motion_target import root_target_loss
    control=torch.zeros(53,6,requires_grad=True)
    target=torch.tensor([.05,-.05,math.radians(10)]).expand(53,3)
    loss=sum(root_target_loss(control[lo:lo+24,:3],target[lo:lo+24],53,.25)
             for lo in range(0,53,24))
    assert float(loss)==pytest.approx(.25)
    loss.backward()
    expected=-2*.25/159/torch.tensor([.05,-.05,math.radians(10)])
    torch.testing.assert_close(control.grad[:,:3],expected.expand(53,3))
    assert torch.equal(control.grad[:,3:],torch.zeros(53,3))


def test_hsi_goal_query_condition_matches_model_and_keeps_object_world():
    from mixer.hsi_motion_target import human_goal_context
    context=dict(is_object=torch.ones(1,dtype=torch.bool),
                 obj_rot_mat_prefix=torch.eye(3)[None],obj_rest_verts=dict(box=torch.ones(12,3)),
                 object_goal=torch.tensor([[1.,.8,2.]]),pelvis_goal=torch.tensor([[.3,0.,.7]]),
                 static_occ_cache=dict(goal='object goal patch'))
    corrected=human_goal_context(context)
    assert not corrected['is_object'].any() and context['is_object'].all()
    for key in ('obj_rot_mat_prefix','obj_rest_verts','object_goal','pelvis_goal'):
        assert corrected[key] is context[key]
    assert corrected['static_occ_cache']=={} and context['static_occ_cache']['goal']=='object goal patch'


def test_full_body_factorization_recovers_planar_motion_and_articulation():
    from mixer.diagnostics import factor_body_prediction
    from mixer.surface_edit import yaw_matrix, object_frame_hands
    from pytorch3d import transforms
    torch.manual_seed(42)
    frames = 5
    joints = torch.randn(frames, 28, 3)
    verts = torch.randn(frames, 40, 3)
    pose = torch.zeros(frames, 22, 3)
    source = dict(joints=joints, verts=verts, pose=pose,
        object_translation=torch.randn(frames,3), object_rotation=torch.eye(3).repeat(frames,1,1))
    yaw = torch.linspace(-.4,.4,frames)
    rotation = yaw_matrix(yaw)
    shift = torch.tensor([.12,0.,-.08]).repeat(frames,1)
    pivot = joints[:, :1]
    full = dict(source)
    for key in ('joints','verts'):
        full[key] = pivot+(rotation[:,None]@(source[key]-pivot)[...,None]).squeeze(-1)+shift[:,None]
        full[key][:,:,1] += .06
    full['joints'][:, 20] += torch.tensor([.02, .03, -.01])
    full['pose'] = pose.clone()
    full['pose'][:,0] = transforms.matrix_to_axis_angle(rotation)
    factors,error = factor_body_prediction(source,full)
    assert error < 1e-6
    assert torch.allclose(factors['residual']['joints'][:,0,:][:,[0,2]],joints[:,0,:][:,[0,2]],atol=1e-6)
    assert torch.allclose(factors['residual']['joints'][:,0,1],joints[:,0,1]+.06,atol=1e-6)
    assert torch.allclose(factors['residual']['object_translation'],source['object_translation'])
    ref = object_frame_hands(source['joints'],source['object_translation'],source['object_rotation'])
    planar = factors['planar']
    assert torch.allclose(object_frame_hands(planar['joints'],planar['object_translation'],planar['object_rotation']),ref,atol=1e-6)
    assert torch.equal(factors['full']['joints'],full['joints'])


def test_native_anchor_jacobian_matches_finite_rotation_differences():
    from mixer.body_projection import NativeBodyProjection
    torch.manual_seed(42)
    frames=60
    source=dict(pose=torch.randn(frames,22,3,dtype=torch.float64)*.2,
                translation=torch.randn(frames,3,dtype=torch.float64))
    offsets=torch.randn(24,3,dtype=torch.float64)*.1
    projection=NativeBodyProjection(source,offsets)
    global_rotation,points=projection.forward(projection.rotation,projection.translation)
    jac=projection.jacobian(global_rotation,points)
    direction=torch.randn(frames,69,dtype=torch.float64)
    h=1e-5
    def step(sign):
        rot=projection.rotation@transforms.axis_angle_to_matrix(direction[:,3:].reshape(frames,22,3)*projection.angle_scale*h*sign)
        trans=projection.translation+direction[:,:3]*projection.position_scale*h*sign
        return projection.forward(rot,trans)[1][:,(22,23,7,8,10,11)]
    actual=((step(1)-step(-1))/(2*h)).flatten(1)
    expected=(jac@direction[...,None]).squeeze(-1)
    torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-7)
    inverse=projection.inverse(jac)
    tangent=direction-(inverse@(jac@direction[...,None])).squeeze(-1)
    assert (jac@tangent[...,None]).abs().max()<1e-8


def test_native_anchor_restoration_preserves_non_anchor_body_changes():
    from mixer.body_projection import NativeBodyProjection
    torch.manual_seed(42)
    frames=60
    source=dict(pose=torch.randn(frames,22,3,dtype=torch.float64)*.15,
                translation=torch.zeros(frames,3,dtype=torch.float64))
    offsets=torch.randn(24,3,dtype=torch.float64)*.1
    projection=NativeBodyProjection(source,offsets)
    direction=torch.randn(frames,69,dtype=torch.float64)*.03
    rotation,translation=projection.update(projection.rotation,projection.translation,direction)
    rotation,translation=projection.restore(rotation,translation,20)
    _,points=projection.forward(rotation,translation)
    assert (points[:,(22,23,7,8,10,11)]-projection.reference).norm(dim=-1).max()<1e-6
    assert (points-projection.points).norm(dim=-1).mean()>1e-5
    assert torch.equal(rotation[projection.fixed],projection.rotation[projection.fixed])
    assert torch.equal(translation[projection.fixed],projection.translation[projection.fixed])


def test_full_body_tangent_velocity_matches_finite_kinematics_and_has_descent_dual():
    from mixer.body_projection import (NativeBodyProjection,body_direction_velocity,
                                      tangent_body_direction,normalize_body_directions)
    torch.manual_seed(42)
    source=dict(pose=torch.randn(60,22,3,dtype=torch.float64)*.2,
                translation=torch.randn(60,3,dtype=torch.float64))
    projection=NativeBodyProjection(source,torch.randn(24,3,dtype=torch.float64)*.1)
    raw=torch.randn(60,69,dtype=torch.float64)
    tangent,jacobian=tangent_body_direction(projection,raw)
    h=1e-5
    def points(scale):
        rotation=projection.rotation@transforms.axis_angle_to_matrix(tangent[:,3:].reshape(-1,22,3)*projection.angle_scale*scale)
        return projection.forward(rotation,projection.translation+tangent[:,:3]*projection.position_scale*scale)[1]
    actual=(points(h)-points(-h))/(2*h)
    torch.testing.assert_close(body_direction_velocity(projection,tangent),actual,atol=1e-8,rtol=1e-6)
    assert (jacobian@tangent[...,None]).abs().max()<1e-9
    assert torch.equal(tangent[projection.fixed],torch.zeros_like(tangent[projection.fixed]))
    gradient=-raw
    assert float((gradient*tangent).sum())<0
    torch.testing.assert_close((gradient*tangent).sum(),-tangent.square().sum(),rtol=1e-7,atol=1e-7)
    values,audit=normalize_body_directions(projection,{'a':tangent,'b':tangent*.2,'zero':tangent*0})
    torch.testing.assert_close(values['a'],values['b'],atol=1e-12,rtol=1e-10)
    assert audit['directions']['zero']['zero_direction']
    assert audit['directions']['zero']['achieved_linear_body_rms_m']==0
    assert audit['directions']['a']['achieved_linear_body_rms_m']==pytest.approx(.001*audit['common_scale'])


def test_native_increment_is_exact_identity_and_preserves_boundary_derivatives():
    from mixer.body_projection import NativeBodyProjection,native_increment_pose
    torch.manual_seed(42)
    source=dict(pose=torch.randn(60,22,3)*.2,translation=torch.randn(60,3))
    projection=NativeBodyProjection(source,torch.randn(24,3)*.1)
    direction=torch.zeros(60,69,dtype=torch.float64,requires_grad=True)
    pose,translation=native_increment_pose(projection,direction)
    assert torch.equal(pose,source['pose']) and torch.equal(translation,source['translation'])
    derivative=torch.autograd.grad(translation.sum()+pose.sum(),direction)[0]
    assert torch.isfinite(derivative).all()
    assert torch.equal(derivative[projection.fixed],torch.zeros_like(derivative[projection.fixed]))
    torch.testing.assert_close(derivative[~projection.fixed,:3],torch.full_like(derivative[~projection.fixed,:3],.05))


def test_native_scene_derivative_matches_plane_and_complete_temporal_denominator(monkeypatch):
    from mixer.body_projection import NativeBodyProjection,NativeSceneDifferential
    frames=53
    local=torch.tensor([[-.02,.1,.03],[.01,-.05,.04]])
    def body(pose,translation,*args,**kwargs):
        rotation=transforms.axis_angle_to_matrix(pose[:,0])
        vertices=(rotation[:,None]@local.to(pose)[None,:,:,None]).squeeze(-1)+translation[:,None]
        return vertices,vertices
    monkeypatch.setattr('utils.run_smplx_model',body)
    source=dict(pose=torch.zeros(frames,22,3),translation=torch.tensor([[-.3,.2,.1]]).repeat(frames,1),
                betas=torch.zeros(16),gender='neutral')
    source['verts'],_=body(source['pose'],source['translation'])
    projection=NativeBodyProjection(source,torch.zeros(24,3))
    grid=torch.linspace(-1,1,9)[:,None,None].expand(9,9,9).clone()
    problem=NativeSceneDifferential(projection,torch.nn.Linear(1,1),grid,
                                    dict(centroid=[0,0,0],extents=[4,3,2]))
    gradient,scalar=problem.gradient()
    assert scalar==pytest.approx(.61,abs=1e-6)
    expected=torch.zeros(frames,dtype=torch.float64)
    expected[~projection.fixed]=-2*.05/frames
    torch.testing.assert_close(gradient[:,0],expected,atol=1e-9,rtol=1e-6)
    torch.testing.assert_close(gradient[:,1:3],torch.zeros_like(gradient[:,1:3]),atol=1e-8,rtol=0)
    assert torch.isfinite(gradient).all()


def test_current_native_window_encoding_roundtrips_world_geometry():
    from types import SimpleNamespace
    from mixer.hsi_motion_target import encode_native_window
    from test_infbagel_hosi import decode_sample_window
    class Normalization:
        def normalize_torch(self,x,is_object=False):return x*.5+1
        def denormalize_torch(self,x,is_object=False):return (x-1)*2
    torch.manual_seed(42)
    points=torch.randn(1,16,28,3)
    rotation=transforms.axis_angle_to_matrix(torch.randn(1,16,22,3)*.2)
    obj=torch.randn(1,16,3)
    obj_rotation=transforms.axis_angle_to_matrix(torch.randn(1,16,3)*.2)
    reference=transforms.axis_angle_to_matrix(torch.tensor([[.2,.4,-.1]]))
    mat=torch.eye(4)[None];mat[:,:3,:3]=yaw_matrix(torch.tensor([.7]));mat[:,:3,3]=torch.tensor([[1.,0.,-2.]])
    dataset=Normalization()
    encoded=encode_native_window(dataset,points,rotation,obj,obj_rotation,torch.zeros(1,16,4),mat,reference)
    cfg=SimpleNamespace(batch_size=1,max_window_size=16,dataset=SimpleNamespace(nb_joints=28))
    decoded=decode_sample_window(cfg,encoded,dataset,mat)
    torch.testing.assert_close(decoded['points_orig'].reshape_as(points),points,atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(decoded['obj_trans_orig'],obj,atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(transforms.rotation_6d_to_matrix(decoded['global_rot_6d'].reshape(1,16,22,6)),rotation,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(decoded['object_rot_mat'].reshape(1,16,3,3)@reference[:,None],obj_rotation,atol=1e-6,rtol=1e-6)


def test_teacher_residual_lifting_cancels_reference_resampling_error():
    from mixer.hsi_motion_target import lift_native_prediction
    torch.manual_seed(42)
    base=dict(pose=torch.randn(12,22,3)*.2,translation=torch.randn(12,3))
    reference=dict(pose=base['pose']+.02,translation=base['translation']+.01)
    pose,translation=lift_native_prediction(base,reference,reference)
    assert torch.equal(pose,base['pose'])
    assert torch.equal(translation,base['translation'])
    prediction=dict(pose=reference['pose'].clone(),translation=reference['translation']+torch.tensor([0.,.03,0.]))
    pose,translation=lift_native_prediction(base,reference,prediction)
    assert torch.equal(pose,base['pose'])
    torch.testing.assert_close(translation,base['translation']+torch.tensor([0.,.03,0.]))


def test_coarse_pose_validation_precedes_native_temporal_interpolation():
    from mixer.hsi_motion_target import native_coarse_pose
    from mixer.kinematic_composition import _local_from_global
    from utils import native_body_pose
    class Kinematics:
        quat_ik_torch=staticmethod(_local_from_global)
    expected=torch.zeros(16,22,3,dtype=torch.float64)
    expected[:,0,1]=torch.linspace(0,.001,16,dtype=torch.float64)
    global_rotation=transforms.axis_angle_to_matrix(expected[:,:1]).expand(-1,22,-1,-1)
    encoded=transforms.matrix_to_rotation_6d(global_rotation)
    direct=native_coarse_pose(encoded,Kinematics())
    torch.testing.assert_close(direct,expected,atol=1e-12,rtol=1e-8)
    # The sealed native near-parallel branch has its own time-grid behaviour.
    # It is retained for evaluation; encoded-input validation must precede it.
    interpolated=native_body_pose(encoded,Kinematics(),3)
    assert (interpolated[::3]-expected).abs().max()>1e-5
