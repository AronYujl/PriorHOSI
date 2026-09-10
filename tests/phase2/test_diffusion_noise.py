"""DDIM derivatives, native interpolation and cross-window latent ownership."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from pytorch3d import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'code'))
from mixer.diffusion_noise import (ddim_transition, native_quaternion_interpolation,
    native_linear_interpolation, HSIDDIM, _PhysicalLoss)


def test_ddim_oracle_inversion_and_reverse_recover_clean_and_gradient():
    clean = torch.tensor([.4, -.3], dtype=torch.double)
    noise = torch.tensor([-.7, .2], dtype=torch.double, requires_grad=True)
    alpha = torch.tensor(.2, dtype=torch.double)
    value = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
    inverted = ddim_transition(value, clean, alpha, alpha.new_tensor(0.))
    torch.testing.assert_close(inverted, noise)
    back = ddim_transition(value, clean, alpha, alpha.new_tensor(1.))
    torch.testing.assert_close(back, clean)
    torch.testing.assert_close(torch.autograd.grad(inverted.sum(), noise)[0], torch.ones_like(noise))


def test_native_slerp_values_and_finite_derivatives_at_identical_rotations():
    from utils import interp_jrot
    angles = torch.zeros(5, 22, 3, dtype=torch.double)
    angles[2:, :, 1] = .2
    quaternion = transforms.axis_angle_to_quaternion(angles).requires_grad_(True)
    actual = native_quaternion_interpolation(quaternion)
    torch.testing.assert_close(actual, interp_jrot(quaternion.detach()).double(), atol=1e-7, rtol=1e-7)
    gradient = torch.autograd.grad(actual.square().sum()+actual[..., 2].sum(), quaternion)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_linear_interpolation_matches_native_values_without_detaching():
    from utils import interpolate_joints
    value = torch.tensor([[.3, .5], [.7, -.4], [1., 1.]], requires_grad=True)
    result = native_linear_interpolation(value)
    torch.testing.assert_close(result, interpolate_joints(value, 3))
    result.sum().backward()
    assert (value.grad > 0).all()


def test_ddim_generated_history_propagates_later_loss_to_earlier_latent():
    decoder = HSIDDIM.__new__(HSIDDIM)
    decoder.windows = [None, None]
    decoder.settings = dict(ddim_steps=2)
    decoder.alpha = torch.linspace(.99, .01, 500)
    decoder.clean = torch.zeros(2, 16, 216)
    decoder.mats = [None, None]
    decoder.reframe = lambda value, old, new: value
    decoder.predict = lambda current, history, window, view, step: torch.cat(
        (history, .1*current[:, 2:]+.5*history.mean(1, keepdim=True)), 1)
    latent = torch.ones(1, 216, 1, 28, requires_grad=True)
    result = decoder.decode(latent, 'correct')
    result[1, 2:].sum().backward()
    assert latent.grad[..., :14].abs().sum() > 0
    assert latent.grad[..., 14:].abs().sum() > 0
    assert torch.equal(result[0, :2], decoder.clean[0, :2])
    torch.testing.assert_close(result[1, :2], result[0, -2:])


def test_native_pose_lift_preserves_source_and_backpropagates_rotation_and_root():
    decoder = HSIDDIM.__new__(HSIDDIM)
    decoder.mats = [torch.eye(4)[None], torch.eye(4)[None]]
    decoder.dataset = SimpleNamespace(denormalize_torch=lambda value: value,
        quat_ik_torch=lambda rotation: torch.cat((rotation[:, :1],
            rotation[:, :1].transpose(-1, -2)@rotation[:, 1:]), 1))
    decoder.dataset_translation = torch.zeros(3)
    clean = torch.zeros(2, 16, 216)
    clean[..., 84:] = transforms.matrix_to_rotation_6d(torch.eye(3)).repeat(22)
    decoder.source = dict(pose=torch.zeros(90, 22, 3), translation=torch.zeros(90, 3))
    decoder.source_decode = decoder.raw_pose(clean)
    pose, translation = decoder.pose(clean)
    torch.testing.assert_close(pose, decoder.source['pose'], atol=1e-6, rtol=0)
    torch.testing.assert_close(translation, decoder.source['translation'], atol=1e-6, rtol=0)
    torch.manual_seed(42)
    prediction = (clean+torch.randn_like(clean)*.01).requires_grad_(True)
    pose, translation = decoder.pose(prediction)
    (pose.square().sum()+translation.square().sum()).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[..., :3].abs().sum() > 0
    assert prediction.grad[..., 84:].abs().sum() > 0
    assert torch.equal(pose[:6], decoder.source['pose'][:6])
    assert torch.equal(pose[-3:], decoder.source['pose'][-3:])


def test_native_chunked_physical_derivative_matches_full_sequence(monkeypatch):
    import utils
    from mixer.body_projection import NATIVE_ANCHORS
    def body(pose, translation, *args, **kwargs):
        joints = pose[:, :1].expand(-1, 28, -1)+translation[:, None]
        return joints, joints
    monkeypatch.setattr(utils, 'run_smplx_model', body)
    torch.manual_seed(42)
    pose = (torch.randn(53, 22, 3, dtype=torch.double)*.01).requires_grad_(True)
    translation = (torch.randn(53, 3, dtype=torch.double)*.01).requires_grad_(True)
    reference = torch.randn(53, 28, 3, dtype=torch.double)*.01
    from mixer.diffusion_noise import NativeDNOObjective
    objective = NativeDNOObjective.__new__(NativeDNOObjective)
    objective.__dict__.update(source=dict(joints=reference, betas=None, gender=None),
        model=None, body_scale=.05, scene_scale=2., edit=True,
        weights=[1.]*4,differential=SimpleNamespace(frame_sums=lambda vertices: vertices.square().sum((1, 2))))
    result = _PhysicalLoss.apply(pose, translation, objective)
    actual = torch.autograd.grad(result, (pose, translation))
    joints, _ = body(pose, translation)
    delta = joints-reference
    expected = (delta.square().mean()/.05**2+delta[:, NATIVE_ANCHORS].square().mean()/.001**2
        +((delta[1:]-delta[:-1])*30).square().mean()/.1**2+joints.square().sum()/53/2)
    wanted = torch.autograd.grad(expected, (pose, translation))
    torch.testing.assert_close(result, expected)
    for a, b in zip(actual, wanted): torch.testing.assert_close(a, b)


def test_official_dno_optimizer_checkpoint_continuation_matches_uninterrupted(tmp_path):
    sys.path.insert(0, '/data/yujinlun/Diffusion-Noise-Optimization')
    from dno import DNO, DNOOptions
    initial = torch.tensor([[[[.8, 1.1, .4, -.2]]]])
    options = DNOOptions(num_opt_steps=6, lr=.05, lr_warm_up_steps=0, decorrelate_scale=0.)
    model = lambda value: value
    criterion = lambda value: (value-.5).square().flatten(1).mean(1)
    complete = DNO(model, criterion, initial, options)
    complete(6)
    interrupted = DNO(model, criterion, initial, options)
    interrupted(3)
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(latent=interrupted.current_z.detach(), initial=interrupted.start_z,
        optimizer=interrupted.optimizer.state_dict(), step=interrupted.step_count,
        options=vars(options), cpu_rng=torch.get_rng_state()), path)
    saved = torch.load(path, weights_only=False)
    resumed = DNO(model, criterion, saved['initial'], DNOOptions(**saved['options']))
    with torch.no_grad(): resumed.current_z.copy_(saved['latent'])
    resumed.optimizer.load_state_dict(saved['optimizer'])
    resumed.step_count = saved['step']
    torch.set_rng_state(saved['cpu_rng'])
    resumed(3)
    assert torch.equal(resumed.current_z, complete.current_z)


def test_complete_dno_target_retains_planar_motion_inside_existing_bounds():
    from mixer.body_projection import smooth_body_target
    length = 120
    source = dict(pose=torch.zeros(length,22,3), translation=torch.zeros(length,3),
                  joints=torch.zeros(length,28,3))
    full = {k:v.clone() for k,v in source.items()}
    full['pose'][:,0,1] = .1
    full['translation'][:,0] = .02
    rotation, translation = smooth_body_target(source, full, keep_planar=True)
    assert float(translation[60,0]) > .019
    assert float(transforms.matrix_to_axis_angle(rotation)[60,0,1]) > .099
    assert torch.equal(translation[:6], source['translation'][:6])
    assert torch.equal(translation[-3:], source['translation'][-3:])
    old_rotation, old_translation = smooth_body_target(source, full)
    assert old_translation.abs().max() == 0
    torch.testing.assert_close(old_rotation, torch.eye(3).expand(length,22,3,3), atol=1e-6, rtol=0)


def test_inversion_stops_at_decoder_endpoint_instead_of_discarding_clean_component():
    decoder = HSIDDIM.__new__(HSIDDIM)
    decoder.windows = [None]
    decoder.settings = dict(inversion_steps=5, inversion_terminal='model')
    decoder.alpha = torch.linspace(.98, .02, 500, dtype=torch.double)
    decoder.clean = torch.full((1,16,216), .4, dtype=torch.double)
    oracle = torch.full_like(decoder.clean, .1)
    decoder.predict = lambda current, history, window, view, step: oracle
    actual = decoder.invert()
    initial_noise = (decoder.clean-decoder.alpha[0].sqrt()*oracle)/(1-decoder.alpha[0]).sqrt()
    expected = decoder.alpha[-1].sqrt()*oracle+(1-decoder.alpha[-1]).sqrt()*initial_noise
    torch.testing.assert_close(actual[0,:,0].T, expected[0,2:])
    decoder.settings.pop('inversion_terminal')
    legacy = decoder.invert()
    torch.testing.assert_close(legacy[0,:,0].T, initial_noise[0,2:])
    assert not torch.allclose(actual, legacy)


def test_gradient_diagnosis_separates_loss_size_from_update_direction():
    from mixer.diffusion_noise import latent_gradient_measures
    latent = torch.tensor([1.,2.], dtype=torch.double, requires_grad=True)
    terms = dict(body=(latent[0]-3).square()+100000., feature=latent[1].square(),
                 decorrelation=(latent[0]+1).square()/1000.)
    result = latent_gradient_measures(terms, latent, 1000.)
    assert result['losses']['body'] > 100000
    assert result['norms']['body'] == 4.
    assert result['norms']['decorrelation1000'] == 4.
    assert abs(result['cosines']['reconstruction__decorrelation1000']+2**-.5) < 1e-10
    assert abs(result['cosines']['reconstruction__total']-2**-.5) < 1e-10
    unregularized = latent_gradient_measures(terms, latent, 0.)
    assert abs(unregularized['cosines']['reconstruction__total']-1) < 1e-10


def test_source_history_intervention_cuts_cross_window_dependence_and_preserves_first_window():
    decoder=HSIDDIM.__new__(HSIDDIM)
    decoder.windows=[None,None];decoder.settings=dict(ddim_steps=2)
    decoder.alpha=torch.linspace(.99,.01,500);decoder.clean=torch.zeros(2,16,216)
    decoder.mats=[None,None];decoder.reframe=lambda value,old,new:value
    decoder.predict=lambda current,history,window,view,step:torch.cat(
        (history,.1*current[:,2:]+.5*history.mean(1,keepdim=True)),1)
    latent=torch.ones(1,216,1,28,requires_grad=True)
    generated=decoder.decode(latent,'correct',source_history=False)
    source=decoder.decode(latent,'correct',source_history=True)
    assert torch.equal(generated[0],source[0])
    assert torch.equal(source[1,:2],decoder.clean[1,:2])
    generated_grad,=torch.autograd.grad(generated[1,2:].sum(),latent)
    source_grad,=torch.autograd.grad(source[1,2:].sum(),latent)
    assert generated_grad[...,:14].abs().sum()>0
    assert source_grad[...,:14].abs().sum()==0
    assert source_grad[...,14:].abs().sum()>0


def test_window_diagnostics_cover_native_frames_and_separate_history_from_boundary_error():
    from mixer.diffusion_noise import history_window_measures
    decoder=SimpleNamespace(clean=torch.zeros(2,16,216),mats=[None,None],
        source=dict(joints=torch.zeros(90,28,3)),
        dataset=SimpleNamespace(denormalize_torch=lambda value:value),
        reframe=lambda value,old,new:value)
    prediction=decoder.clean.clone();prediction[0,2:]=.1
    joints=torch.zeros(90,28,3);joints[:48,:,0]=.01;joints[48:,:,0]=.02
    rows,distance=history_window_measures(decoder,prediction,joints)
    assert [(r['native_start'],r['native_stop']) for r in rows]==[(0,48),(48,90)]
    assert abs(rows[0]['body_mean_cm']-1)<1e-6 and abs(rows[1]['body_mean_cm']-2)<1e-6
    assert rows[1]['input_history_feature_mse']==0
    assert rows[1]['clean_history_feature_max_error']==0
    assert abs(rows[1]['boundary_feature_mse']-.01)<1e-6
    torch.testing.assert_close(distance.mean(),torch.tensor((48+42*2)/90))


def _constrained_fixture(monkeypatch,length=53,height=0.):
    import utils
    from mixer.diffusion_noise import ConstrainedDNOObjective
    offsets=torch.zeros(28,3,dtype=torch.double);offsets[26,0]=1.
    def body(pose,translation,*args,**kwargs):
        joints=pose[:,:1]+translation[:,None]+offsets+1e-7
        return joints,joints
    monkeypatch.setattr(utils,'run_smplx_model',body)
    translation=torch.zeros(length,3,dtype=torch.double);translation[:,1]=height
    source=dict(pose=torch.zeros(length,22,3,dtype=torch.double),translation=translation,
        joints=offsets[None].expand(length,-1,-1)+translation[:,None],
        verts=offsets[None].expand(length,-1,-1)+translation[:,None],betas=None,gender=None,
        object_rotation=torch.eye(3,dtype=torch.double).expand(length,-1,-1),
        object_translation=torch.zeros(length,3,dtype=torch.double))
    projection=SimpleNamespace(source=source,translation=translation,fixed=torch.zeros(length,dtype=torch.bool),
        seams=torch.tensor([48] if length>48 else [],dtype=torch.long))
    objective=ConstrainedDNOObjective(projection,torch.nn.Identity(),torch.ones(5,5,5),
        dict(centroid=[0,0,0],extents=[10,10,10]),1.,0.,torch.zeros(1,3,dtype=torch.double))
    return source,objective,body


def test_masked_native_constraint_derivatives_and_exact_zero_despite_source_fk_rounding(monkeypatch):
    source,objective,body=_constrained_fixture(monkeypatch)
    pose=source['pose'].clone().requires_grad_(True);translation=source['translation'].clone().requires_grad_(True)
    zero=objective(pose,translation)
    gradients=torch.autograd.grad(zero,(pose,translation))
    assert float(zero)==0 and all(g.abs().sum()==0 for g in gradients)
    assert objective.source_fk_reference_max_error_m>0
    torch.manual_seed(42)
    pose=(pose.detach()+torch.randn_like(pose)*.002).requires_grad_(True)
    translation=(translation.detach()+torch.randn_like(translation)*.002).requires_grad_(True)
    objective.weights=[.7,1.3,.4,1.1,.8,1.4,.6]
    actual=objective(pose,translation);actual_grad=torch.autograd.grad(actual,(pose,translation))
    joints,_=body(pose,translation);reference,_=body(source['pose'],source['translation'])
    delta=joints-reference;velocity=(delta[1:]-delta[:-1])*30
    expected_terms=torch.stack((delta.square().mean()/.05**2,
        delta[:,24].square().mean()/.01**2,delta[:,[7,8,10,11]].square().mean()/.005**2,
        delta[42:48].square().mean()/.01**2,velocity.square().mean()/.1**2,
        velocity[47].square().mean()/.1**2,delta.sum()*0))
    expected=(expected_terms*expected_terms.new_tensor(objective.weights)).sum()
    wanted=torch.autograd.grad(expected,(pose,translation))
    torch.testing.assert_close(actual,expected)
    for a,b in zip(actual_grad,wanted):torch.testing.assert_close(a,b)


def test_empty_contact_stance_and_boundary_sets_have_zero_physical_contribution(monkeypatch):
    source,objective,_=_constrained_fixture(monkeypatch,length=48,height=3.)
    pose=source['pose'].clone().requires_grad_(True)
    translation=(source['translation']+.001).requires_grad_(True)
    loss=objective(pose,translation);loss.backward()
    terms=objective.term_record()
    assert terms['body']>0
    assert all(terms[k]==0 for k in ['hand','stance','boundary','seam_velocity'])
    assert torch.isfinite(pose.grad).all() and torch.isfinite(translation.grad).all()
