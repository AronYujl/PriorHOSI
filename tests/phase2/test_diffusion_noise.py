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
    objective = SimpleNamespace(source=dict(joints=reference, betas=None, gender=None),
        model=None, body_scale=.05, scene_scale=2., edit=True,
        differential=SimpleNamespace(frame_sums=lambda vertices: vertices.square().sum((1, 2))))
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
