"""DDIM derivatives, native interpolation and cross-window latent ownership."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from pytorch3d import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'code'))
from mixer.diffusion_noise import (ddim_transition, native_quaternion_interpolation,
    native_linear_interpolation, HSIDDIM, _PhysicalLoss)


def _hoi_test_dataset():
    return SimpleNamespace(normalize_torch=lambda v, **k: v, denormalize_torch=lambda v, **k: v)


def _hoi_test_context(angle=0., shift=(0., 0., 0.), object_angle=.1):
    mat = torch.eye(4, dtype=torch.double)[None]
    mat[0, :3, :3] = transforms.axis_angle_to_matrix(torch.tensor([0., angle, 0.], dtype=torch.double))
    mat[0, :3, 3] = torch.tensor(shift, dtype=torch.double)
    return dict(mat=mat, obj_rot_mat_prefix=transforms.axis_angle_to_matrix(
        torch.tensor([[0., object_angle, 0.]], dtype=torch.double)),
        obj_rot_mat_ref=transforms.axis_angle_to_matrix(torch.tensor([[.2, 0., 0.]], dtype=torch.double)))


def _hoi_test_motion(frames=16):
    value = torch.zeros(1, frames, 232, dtype=torch.double)
    value[..., :84] = torch.arange(84, dtype=torch.double)*.01
    value[..., 84:216] = transforms.matrix_to_rotation_6d(torch.eye(3, dtype=torch.double)).repeat(22)
    value[..., 216:219] = torch.tensor([.2, .4, -.1], dtype=torch.double)
    value[..., 219:228] = torch.eye(3, dtype=torch.double).flatten()
    value[..., 228:] = .8
    return value


def test_hoi_frame_change_preserves_world_human_object_and_derivatives():
    from mixer.hoi_diffusion_noise import reframe_hoi
    dataset = _hoi_test_dataset()
    old = _hoi_test_context(.6, (.2, .1, -.3), .4)
    new = _hoi_test_context(-.4, (-.7, .2, .9), -.3)
    value = _hoi_test_motion(2).requires_grad_(True)
    shifted = reframe_hoi(value, dataset, old, new)
    recovered = reframe_hoi(shifted, dataset, new, old)
    torch.testing.assert_close(recovered, value, atol=1e-12, rtol=1e-12)
    old_world = old['obj_rot_mat_prefix'][:, None] @ value[..., 219:228].reshape(1, 2, 3, 3) @ old['obj_rot_mat_ref'][:, None]
    new_world = new['obj_rot_mat_prefix'][:, None] @ shifted[..., 219:228].reshape(1, 2, 3, 3) @ new['obj_rot_mat_ref'][:, None]
    torch.testing.assert_close(old_world, new_world)
    recovered.sum().backward()
    assert torch.isfinite(value.grad).all()
    assert value.grad[..., 216:228].abs().sum() > 0


def _hoi_test_decoder():
    from mixer.hoi_diffusion_noise import HOIDDIM
    decoder = HOIDDIM.__new__(HOIDDIM)
    decoder.dataset = _hoi_test_dataset()
    contexts = [_hoi_test_context(), _hoi_test_context(.3, (.1, 0., .2))]
    decoder.windows = [dict(context=c, arguments=dict(text_embedding=torch.ones(1)), local_bps=None) for c in contexts]
    decoder.mats = torch.cat([c['mat'] for c in contexts])
    decoder.clean = _hoi_test_motion().expand(2, -1, -1).clone()
    decoder.alpha = torch.linspace(.99, .01, 500, dtype=torch.double)
    decoder.times = [499, 0]
    decoder.device = torch.device('cpu')
    decoder.translation_offset = torch.zeros(3, dtype=torch.double)
    decoder.hoi_calls = 0
    def raw(value, timestep, arguments, local_bps):
        assert set(arguments) == {'text_embedding'}
        return .1*value + .7*value[:, :2].mean(1, keepdim=True)
    decoder.sampler = SimpleNamespace(_hoi_raw_x0=raw)
    decoder.teacher = SimpleNamespace(model=lambda *a, **k: (_ for _ in ()).throw(AssertionError('HSI entered HOI generation')))
    return decoder


def test_hoi_ddim_replays_without_hsi_and_backpropagates_across_generated_history():
    decoder = _hoi_test_decoder()
    initial = _hoi_test_motion(28).transpose(1, 2).unsqueeze(2).requires_grad_(True)
    prediction = decoder.decode(initial)
    assert torch.equal(prediction, decoder.decode(initial))
    torch.testing.assert_close(prediction[0, :2], decoder.clean[0, :2], rtol=0, atol=0)
    from mixer.hoi_diffusion_noise import reframe_hoi
    expected = reframe_hoi(prediction[:1, -2:], decoder.dataset, decoder.windows[0]['context'], decoder.windows[1]['context'])
    torch.testing.assert_close(prediction[1:2, :2], expected)
    loss = prediction[1, 2:, :3].square().sum()+prediction[1, 2:, 216:228].square().sum()
    gradient, = torch.autograd.grad(loss, initial)
    assert torch.isfinite(gradient).all()
    assert gradient[..., :14].abs().sum() > 0 and gradient[..., 14:].abs().sum() > 0
    assert gradient[:, :3].abs().sum() > 0 and gradient[:, 216:219].abs().sum() > 0


def test_hoi_reference_and_final_readouts_use_the_same_differentiable_forward():
    decoder = _hoi_test_decoder()
    modes = []
    original = decoder.sampler._hoi_raw_x0
    def observe(value, *args):
        modes.append((torch.is_grad_enabled(), value.requires_grad))
        return original(value, *args)
    decoder.sampler._hoi_raw_x0 = observe
    initial = _hoi_test_motion(28).transpose(1, 2).unsqueeze(2)
    with torch.no_grad():
        source = decoder.sample(initial)
    optimized = decoder.decode(initial.requires_grad_(True))
    assert torch.equal(source, optimized)
    assert all(enabled and required for enabled, required in modes)
    assert not source.requires_grad


def test_hoi_coarse_fk_teacher_loss_reaches_body_rotations_and_previous_latent():
    decoder = _hoi_test_decoder()
    decoder.rest_offsets = torch.ones(24, 3, dtype=torch.double)*.02
    initial = _hoi_test_motion(28).transpose(1, 2).unsqueeze(2).requires_grad_(True)
    predicted = decoder.decode(initial)
    points = decoder.coarse_body(predicted)
    assert points.shape == (2, 16, 24, 3)
    target = points.detach()+points.new_tensor([.03, -.02, .01])
    gradient, = torch.autograd.grad(decoder.teacher_loss(predicted, target), initial)
    assert torch.isfinite(gradient).all()
    assert gradient[:, 84:216].abs().sum() > 0
    assert gradient[..., :14].abs().sum() > 0


def test_hoi_native_object_interpolation_and_initial_history_match_native():
    from utils import interp_object
    decoder = _hoi_test_decoder()
    initial = _hoi_test_motion(28).transpose(1, 2).unsqueeze(2).requires_grad_(True)
    prediction = decoder.decode(initial)
    pose, translation, obj, rotation = decoder.native(prediction)
    world = decoder.world(prediction)
    wanted_obj, wanted_rotation = interp_object(world['object_translation_world'].detach().numpy(),
                                               world['object_rotation_world'].detach().reshape(-1, 9).numpy(), 3)
    # The shared native GPU interpolation uses float32 temporal weights.
    torch.testing.assert_close(obj, torch.from_numpy(wanted_obj), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(rotation, torch.from_numpy(wanted_rotation).reshape(-1, 3, 3), atol=1e-7, rtol=1e-7)
    assert len(pose) == len(translation) == len(obj) == 90
    changed = decoder.native(decoder.decode(initial+.02))
    for first, second in zip((pose, translation, obj, rotation), changed):
        torch.testing.assert_close(first[:3], second[:3], atol=1e-12, rtol=1e-12)
    loss = translation[60:].square().sum()+obj[60:].square().sum()+rotation[60:, 0, 1].sum()+pose[60:].square().sum()
    gradient, = torch.autograd.grad(loss, initial)
    assert torch.isfinite(gradient).all()
    assert gradient[:, 216:219].abs().sum() > 0
    assert gradient[:, 219:228].abs().sum() > 0


def test_hoi_content_and_contact_coordinates_allow_joint_rigid_route_changes():
    from mixer.hoi_diffusion_noise import root_local_body
    from mixer.surface_edit import object_frame_hands
    torch.manual_seed(42)
    pose = torch.randn(7, 22, 3, dtype=torch.double)*.1
    joints = torch.randn(7, 28, 3, dtype=torch.double)
    position = torch.randn(7, 3, dtype=torch.double)
    rotation = transforms.axis_angle_to_matrix(torch.randn(7, 3, dtype=torch.double)*.1)
    turn = transforms.axis_angle_to_matrix(torch.tensor([0., .4, 0.], dtype=torch.double))
    shift = torch.tensor([.8, 0., -.5], dtype=torch.double)
    changed_joints = (turn@joints[..., None]).squeeze(-1)+shift
    changed_position = (turn@position[..., None]).squeeze(-1)+shift
    changed_pose = pose.clone()
    changed_pose[:, 0] = transforms.matrix_to_axis_angle(turn@transforms.axis_angle_to_matrix(pose[:, 0]))
    torch.testing.assert_close(root_local_body(pose, joints), root_local_body(changed_pose, changed_joints))
    torch.testing.assert_close(object_frame_hands(joints, position, rotation),
                               object_frame_hands(changed_joints, changed_position, turn@rotation))


def test_joint_native_objective_chunks_match_whole_motion_and_all_four_derivatives(monkeypatch):
    import utils
    from mixer.hoi_diffusion_noise import JointMotionObjective, _JointPhysicalLoss
    def body(pose, translation, *args, **kwargs):
        joints = pose[:, :1].expand(-1, 28, -1)+translation[:, None]
        return joints, joints
    monkeypatch.setattr(utils, 'run_smplx_model', body)
    torch.manual_seed(42)
    p = torch.randn(53, 22, 3, dtype=torch.double)*.01
    t = torch.randn(53, 3, dtype=torch.double)*.01
    op = torch.randn(53, 3, dtype=torch.double)*.01
    orm = transforms.axis_angle_to_matrix(torch.randn(53, 3, dtype=torch.double)*.01)
    vertices, joints = body(p, t)
    source = dict(pose=p, translation=t, object_translation=op, object_rotation=orm,
                  joints=joints, verts=vertices, betas=torch.zeros(10), gender='male')
    scales = dict(body_m=.05, hand_m=.01, stance_height_m=.01, stance_speed_m_s=.05,
                  local_velocity_m_s=.2, trajectory_acceleration_m_s2=.5, goal_m=.05, domain_m=.05)
    objective = JointMotionObjective(source, None, torch.randn(8, 3, dtype=torch.double)*.01,
        torch.ones(4, 4, 4), dict(centroid=[0, 0, 0], extents=[2, 2, 2]),
        dict(scene_name='fixture', test_idx=0, pelvis_goal=[.1, 0, .1], object_goal=[.1, .1, .1]),
        dict(feet_height=0., scene_human_penetration_s_mean=1., scene_obj_penetration_s_mean=1.), scales)
    objective.scene.frame_sums = lambda v: v.square().sum((1, 2))
    objective.reference_chunks[0, 53] = (vertices, joints)
    values = [(v+torch.randn_like(v)*.001).requires_grad_(True) for v in (p, t, op, orm)]
    actual = _JointPhysicalLoss.apply(*values, objective)
    gradients = torch.autograd.grad(actual, values)
    pp, tt, oo, rr = values
    vv, jj = body(pp, tt)
    expected = objective.chunk_terms(pp, jj, vv, oo, rr, 0, 0, 53).sum()
    wanted = torch.autograd.grad(expected, values)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-10)
    for a, b in zip(gradients, wanted):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-9)
        assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_hoi_teacher_targets_use_paired_noise_and_only_change_scene_arguments(monkeypatch):
    import mixer.hoi_diffusion_noise as module
    from priors.hoi.diffusion import GaussianDiffusion
    from contextlib import nullcontext
    decoder = _hoi_test_decoder()
    decoder.ordinal = 17
    decoder.settings = dict(hsi_level=199)
    decoder.task = dict(start_location=[0., 0., 0.])
    for w in decoder.windows:
        w['context']['is_object'] = torch.ones(1)
    def arguments(clean, previous, timestep, context):
        values = [torch.zeros(1, dtype=clean.dtype) for _ in range(17)]
        values[0] = context['mat'][:, 0, 0]
        values[1] = timestep
        return tuple(values)
    decoder.sampler.inner_hoi = SimpleNamespace(diffusion=GaussianDiffusion())
    decoder.sampler.hsi_sampler = SimpleNamespace(emb_f=0)
    decoder.sampler._hsi_model_arguments = arguments
    decoder.teacher.calls = 0
    decoder.teacher.model = lambda x, *a, **k: .5*x+.1*a[0][:, None, None]
    decoder.coarse_body = lambda x: x[..., :216].reshape(*x.shape[:2], 72, 3)
    monkeypatch.setattr(module.torch.random, 'fork_rng', lambda **k: nullcontext())
    monkeypatch.setattr(module, 'observation_outside', lambda *a: 0.)
    prediction = decoder.clean.clone().requires_grad_(True)
    correct, c = decoder.teacher_target(prediction, 'correct', 7, capture=True)
    wrong, w = decoder.teacher_target(prediction, 'wrong', 7, capture=True)
    assert not correct.requires_grad and not wrong.requires_grad
    assert not torch.equal(correct, wrong)
    assert torch.equal(prediction, decoder.clean)
    for first, second in zip(c, w):
        assert torch.equal(first['noisy'], second['noisy'])
        assert (first['noisy'][:, :2, 216:] == 0).all()
        assert all(torch.equal(first['arguments'][i], second['arguments'][i]) for i in range(17) if i not in (0, 15))
    gradient, = torch.autograd.grad(decoder.teacher_loss(prediction, correct), prediction)
    assert gradient[:, 2:, :216].abs().sum() > 0
    assert (gradient[..., 216:] == 0).all()


@pytest.mark.parametrize('diff_penalty_scale', [0., .01])
def test_hoi_dno_optimizer_resume_reproduces_uninterrupted_edit(tmp_path, monkeypatch, diff_penalty_scale):
    import pytest
    import mixer.hoi_diffusion_noise as module
    monkeypatch.setattr(module.torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(module.torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(module.torch.cuda, 'get_rng_state', lambda *a: torch.get_rng_state())
    monkeypatch.setattr(module.torch.cuda, 'set_rng_state', lambda rng, *a: torch.set_rng_state(rng))
    class Decoder:
        ordinal = 1
        calls = 0
        interrupt = False
        def decode(self, value):
            self.calls += 1
            if self.interrupt and self.calls == 3:
                raise RuntimeError('simulated interruption after durable checkpoint')
            return value
        def native(self, value):
            return (value,)
    class Objective:
        def __call__(self, value):
            loss = (value-.2).square().mean()
            self.value = float(loss.detach())
            return loss
        def term_record(self):
            return dict(body=self.value)
    initial = torch.tensor([[[[.8, 1.1, .4, -.2]]]])
    settings = dict(editing_steps=6, checkpoint_every=2, lr=.05, warmup=0, diff_penalty_scale=diff_penalty_scale, hsi_weight=.25)
    def destination(name):
        path = tmp_path/name; path.mkdir()
        return dict(repository='/data/yujinlun/Diffusion-Noise-Optimization', path=path, commit='fixture', memory_limit=8.)
    complete = module.optimize_hoi_latent(Decoder(), Objective(), initial, None, settings, destination('complete'), 'G')
    dest = destination('interrupted'); decoder = Decoder(); decoder.interrupt = True
    with pytest.raises(RuntimeError, match='simulated interruption'):
        module.optimize_hoi_latent(decoder, Objective(), initial, None, settings, dest, 'G')
    assert (dest['path']/'G-step0002.pt').exists()
    resumed = module.optimize_hoi_latent(Decoder(), Objective(), initial, None, settings, dest, 'G', resume=True)
    torch.testing.assert_close(resumed, complete, rtol=0, atol=0)
    checkpoint = torch.load(dest['path']/'G-step0006.pt', weights_only=False)
    assert len(checkpoint['traces']) == len(checkpoint['history']) == 6


def test_hoi_summary_includes_paired_task_completion_uncertainty(tmp_path):
    import json
    from mixer.hoi_diffusion_noise import summarize_hoi_dno
    baseline = dict(completed=True, contact_percent=.8, source_floor_support_fraction=.8,
        foot_sliding=.1, active_hand_samples=20, source_hand_contact_retention=1.,
        active_hand_mean_drift_cm=0., root_local_body_mean_drift_cm=0.,
        nonroot_rotation_mean_change_deg=0., seam_speed_mean_cm_s=10.,
        trajectory_acceleration_mean_cm_s2=20., initial_body_max_error_m=0.,
        initial_object_max_error_m=0., initial_object_rotation_max_error=0.,
        scene_human_penetration_s_mean=1., scene_obj_penetration_s_mean=1.)
    for task in range(2):
        folder = tmp_path/'lanes'/'lane-00'/f'task-{task:03d}'
        folder.mkdir(parents=True)
        means = {arm:dict(baseline) for arm in ('DDPM_reference', 'source', 'G', 'C', 'W')}
        means['C']['completed'] = task == 1
        (folder/'metrics.json').write_text(json.dumps(dict(task=task, scene=f'scene{task}',
            means=means, windows=1, native_frames=48, hoi_calls=2, hsi_calls=2,
            seconds=1., peak_memory_gib=.1, source_replay_exact=True)))
    manifest = tmp_path/'tasks.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(canonical_ordinal=i) for i in range(2)])))
    result = summarize_hoi_dno(tmp_path, manifest, 'cpu')
    completed = result['contrasts']['C__minus__source']['task']['completed']
    assert completed == dict(delta=-.5, ci=[-1., 0.], n=2)
    assert result['aggregate_protection']['completed_vs_source'] is False


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


def test_manipulation_target_fills_contact_gaps_and_preserves_initial_history():
    from mixer.hoi_diffusion_noise import manipulation_contact_mask
    distance = torch.tensor([[.09, .1], [.09, .1], [.04, .1], [.04, .1],
                             [.08, .1], [.09, .04], [.08, .1], [.04, .1], [.09, .1]])
    position = torch.zeros(9, 3)
    mask, hand, interval = manipulation_contact_mask(distance, position, dict(object_motion_speed_m_s=.01))
    assert hand == 0 and interval == (2, 8)
    assert not mask[:3].any() and mask[3:8, 0].all() and not mask[8].any()
    assert mask[:, 1].nonzero().flatten().tolist() == [5]
    # Translation establishes manipulation even when the source never engages.
    position[:, 0] = torch.arange(9)*.01
    mask, hand, interval = manipulation_contact_mask(torch.ones(9, 2)*.2, position,
                                                     dict(object_motion_speed_m_s=.01))
    assert hand == 0 and interval == (1, 9) and mask[3:, 0].all()


def _metric_motion_fixture(monkeypatch):
    import json
    import utils
    from mixer.hoi_diffusion_noise import MetricMotionObjective
    length = 53
    offsets = torch.linspace(-.02, .02, 28, dtype=torch.double)[:, None].expand(-1, 3).clone()
    offsets[24] = torch.tensor([.12, .02, .01])
    offsets[26] = torch.tensor([-.14, .01, .02])
    def body(pose, translation, *args, **kwargs):
        joints = pose[:, :1]+translation[:, None]+offsets
        return joints, joints
    monkeypatch.setattr(utils, 'run_smplx_model', body)
    p = torch.zeros(length, 22, 3, dtype=torch.double)
    t = torch.zeros(length, 3, dtype=torch.double)
    t[:, 0] = torch.linspace(0, .1, length)
    op = torch.zeros_like(t); op[:, 0] = torch.linspace(0, .02, length)
    rotation = transforms.axis_angle_to_matrix(torch.ones(length, 3, dtype=torch.double)*.02)
    vertices, joints = body(p, t)
    source = dict(pose=p, translation=t, object_translation=op, object_rotation=rotation,
                  joints=joints, verts=vertices, betas=None, gender=None)
    protocol = json.loads((Path(__file__).resolve().parents[2]/
        'experiments/protocols/p2_hoi_dno_metrics_s42_20260911.json').read_text())['method']
    grid = torch.linspace(-1, 1, 9)[:, None, None].expand(9, 9, 9).clone()-.1
    obj = torch.tensor([[.01, .01, .01], [-.01, -.01, -.01], [.01, -.01, .01]], dtype=torch.double)
    objective = MetricMotionObjective(source, None, obj, grid,
        dict(centroid=[0., 0., 0.], extents=[2., 2., 2.]),
        dict(scene_name='fixture', test_idx=0, pelvis_goal=[.2, 0., .1], object_goal=[.1, .1, .1]),
        dict(feet_height=0., scene_human_penetration_s_mean=1., scene_obj_penetration_s_mean=1.),
        protocol['physical_scales'], protocol['metric_targets'], grid,
        dict(centroid=[0., 0., 0.], extents=[2., 2., 2.]))
    objective.reference_chunks[0, length] = (vertices, joints)
    return source, objective, body


def test_metric_objective_repairs_source_contact_slip_and_collision(monkeypatch):
    source, objective, body = _metric_motion_fixture(monkeypatch)
    values = [source[k].clone().requires_grad_(True) for k in
              ('pose', 'translation', 'object_translation', 'object_rotation')]
    for term in ('contact', 'stance_speed', 'human_object'):
        objective.weights = [float(name == term) for name in objective.term_names]
        loss = objective(*values)
        gradients = torch.autograd.grad(loss, values)
        assert float(loss) > 0
        assert sum(float(g.square().sum()) for g in gradients) > 0
        # A small actual motion step in the computed direction must repair it.
        scale = 1e-6/max(float(g.abs().max()) for g in gradients)
        stepped = [v.detach()-scale*g for v,g in zip(values, gradients)]
        assert float(objective(*stepped)) < float(loss)


def test_metric_objective_overlap_matches_full_derivatives(monkeypatch):
    source, objective, body = _metric_motion_fixture(monkeypatch)
    torch.manual_seed(42)
    values = [(source[k]+torch.randn_like(source[k])*.0003).requires_grad_(True) for k in
              ('pose', 'translation', 'object_translation', 'object_rotation')]
    objective.weights = [1.+i*.07 for i in range(len(objective.term_names))]
    actual = objective(*values)
    gradients = torch.autograd.grad(actual, values)
    p, t, op, rotation = values
    vertices, joints = body(p, t)
    terms = objective.chunk_terms(p, joints, vertices, op, rotation, 0, 0, len(p))
    expected = (terms*terms.new_tensor(objective.weights)).sum()
    expected_gradients = torch.autograd.grad(expected, values)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-7)
    for a,b in zip(gradients, expected_gradients):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_metric_object_sdf_matches_native_frame_and_value(monkeypatch):
    import numpy as np
    from eval_metrics import compute_collision
    from utils import yup_to_zup, yup_to_zup_rotation_matrix
    source, objective, _ = _metric_motion_fixture(monkeypatch)
    vertices = source['verts'].float().requires_grad_(True)
    position = source['object_translation'].float().requires_grad_(True)
    rotation = source['object_rotation'].float().requires_grad_(True)
    actual = (-objective.object_signed(vertices, position, rotation)).clamp_min(0).mean()*100
    expected, _ = compute_collision(yup_to_zup(vertices), objective.object_sdf[0].numpy(),
        dict(centroid=[0., 0., 0.], extents=[2., 2., 2.]),
        yup_to_zup_rotation_matrix(rotation), yup_to_zup(position))
    np.testing.assert_allclose(float(actual), expected, rtol=1e-7, atol=1e-7)
    gradients = torch.autograd.grad(actual, (vertices, position, rotation))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_metric_chain_summary_uses_matched_final_reference_and_separate_hsi_gate(tmp_path):
    import json
    from mixer.hoi_diffusion_noise import summarize_hoi_dno
    baseline = dict(completed=True, contact_percent=.6, source_floor_support_fraction=.8,
        foot_sliding=.2, active_hand_samples=20, source_hand_contact_retention=1.,
        active_hand_mean_drift_cm=2., root_local_body_mean_drift_cm=1.,
        nonroot_rotation_mean_change_deg=1., seam_speed_mean_cm_s=10.,
        trajectory_acceleration_mean_cm_s2=20., initial_body_max_error_m=0.,
        initial_object_max_error_m=0., initial_object_rotation_max_error=0.,
        scene_human_penetration_s_mean=1., scene_obj_penetration_s_mean=1.,
        scene_human_penetration_frame_ratio=.3, scene_obj_penetration_frame_ratio=.3,
        human_pen_loss_infbagel=10.)
    raw = ('DDPM_reference', 'source', 'G', 'C', 'W')
    for task in range(2):
        folder = tmp_path/'lanes'/'lane-00'/f'task-{task:03d}'
        folder.mkdir(parents=True)
        means = {a+stage:dict(baseline) for a in raw for stage in ('', '_relation', '_final')}
        # A misleading raw reference cannot grant progress: only matched final
        # outputs enter the registered metric decision.
        means['source']['foot_sliding'] = .1
        for a in ('G', 'C', 'W'):
            means[a+'_final'].update(contact_percent=.7, human_pen_loss_infbagel=8., foot_sliding=.17)
        post = {a:dict(relation_seconds_including_evaluation=1.,
                      terminal=dict(solver=dict(arm_seconds_including_evaluation=1.))) for a in raw}
        (folder/'metrics.json').write_text(json.dumps(dict(task=task, scene=f'scene{task}',
            object='fixture', means=means, postprocess=post, windows=1, native_frames=48,
            hoi_calls=2, hsi_calls=2, seconds=1., peak_memory_gib=.1, source_replay_exact=True)))
    manifest = tmp_path/'tasks.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(canonical_ordinal=i) for i in range(2)])))
    result = summarize_hoi_dno(tmp_path, manifest, 'cpu')
    assert result['utility'] and not result['hsi_scene_utility']
    assert result['final_protection_pass_counts']['C'] == 2
    assert 'hand_drift' not in result['final_aggregate_protection']
    difference = result['contrasts']['C_final__minus__source_final']['task']['foot_sliding']
    assert abs(difference['delta']+.03) < 1e-12
    assert result['contrasts']['C_final__minus__W_final']['task']['completed']['ci'] == [0., 0.]
