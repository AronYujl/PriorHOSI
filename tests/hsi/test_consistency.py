"""Consistency conditioning, active supervision, and sampler geometry mapping."""

from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from models.infbagel import Sampler, Unet


def sampler(**kwargs):
    return Sampler(
        device="cpu", mask_ind=0, emb_f=0, batch_size=2, channel=232,
        auto_regre_num=2, timesteps=500, ddim_timesteps=25, cm_timesteps=16,
        **kwargs,
    )


class SceneFeatures(torch.nn.Module):
    def forward(self, value):
        return value.clone()


def model_inputs():
    torch.manual_seed(42)
    batch = 3
    return dict(
        x=torch.randn(batch, 16, 232), cond=torch.randn(batch, 32),
        timesteps=torch.tensor([499, 259, 19]), text_emb=torch.zeros(batch, 768),
        pelvis_goal=torch.zeros(batch, 3), scene_goal=torch.zeros(batch, 3),
        is_loco=torch.zeros(batch, dtype=torch.bool), need_scene=torch.ones(batch, dtype=torch.bool),
        need_pelvis_dir=torch.ones(batch, dtype=torch.bool), pi=torch.zeros(batch),
        end_pi=torch.ones(batch), seq_length=torch.ones(batch),
        need_pi=torch.zeros(batch, dtype=torch.bool), object_goal=torch.zeros(batch, 3),
        is_object=torch.zeros(batch, dtype=torch.bool), obj_bps_data=torch.zeros(batch, 1024, 3),
        occ_list=torch.randn(4 * batch, 32), occ_pos=torch.zeros(4, batch, 2),
    )


def select_rows(inputs, rows):
    result = {key: value[rows] for key, value in inputs.items()
              if key not in ("occ_list", "occ_pos")}
    result["occ_list"] = inputs["occ_list"].reshape(4, 3, 32)[:, rows].reshape(-1, 32)
    result["occ_pos"] = inputs["occ_pos"][:, rows]
    return result


@pytest.mark.parametrize("student", [False, True])
def test_conditioning_is_independent_of_batch_order_and_partition(student):
    model = Unet(
        dim_model=32, num_heads=4, num_layers=1, dropout_p=0,
        dim_input=232, dim_output=232, scene_type="occ_temp", nb_voxels=(32, 32, 32),
        load_scene=False, load_language=False, load_scene_goal=False,
        load_pelvis_goal=False, load_object_goal=False,
    ).eval()
    # Keep the real token assembly/transformer, with already-encoded scene inputs.
    model.load_scene = True
    model.scene_embedding = SceneFeatures()
    inputs = model_inputs()
    if student:
        inputs["cfg_scale"] = torch.ones(3, 1)
    mode = dict(is_sample=not student)
    with torch.no_grad():
        together = model(**inputs, **mode)
        permuted = model(**select_rows(inputs, [2, 0, 1]), **mode)
        separate = torch.cat([model(**select_rows(inputs, [i]), **mode) for i in range(3)])
    torch.testing.assert_close(together[[2, 0, 1]], permuted, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)

    changed = {key: value.clone() for key, value in inputs.items()}
    changed["occ_list"][3:] += 10
    with torch.no_grad():
        shifted = model(**changed, **mode)
    torch.testing.assert_close(together[0], shifted[0], atol=0, rtol=0)
    assert (together[1:] - shifted[1:]).abs().max() > 1e-4

    if student:
        seen = []
        hook = model.cfg_scale_embedding.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].clone())
        )
        with torch.no_grad():
            model(**inputs)
        hook.remove()
        torch.testing.assert_close(seen[0], torch.ones(3, 1), atol=0, rtol=0)


class Prediction(torch.nn.Module):
    def __init__(self, teacher=False):
        super().__init__()
        self.value = torch.nn.Parameter(torch.full((232,), 0.1))
        self.teacher = teacher
        self.scales = []

    def forward(self, x, *args, **kwargs):
        if self.teacher:
            return torch.ones_like(x) * (1 if kwargs["is_uncondition"] else 3)
        self.scales.append(kwargs["cfg_scale"].clone())
        return self.value.expand_as(x)


def test_fixed_scale_target_and_hsi_supervision_gradients():
    engine = sampler(cm_fixed_cfg_scale=1)
    engine.dataset = SimpleNamespace(load_scene=True, use_object_keypoints=False)
    engine._compute_occ = lambda *args: (None, None, None)
    engine.student_model = Prediction()
    engine.target_model = Prediction().requires_grad_(False)
    engine.target_model.value.fill_(0.5)
    engine.teacher_model = Prediction(teacher=True).requires_grad_(False)
    x = torch.zeros(2, 16, 232)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :2] = True
    inputs = dict(
        x_start=x, joints=x[..., :84], mat=None, scene_flag=None, mask=mask, t=None,
        text_emb=None, pelvis_goal=None, scene_goal=None, object_goal=None,
        need_scene=None, need_pelvis_dir=None, pi=None, end_pi=None, seq_length=None,
        need_pi=None, is_loco=None, is_object=torch.tensor([False, True]),
        obj_bps_data=None, obj_rot_mat_ref=None, rest_pose_obj_nn_pts=None,
        transformed_obj_verts=None, rest_human_offsets=None, noise=torch.zeros_like(x),
    )
    with patch.object(engine.solver, "ddim_step", wraps=engine.solver.ddim_step) as solve:
        result = engine.consistency_loss(**inputs)
    torch.testing.assert_close(solve.call_args.args[0], torch.full_like(x, 5))
    for model in (engine.student_model, engine.target_model):
        torch.testing.assert_close(model.scales[0], torch.ones(2, 1))
    result["loss_consistency"].backward()
    assert engine.student_model.value.grad[:216].abs().min() > 0
    assert engine.student_model.value.grad[216:].abs().min() > 0  # object row is supervised
    engine.student_model.zero_grad()
    inputs["is_object"][:] = False
    engine.consistency_loss(**inputs)["loss_consistency"].backward()
    assert engine.student_model.value.grad[:216].abs().min() > 0
    assert engine.student_model.value.grad[216:].count_nonzero() == 0


class GeometryDataset:
    max_window_size = 16

    def denormalize_torch(self, points):
        return points

    def quat_ik_torch(self, rotations):
        return rotations

    def quat_fk_torch(self, rotations, offsets):
        return None, offsets[:, :1].expand(-1, 24, -1)

    def get_nearest_free_voxel(self, *args):
        raise AssertionError("analytic energy replaces voxel lookup")


@pytest.mark.parametrize("solver_index", [0, 10, 24])
def test_cm_guidance_matches_clean_displacement_through_actual_renoising(solver_index):
    engine = sampler()
    engine.batch_size = 1
    engine.dataset = GeometryDataset()
    engine._compute_occ_sample = lambda *args: (None, None, None)
    engine.set_fixed_points = lambda *args, **kwargs: None
    prediction = torch.zeros(1, 16, 232)
    prediction[..., 0] = 0.5
    prediction[..., 84:216] = torch.tensor([1, 0, 0, 0, 1, 0]).repeat(22)
    model = lambda *args, **kwargs: prediction.clone()
    engine.student_model = model
    args = dict(
        model=model, x0=prediction, x=prediction, fixed_points=None,
        mat=torch.eye(4)[None], scene_flag=torch.zeros(1, dtype=torch.long),
        t=torch.tensor([solver_index]), t_index=1, text_emb=None,
        pelvis_goal=None, scene_goal=None, object_goal=None, need_scene=None,
        need_pelvis_dir=None, pi=None, end_pi=None, seq_length=None, need_pi=None,
        is_loco=None, is_object=torch.tensor([False]), obj_bps_data=None,
        object_points=None, obj_rot_mat_ref=None, obj_rest_verts=None,
        obj_vert_normals=None, seq_name_dict=None, obj_rot_mat_prefix=None,
        human_dict=dict(rest_human_offsets=torch.zeros(1, 16, 24, 3),
                        transl=None, betas=None, gender=None),
        guidance_fn=None, guidance_scale=1.0, w=1,
    )
    with patch("models.infbagel.apply_hsi_guidance_loss", side_effect=lambda j, *a: j.square().sum()):
        torch.manual_seed(42)
        uncorrected, _, clean = engine.cm_sample(**args)
        args["guidance_fn"] = True
        torch.manual_seed(42)
        legacy, _, _ = engine.cm_sample(**args)
        engine.hsi_cm_guidance_x0_coef = True
        torch.manual_seed(42)
        corrected, _, _ = engine.cm_sample(**args)
    # The solver independently supplies the Jacobian for the same phase index.
    mapped, _ = engine.solver.ddim_style_multiphase_pred(
        clean + (legacy - uncorrected), torch.zeros_like(clean), args["t"], 16
    )
    base, _ = engine.solver.ddim_style_multiphase_pred(
        clean, torch.zeros_like(clean), args["t"], 16
    )
    assert (legacy - uncorrected).abs().max() > 0
    torch.testing.assert_close((corrected - uncorrected).double(), mapped - base, atol=1e-6, rtol=1e-5)


class ConstantTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        value = torch.zeros(232)
        value[84:216] = torch.tensor([1, 0, 0, 0, 1, 0]).repeat(22)
        self.value = torch.nn.Parameter(value)
        self.calls = []

    def forward(self, x, occ, t, *args, **kwargs):
        unconditional = kwargs.get("is_uncondition", False)
        self.calls.append((t.clone(), x[:, :2].clone(), unconditional))
        result = self.value.expand_as(x).clone()
        result[..., 0] = 0.25 if unconditional else 0.5
        return result


def diffusion_inputs():
    engine = sampler(w=1)
    engine.batch_size = 1
    engine.dataset = GeometryDataset()
    engine._compute_occ_sample = lambda *args: (None, None, None)
    engine.student_model = ConstantTeacher().eval()
    x = torch.randn(1, 16, 232)
    args = dict(
        fixed_points=x[:, :2].clone(), mat=torch.eye(4)[None],
        scene_flag=torch.zeros(1, dtype=torch.long), text_emb=None,
        pelvis_goal=None, scene_goal=None, object_goal=None, need_scene=None,
        need_pelvis_dir=None, pi=None, end_pi=None, seq_length=None, need_pi=None,
        is_loco=None, is_object=torch.tensor([False]), obj_bps_data=None,
        object_points=None, obj_rot_mat_ref=None, obj_rest_verts=None,
        obj_vert_normals=None, seq_name_dict=None,
        human_dict=dict(rest_human_offsets=torch.zeros(1, 16, 24, 3),
                        transl=None, betas=None, gender=None),
        guidance_fn=None, guidance_scale=1.0,
    )
    return engine, x, args


def test_ddim_rollout_visits_teacher_grid_preserves_history_and_only_draws_initial_noise():
    engine, _, args = diffusion_inputs()
    torch.manual_seed(42)
    torch.randn(1, 16, 232)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    samples, _ = engine.p_sample_loop(**args, use_ddim=True)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(samples) == 25 and len(engine.student_model.calls) == 50
    assert [int(t[0]) for t, _, _ in engine.student_model.calls[::2]] == list(range(499, 18, -20))
    assert [flag for _, _, flag in engine.student_model.calls] == [False, True] * 25
    for _, prefix, _ in engine.student_model.calls:
        torch.testing.assert_close(prefix, args["fixed_points"], rtol=0, atol=0)
    expected = engine.student_model.value.expand_as(samples[-1]).clone()
    expected[..., 0] = 0.75  # conditional + 1 * (conditional - unconditional)
    expected[:, :2] = args["fixed_points"]
    torch.testing.assert_close(samples[-1], expected, rtol=0, atol=0)


def test_remaining_teacher_endpoint_matches_native_suffix_and_preserves_input():
    import inspect
    from priors.hsi.diagnostics import remaining_teacher_endpoint
    engine, _, args = diffusion_inputs()
    original = engine.p_sample
    signature = inspect.signature(original)
    captured = {}
    occupancy_sources = []

    def observe(*values, **kwargs):
        bound = signature.bind(*values, **kwargs)
        bound.apply_defaults()
        if bound.arguments["t_index"] == 59:
            captured.update({k: v.clone() if torch.is_tensor(v) else v
                             for k, v in bound.arguments.items()})
        return original(*values, **kwargs)

    engine.p_sample = observe
    torch.manual_seed(42)
    samples, _ = engine.p_sample_loop(**args, use_ddim=True)
    engine.p_sample = original
    before = captured["x"].clone()
    engine._compute_occ_sample = lambda x, x0, *rest: (occupancy_sources.append(x0.clone()) or (None, None, None))
    local, final = remaining_teacher_endpoint(engine, engine.student_model, captured)
    torch.testing.assert_close(final, samples[-1], rtol=0, atol=0)
    torch.testing.assert_close(captured["x"], before, rtol=0, atol=0)
    torch.testing.assert_close(local[:, :2], args["fixed_points"], rtol=0, atol=0)
    assert len(occupancy_sources) == 3
    torch.testing.assert_close(occupancy_sources[0], captured["x0"])
    # The next crop follows the preceding raw teacher prediction, not x_t.
    torch.testing.assert_close(occupancy_sources[1][:, 2:], local[:, 2:])


@pytest.mark.parametrize("source", ["ddim", "consistency"])
def test_alignment_observation_preserves_native_trajectory_rng_and_common_timesteps(source):
    import numpy as np
    from priors.hsi.diagnostics import DistillationAlignmentProbe
    engine, _, args = diffusion_inputs()
    run = lambda: (engine.cm_sample_loop(**args, w=1) if source == "consistency"
                   else engine.p_sample_loop(**args, use_ddim=True))
    torch.manual_seed(42)
    baseline, _ = run()
    baseline_rng = torch.get_rng_state()
    probe = object.__new__(DistillationAlignmentProbe)
    probe.cfg = SimpleNamespace(sample_type=source)
    probe.timesteps = (499, 279, 59)
    seen = []

    def consume_randomness(state):
        seen.append(state["t_index"])
        torch.randn(13)
        np.random.rand(7)

    probe.observe = consume_randomness
    probe.attach(engine)
    np.random.seed(42)
    numpy_before = np.random.get_state()
    torch.manual_seed(42)
    observed, _ = run()
    assert seen == [499, 279, 59]
    for a, b in zip(baseline, observed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])


def test_alignment_metrics_separate_root_body_rotation_and_boundary_derivatives():
    from priors.hsi.diagnostics import alignment_endpoint_metrics
    arms = ("teacher_local", "teacher_endpoint", "student")
    positions = {key: torch.zeros(1, 16, 28, 3) for key in arms}
    joints = {key: torch.zeros(1, 16, 24, 3) for key in arms}
    rotations = {key: torch.eye(3).expand(1, 16, 22, 3, 3).clone() for key in arms}
    # A future-only 10 cm root translation leaves body-relative geometry intact.
    positions["student"][:, 2:, :, 0] = 0.1
    joints["student"][:, 2:, :, 0] = 0.1
    rotations["student"][:, 2:] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    metrics = alignment_endpoint_metrics(positions, joints, rotations)
    for key, expected in dict(student_body_cm=0, student_direct_body_cm=0,
                              student_root_cm=10, student_global_fk_cm=10,
                              student_boundary_velocity=1, student_boundary_acceleration=10,
                              student_boundary_jerk=200, student_rotation_deg=90).items():
        torch.testing.assert_close(metrics[key], torch.tensor([float(expected)]))
    assert all(value.item() == 0 for key, value in metrics.items() if key.startswith("teacher_local"))
    joints["student"][:, 2:, 1:22, 1] += 0.02
    metrics = alignment_endpoint_metrics(positions, joints, rotations)
    torch.testing.assert_close(metrics["student_body_cm"], torch.tensor([2.]))
    torch.testing.assert_close(metrics["student_self_body_cm"], torch.tensor([2.]))


def test_cm25_rollout_visits_all_solver_steps_and_preserves_terminal_history():
    engine, _, args = diffusion_inputs()
    engine.cm_timesteps = 25
    engine.hsi_cm_guidance_x0_coef = True
    with patch.object(engine.solver, "ddim_style_multiphase_pred",
                      wraps=engine.solver.ddim_style_multiphase_pred) as solve:
        samples, _ = engine.cm_sample_loop(**args, w=1)
    assert len(samples) == len(engine.student_model.calls) == 25
    assert [int(t[0]) for t, _, _ in engine.student_model.calls] == list(range(499, 18, -20))
    for _, prefix, _ in engine.student_model.calls:
        torch.testing.assert_close(prefix, args["fixed_points"], rtol=0, atol=0)
    assert solve.call_count == 1
    assert solve.call_args.args[2].item() == 0
    expected = solve.call_args.args[0].clone()
    expected[:, :2] = args["fixed_points"]
    torch.testing.assert_close(samples[-1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("solver_index", [0, 10, 24])
def test_ddim_guidance_is_the_clean_jacobian_of_the_deterministic_teacher_step(solver_index):
    engine, x, args = diffusion_inputs()
    t = int(engine.solver.ddim_timesteps[solver_index])
    args.update(model=engine.student_model, x0=x, x=x, t=torch.tensor([t]),
                t_index=t, ddim_index=solver_index)
    base, _, clean = engine.p_sample(**args)
    engine._hsi_guidance_loss = lambda joints, scene: joints.square().sum()
    args["guidance_fn"] = True
    guided, _, _ = engine.p_sample(**args)
    displacement = torch.zeros_like(clean)
    displacement[..., :3] = -48 * clean[..., :3]  # 24 joints at the pelvis in this fixture
    alpha = engine.alpha_cumprod[t]
    previous = engine.alpha_cumprod[t - 20] if solver_index else torch.tensor(1.)
    def reference(prediction):
        noise = (x - alpha.sqrt() * prediction) / (1 - alpha).sqrt()
        return previous.sqrt() * prediction + (1 - previous).sqrt() * noise
    expected = reference(clean + displacement) - reference(clean)
    if solver_index == 0:
        expected.zero_()  # the native final denoising transition has no external correction
    torch.testing.assert_close(guided - base, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        engine.solver.ddim_x0_coefficient(torch.tensor([solver_index]), x.shape).float().squeeze(),
        previous.sqrt() - (1 - previous).sqrt() * alpha.sqrt() / (1 - alpha).sqrt(),
        atol=1e-7, rtol=1e-5,
    )


@pytest.mark.parametrize("timestep", [0, 19, 249, 499])
def test_ddpm_default_step_preserves_posterior_and_random_draw(timestep):
    engine, x, args = diffusion_inputs()
    args.update(model=engine.student_model, x0=x, x=x, t=torch.tensor([timestep]),
                t_index=timestep)
    torch.manual_seed(42)
    actual, _, clean = engine.p_sample(**args)
    actual_rng = torch.get_rng_state()
    expected = engine.posterior_mean_coef1[timestep] * clean + engine.posterior_mean_coef2[timestep] * x
    torch.manual_seed(42)
    if timestep:
        expected += (0.5 * engine.posterior_log_variance_clipped[timestep]).exp() * torch.randn_like(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng, torch.get_rng_state())


def test_endpoint_crop_selection_preserves_temporal_blocks_and_row_order():
    from priors.hsi.distillation import select_occupancy_rows
    occ = torch.arange(5.)[:, None]
    crops = torch.arange(20.).reshape(20, 1)
    positions = torch.arange(40.).reshape(4, 5, 2)
    rows = torch.tensor([4, 1])
    a, b, c = select_occupancy_rows((occ, crops, positions), rows, 5)
    torch.testing.assert_close(a, occ[rows])
    torch.testing.assert_close(b.flatten(), torch.tensor([4., 1., 9., 6., 14., 11., 19., 16.]))
    torch.testing.assert_close(c, positions[:, rows])


@pytest.mark.parametrize('start_index', [0, 1, 2])
def test_low_noise_endpoint_matches_production_ddim_and_stops_teacher_gradient(start_index):
    from priors.hsi.distillation import BodyEndpointObjective
    engine, x, args = diffusion_inputs()

    class StateDependentTeacher(ConstantTeacher):
        def forward(self, x, *a, **kw):
            return super().forward(x, *a, **kw) + 0.05 * x

    teacher = StateDependentTeacher().eval()
    engine.student_model = teacher
    objective = BodyEndpointObjective(engine, teacher)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :2] = True
    clean = x.clone()
    target = objective.teacher_endpoint(x, clean, mask, start_index, (None, None, None), args, torch.ones(1, 1))
    state, previous = x.clone(), x.clone()
    for index in range(start_index, -1, -1):
        t = int(engine.solver.ddim_timesteps[index])
        state, _, previous = engine.p_sample(teacher, previous, state, t=torch.tensor([t]),
                                            t_index=t, ddim_index=index, **args)
        state[:, :2] = clean[:, :2]
    torch.testing.assert_close(target, state, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(target[:, :2], clean[:, :2], rtol=0, atol=0)
    assert target.requires_grad is False and teacher.value.grad is None
    assert [int(t[0]) for t, _, _ in teacher.calls[:2 * (start_index + 1)]] == [
        t for t in range(start_index * 20 + 19, 18, -20) for _ in range(2)]


def endpoint_objective_fixture(batch=4):
    from datasets.infbagel import InfBaGelDataset
    from priors.hsi.distillation import BodyEndpointObjective

    class BodyDataset(GeometryDataset):
        parents_22 = [-1] + [0] * 21
        parents_24 = [-1] + [0] * 23
        quat_ik_torch = InfBaGelDataset.quat_ik_torch
        quat_fk_torch = InfBaGelDataset.quat_fk_torch

    engine = sampler(w=1)
    engine.dataset = BodyDataset()
    engine._compute_occ_sample = lambda *a, **kw: (None, None, None)
    teacher = ConstantTeacher().eval().requires_grad_(False)
    engine.student_model = teacher
    prediction = teacher.value.detach().expand(batch, 16, 232).clone()
    prediction[..., 0] = .75
    offsets = torch.zeros(batch, 24, 3)
    offsets[:, 1:, 0] = .2
    inputs = {key: torch.zeros(batch, 1) for key in (
        'text_emb', 'pelvis_goal', 'scene_goal', 'object_goal', 'is_loco', 'need_scene',
        'need_pelvis_dir', 'pi', 'end_pi', 'seq_length', 'need_pi', 'obj_bps_data',
        'object_points', 'obj_rot_mat_ref', 'scene_flag')}
    inputs.update(mat=torch.eye(4).expand(batch, 4, 4).clone(),
                  rest_offsets=offsets, is_object=torch.zeros(batch, dtype=torch.bool))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[:, :2] = True
    occupancy = (torch.zeros(batch, 1), torch.zeros(4 * batch, 1), torch.zeros(4, batch, 2))
    return BodyEndpointObjective(engine, teacher), prediction, mask, occupancy, inputs


def test_endpoint_objective_full_batch_normalization_root_pose_and_hsi_gradient_scope():
    objective, prediction, mask, occupancy, inputs = endpoint_objective_fixture()
    clean = prediction.clone()
    inputs['is_object'][2] = True
    prediction[:, 2:, 0] += .1
    prediction.requires_grad_()
    indices = torch.tensor([0, 1, 2, 24])
    loss, counts = objective(prediction, clean, clean, mask, indices, occupancy, inputs, torch.ones(4, 1))
    # Two eligible rows, ten cm in x at all22 joints: (0.1²/3)*(2/4).
    torch.testing.assert_close(loss, torch.tensor(.01 / 6))
    assert counts.tolist() == [1, 1, 0]
    loss.backward()
    assert prediction.grad[:2, 2:, 0].abs().min() > 0
    assert prediction.grad[:, :2].count_nonzero() == 0
    assert prediction.grad[2:].count_nonzero() == 0
    assert prediction.grad[..., 216:].count_nonzero() == 0
    assert objective.teacher.value.grad is None

    # A changed root orientation moves the articulated body, independently of
    # translation. The absolute geometry target must supervise that output too.
    posed = clean.clone()
    posed[0, 2:, 85] = .1
    posed.requires_grad_()
    pose_loss, _ = objective(posed, clean, clean, mask, indices, occupancy, inputs, torch.ones(4, 1))
    pose_loss.backward()
    assert pose_loss > 0 and posed.grad[0, 2:, 84:90].abs().sum() > 0
    high = clean.clone().requires_grad_()
    zero, counts = objective(high, clean, clean, mask, torch.full((4,), 24), occupancy, inputs, torch.ones(4, 1))
    zero.backward()
    assert zero.item() == 0 and high.grad.count_nonzero() == 0 and counts.sum() == 0


def test_endpoint_addition_preserves_original_consistency_losses_teacher_modes_and_rng():
    import copy
    from priors.hsi.distillation import BodyEndpointObjective
    engine = sampler(cm_fixed_cfg_scale=1, w=1)
    batch = 2
    engine.dataset = GeometryDataset()
    engine.dataset.load_scene = True
    engine.dataset.use_object_keypoints = False
    occupancy = (torch.zeros(batch, 1), torch.zeros(4 * batch, 1), torch.zeros(4, batch, 2))
    engine._compute_occ = lambda *a: occupancy
    engine.student_model = Prediction()
    engine.teacher_model = Prediction(teacher=True).requires_grad_(False).train()
    engine.target_model = Prediction().requires_grad_(False).train()
    engine.target_model.value.fill_(.5)
    enabled = copy.deepcopy(engine)
    auxiliary = sampler(w=1)
    auxiliary.dataset = engine.dataset
    auxiliary._compute_occ_sample = lambda *a: (None, None, None)
    teacher = ConstantTeacher().eval().requires_grad_(False)
    enabled.body_endpoint_objective = BodyEndpointObjective(auxiliary, teacher)
    native = enabled.body_endpoint_objective.teacher_endpoint

    def draws_randomness(*a, **kw):
        torch.rand(17)
        return native(*a, **kw)

    enabled.body_endpoint_objective.teacher_endpoint = draws_randomness
    x = torch.zeros(batch, 16, 232)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :2] = True
    args = {key: torch.zeros(batch, 1) for key in (
        'scene_flag', 'text_emb', 'pelvis_goal', 'scene_goal', 'object_goal', 'need_scene',
        'need_pelvis_dir', 'pi', 'end_pi', 'seq_length', 'need_pi', 'is_loco',
        'obj_bps_data', 'obj_rot_mat_ref', 'rest_pose_obj_nn_pts', 'transformed_obj_verts', 'object_points')}
    args.update(x_start=x, joints=x[..., :84], mask=mask, mat=torch.eye(4).expand(batch, 4, 4),
                t=None, is_object=torch.zeros(batch, dtype=torch.bool), rest_human_offsets=torch.zeros(batch, 24, 3))
    with patch('torch.randint', return_value=torch.tensor([0, 1])):
        torch.manual_seed(42)
        baseline = engine.consistency_loss(**args)
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(42)
        result = enabled.consistency_loss(**args)
    torch.testing.assert_close(result['loss_consistency'], baseline['loss_consistency'], rtol=0, atol=0)
    assert result['loss_fk'] is baseline['loss_fk'] is None
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert enabled.teacher_model.training and enabled.target_model.training and not teacher.training
    assert result['loss_body_endpoint'] > 0


def test_endpoint_calibration_uses_medians_and_both_registered_gradient_limits():
    from priors.hsi.diagnostics import endpoint_fk_calibrated_weight
    records = [dict(cm1_total=dict(trunk_gradient_norm=10.),
                    consistency=dict(rotation_head_gradient_norm=2.),
                    endpoint=dict(trunk_gradient_norm=2., rotation_head_gradient_norm=4.)) for _ in range(8)]
    result = endpoint_fk_calibrated_weight(records)
    assert result['trunk_10_percent_weight'] == .5
    assert result['rotation_head_25_percent_weight'] == .125
    assert result['cm_endpoint_loss_weight'] == .125
