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
