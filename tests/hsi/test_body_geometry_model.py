"""Full-diffusion integration of candidate-body geometry and learned correction."""

from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from models.infbagel import Sampler, Unet


class _SceneFeatures(torch.nn.Module):
    def forward(self, value):
        return value.clone()


def _tiny_model(enabled):
    model = Unet(
        dim_model=32, num_heads=4, num_layers=1, dropout_p=0,
        dim_input=232, dim_output=232, scene_type="occ_temp", nb_voxels=(32, 32, 32),
        load_scene=False, load_language=False, load_scene_goal=False,
        load_pelvis_goal=False, load_object_goal=False,
        body_geometry_enabled=enabled,
    ).eval()
    model.load_scene = True
    model.scene_embedding = _SceneFeatures()
    return model


def _inputs():
    generator = torch.Generator().manual_seed(42)
    batch = 3
    return dict(
        x=torch.randn(batch, 16, 232, generator=generator),
        cond=torch.randn(batch, 32, generator=generator),
        timesteps=torch.full((batch,), 259), text_emb=torch.zeros(batch, 768),
        pelvis_goal=torch.zeros(batch, 3), scene_goal=torch.zeros(batch, 3),
        is_loco=torch.zeros(batch, dtype=torch.bool),
        need_scene=torch.ones(batch, dtype=torch.bool),
        need_pelvis_dir=torch.ones(batch, dtype=torch.bool), pi=torch.zeros(batch),
        end_pi=torch.ones(batch), seq_length=torch.ones(batch),
        need_pi=torch.zeros(batch, dtype=torch.bool), object_goal=torch.zeros(batch, 3),
        is_object=torch.zeros(batch, dtype=torch.bool),
        obj_bps_data=torch.zeros(batch, 1024, 3),
        occ_list=torch.randn(4 * batch, 32, generator=generator),
        occ_pos=torch.zeros(4, batch, 2),
    )


def _features(value):
    return lambda clean: clean.new_full((*clean.shape[:2], 24, 5), value)


def _enable_geometry_response(model):
    # One positive distance feature drives a residual in every human channel.
    with torch.no_grad():
        refiner = model.body_geometry_refiner
        refiner.input.weight.zero_()
        refiner.input.bias.zero_()
        refiner.input.weight[0, model.dim_model] = 1.0
        refiner.output.weight.zero_()
        refiner.output.bias.zero_()
        refiner.output.weight[:, 0] = 0.1


def test_branch_initialization_preserves_existing_weights_rng_and_eval_prediction():
    torch.manual_seed(42)
    original = _tiny_model(False)
    original_rng = torch.get_rng_state()
    torch.manual_seed(42)
    enhanced = _tiny_model(True)
    assert torch.equal(torch.get_rng_state(), original_rng)
    old_state, new_state = original.state_dict(), enhanced.state_dict()
    assert old_state.keys() == {
        key for key in new_state if not key.startswith("body_geometry_refiner.")
    }
    for key, value in old_state.items():
        assert torch.equal(value, new_state[key]), key
    inputs = _inputs()
    with torch.no_grad():
        expected = original(**inputs, is_sample=True)
        expected_rng = torch.get_rng_state()
        actual = enhanced(**inputs, is_sample=True, body_geometry_query=_features(1.0))
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_nonzero_refiner_uses_query_and_changes_human_channels_only():
    model = _tiny_model(True)
    _enable_geometry_response(model)
    inputs = _inputs()
    seen = []

    def query(clean):
        seen.append(clean.clone())
        return _features(1.0)(clean)

    with torch.no_grad():
        base = model(**inputs, is_sample=True, body_geometry_query=_features(0.0))
        corrected = model(**inputs, is_sample=True, body_geometry_query=query)
    assert len(seen) == 1
    assert torch.equal(seen[0], base)
    assert (corrected[..., :216] - base[..., :216]).min() > 0.07
    assert torch.equal(corrected[..., 216:], base[..., 216:])


@pytest.mark.parametrize("mask_kind", ["need_scene", "unconditional", "initial_noise", "cfg_null"])
def test_body_branch_follows_scene_and_sampling_condition_masks(mask_kind):
    model = _tiny_model(True)
    _enable_geometry_response(model)
    inputs, options = _inputs(), dict(is_sample=True)
    active = torch.tensor([False, True, False])
    if mask_kind == "need_scene":
        inputs["need_scene"] = active.clone()
    elif mask_kind == "unconditional":
        options["is_uncondition"] = True
        active[:] = False
    elif mask_kind == "initial_noise":
        inputs["timesteps"] = torch.tensor([499, 259, 499])
    else:
        options["cfg_scale"] = torch.tensor([[-1.0], [1.0], [-1.0]])
    with torch.no_grad():
        base = model(**inputs, **options, body_geometry_query=_features(0.0))
        corrected = model(**inputs, **options, body_geometry_query=_features(1.0))
    assert torch.equal(corrected[~active], base[~active])
    if active.any():
        assert (corrected[active, :, :216] - base[active, :, :216]).min() > 0.07
    assert torch.equal(corrected[..., 216:], base[..., 216:])


def test_training_geometry_reuses_the_temporal_scene_dropout_draw():
    model = _tiny_model(True).train()
    _enable_geometry_response(model)
    inputs = _inputs()
    # The existing temporal-crop drop probability is 0.1: rows 0 and 2 drop.
    draw = torch.tensor([0.05, 0.5, 0.05]).reshape(3, 1, 1)
    outputs = []
    for value in (0.0, 1.0):
        with patch("models.infbagel.torch.rand", return_value=draw) as random_draw:
            outputs.append(model(**inputs, body_geometry_query=_features(value)))
        assert random_draw.call_count == 1
    base, corrected = outputs
    assert torch.equal(corrected[[0, 2]], base[[0, 2]])
    assert (corrected[1, :, :216] - base[1, :, :216]).min() > 0.07
    assert torch.equal(corrected[..., 216:], base[..., 216:])


def _sampler():
    return Sampler(
        device="cpu", mask_ind=0, emb_f=0, batch_size=1, channel=232,
        auto_regre_num=2, timesteps=500, ddim_timesteps=25, cm_timesteps=16,
        body_geometry_enabled=True, w=1,
    )


class _PlaneField:
    voxel_size = 0.02

    def signed_distance(self, points, scene_flag):
        return points[..., 1], torch.zeros(points.shape[:-1], dtype=torch.bool)


def test_predict_clean_queries_coarse_prediction_with_known_history_and_input_gradients():
    engine = _sampler()
    engine._get_pen_sdf_bank = lambda: _PlaneField()
    candidates = []

    def reconstruct(candidate, joints, mat, offsets):
        candidates.append(candidate.clone())
        direct = candidate[..., :84].reshape(1, 16, 28, 3)
        return direct, direct[..., :24, :]

    engine._compute_human_joints = reconstruct

    def model(x, occ, t, *, body_geometry_query):
        coarse = 2.0 * x + 0.05
        features = body_geometry_query(coarse)
        # Read only geometry, so the tested derivative must traverse its query.
        human = features[..., 0].mean(-1, keepdim=True).expand(-1, -1, 216)
        return torch.cat((human, coarse[..., 216:]), dim=-1)

    x = torch.full((1, 16, 232), 0.01, requires_grad=True)
    result = engine.predict_clean(
        model, x, None, torch.tensor([259]), mat=torch.eye(4)[None],
        scene_flag=torch.zeros(1, dtype=torch.long),
        rest_human_offsets=torch.zeros(1, 24, 3),
    )
    assert len(candidates) == 1
    assert torch.equal(candidates[0][:, :2], x[:, :2])
    assert torch.equal(candidates[0][:, 2:], 2.0 * x[:, 2:] + 0.05)
    gradient = torch.autograd.grad(result[..., 0].sum(), x)[0]
    expected = torch.zeros_like(x)
    expected[:, :2, 1:72:3] = 4.0 / 24
    expected[:, 2:, 1:72:3] = 8.0 / 24
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("ddim_index", [None, 12])
def test_ddpm_and_ddim_route_both_cfg_queries_through_predict_clean(ddim_index):
    engine = _sampler()
    engine._compute_occ_sample = lambda *args: (None, None, None)
    x = torch.zeros(1, 16, 232)
    offsets = torch.randn(1, 16, 24, 3)

    def direct_model(*args, **kwargs):
        raise AssertionError("the denoising path must use predict_clean")

    def prediction(model, noisy, occ, t, *args, **kwargs):
        assert model is direct_model
        assert torch.equal(kwargs["rest_human_offsets"], offsets[:, 0])
        value = 1.0 if kwargs.get("is_uncondition", False) else 2.0
        return torch.full_like(noisy, value)

    kwargs = dict(
        model=direct_model, x0=x, x=x, fixed_points=None, mat=torch.eye(4)[None],
        scene_flag=torch.zeros(1, dtype=torch.long), t=torch.tensor([259]), t_index=259,
        text_emb=None, pelvis_goal=None, scene_goal=None, object_goal=None, need_scene=None,
        need_pelvis_dir=None, pi=None, end_pi=None, seq_length=None, need_pi=None,
        is_loco=None, is_object=torch.tensor([False]), obj_bps_data=None,
        object_points=None, obj_rot_mat_ref=None, obj_rest_verts=None,
        obj_vert_normals=None, seq_name_dict=None,
        human_dict=dict(rest_human_offsets=offsets), guidance_fn=None, guidance_scale=1.0,
        ddim_index=ddim_index,
    )
    with patch.object(engine, "predict_clean", side_effect=prediction) as predict:
        state, _, clean = engine.p_sample(**kwargs)
    assert predict.call_count == 2
    assert not predict.call_args_list[0].kwargs.get("is_uncondition", False)
    assert predict.call_args_list[1].kwargs["is_uncondition"]
    assert torch.equal(clean, torch.full_like(x, 3.0))
    assert state.shape == x.shape
    assert torch.isfinite(state).all()


@pytest.mark.parametrize("mode", ["teacher_forced_boundary", "predictor_decomp"])
def test_diagnostic_records_use_complete_prediction_and_supply_geometry_frame(mode):
    from types import SimpleNamespace
    import numpy as np
    import test_infbagel_lingo_hsi as evaluator

    item = dict(
        joints=torch.arange(16).float()[:, None].expand(16, 84).clone(),
        global_rot_6d=torch.zeros(16, 22, 6), object_trans=torch.zeros(16, 3),
        object_rot_mat=torch.zeros(16, 3, 3), contact_label=torch.zeros(16, 4),
        mat=torch.eye(4), scene_flag=torch.tensor(7),
        text_clip_embedding=torch.zeros(768), pelvis_goal=torch.zeros(3),
        scene_goal=torch.zeros(3), object_goal=torch.zeros(3),
        need_scene=torch.tensor(True), need_pelvis_dir=torch.tensor(True),
        pi=torch.tensor(0), end_pi=torch.tensor(48), seg_len=torch.tensor(48),
        need_pi=torch.tensor(False), is_loco=torch.tensor(True), is_object=torch.tensor(False),
        obj_bps_data=torch.zeros(1024, 3), object_points=torch.zeros(5, 3),
        obj_rot_mat_ref=torch.eye(3), rest_human_offsets=torch.ones(24, 3),
        betas=torch.zeros(16), gender="male", seq_name="test_window",
    )
    dataset = SimpleNamespace(
        lingo_dataset=SimpleNamespace(
            start_ind=[0], transl=np.zeros((1, 3)), joints=np.zeros((1, 28, 3)), step=3,
        ),
        denormalize_torch=lambda value: value,
    )
    calls, rebuilt = [], []

    def direct_model(*args, **kwargs):
        raise AssertionError("diagnostics must use the shared clean prediction")

    def predict_clean(model, noisy, occ, timestep, *args, **kwargs):
        assert model is direct_model
        torch.testing.assert_close(kwargs["mat"], item["mat"][None])
        torch.testing.assert_close(kwargs["scene_flag"], item["scene_flag"][None])
        torch.testing.assert_close(kwargs["rest_human_offsets"], item["rest_human_offsets"][None])
        calls.append((noisy.clone(), kwargs))
        return torch.full_like(noisy, 10.0 * len(calls))

    def reconstruct(value, *args, **kwargs):
        rebuilt.append(value.clone())
        count = kwargs.get("frame_count", 16)
        return value[:, :count, :72].reshape(1, count, 24, 3)

    engine = SimpleNamespace(
        student_model=direct_model, predict_clean=predict_clean,
        q_sample=lambda x_start, t, noise: x_start.clone(),
        _compute_occ=lambda *args: (None, None, None),
    )
    cfg = SimpleNamespace(device="cpu", seed=42, smplx_batch_size=1,
                          fps=30, predictor_decomp_timestep=259)
    selection = dict(data_idx=0, episode_id="example", stratum="walk", window_index=0)
    with patch.object(evaluator, "_lingo_item", return_value=item), patch.object(
        evaluator, "_teacher_forced_smplx_joints", side_effect=reconstruct
    ):
        if mode == "teacher_forced_boundary":
            record = evaluator._teacher_forced_window_record(
                cfg, dataset, engine, selection, [259], {},
            )
            assert len(calls) == 1
            assert set(record["metrics"]) == {"259"}
            assert torch.equal(rebuilt[-1], torch.full((1, 16, 232), 10.0))
        else:
            _, arrays = evaluator._predictor_decomp_window_record(
                cfg, dataset, engine, selection, {},
            )
            assert len(calls) == 3
            assert calls[0][1]["is_sample"]
            assert calls[1][1]["is_uncondition"]
            assert torch.equal(calls[2][0][:, 1, :84], calls[2][0][:, 0, :84])
            assert np.array_equal(arrays["conditional"], np.full((16, 232), 10.0))
            assert np.array_equal(arrays["unconditional"], np.full((16, 232), 20.0))
            assert np.array_equal(arrays["zero_velocity_history"], np.full((16, 232), 30.0))
