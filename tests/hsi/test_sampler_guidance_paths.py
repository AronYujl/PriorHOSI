"""Behavioral CPU tests for guidance in both formal sampler paths."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import guidance_loss  # noqa: E402
import models.infbagel as infbagel  # noqa: E402
from models.infbagel import Sampler  # noqa: E402
from utils import load_object_geometry_w_rest_geo  # noqa: E402


class IdentityModel:
    def __call__(self, x, *args, **kwargs):
        return x


class SamplerGuidancePathTests(unittest.TestCase):
    def _sampler(self, enabled):
        sampler = Sampler.__new__(Sampler)
        sampler.device = torch.device("cpu")
        sampler.batch_size = 1
        sampler.mask_ind = 0
        sampler.is_mix = False
        sampler.w = 0.0
        sampler.hsi_guidance_norm_cap = None
        sampler.hsi_guidance_alpha_decay = False
        sampler.hsi_guidance_dose_scale = None
        sampler.hsi_guidance_sdf_proxy = "area512" if enabled else None
        sampler.hsi_guidance_sdf_weight = 1.0 if enabled else 0.0
        sampler.cm_timesteps = 2
        sampler.alpha_cumprod = torch.tensor([1.0, 0.9, 0.8, 0.7])
        sampler.posterior_mean_coef1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sampler.posterior_mean_coef2 = torch.tensor([0.9, 0.8, 0.7, 0.6])
        sampler.posterior_log_variance_clipped = torch.tensor([-0.1, -0.2, -0.3, -0.4])
        sampler.solver = SimpleNamespace(
            ddim_timesteps=torch.tensor([0, 1, 2, 3]),
            ddim_timesteps_prev=torch.tensor([0, 0, 1, 2]),
            ddim_alpha_cumprods_prev=torch.tensor([1.0, 0.9, 0.8, 0.7]),
        )
        sampler.dataset = SimpleNamespace(
            max_window_size=1,
            load_scene=False,
            vis=False,
            denormalize_torch=lambda data, is_object=False: data,
            quat_ik_torch=lambda rotations: rotations,
            quat_fk_torch=lambda local_rotations, positions: (None, positions),
            get_nearest_free_voxel=lambda points, flags: (
                torch.zeros(points.shape[:-1], dtype=torch.bool),
                torch.zeros_like(points),
            ),
            prepare_nearest_free_voxel=MagicMock(return_value=MagicMock(name="prepared_query")),
            scene_geometry=MagicMock(return_value=object()),
        )
        sampler._compute_occ_sample = MagicMock(return_value=(None, None, None))
        sampler.set_fixed_points = MagicMock()
        return sampler

    @staticmethod
    def _inputs():
        x = torch.zeros(1, 1, 232)
        x[..., :3] = torch.tensor([0.3, 0.4, 0.5])
        x[..., 84:216] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(22)
        return dict(
            x0=x.clone(),
            x=x.clone(),
            fixed_points=torch.zeros(1, 0, 84),
            mat=torch.eye(4).reshape(1, 4, 4),
            scene_flag=torch.tensor([0]),
            t=torch.tensor([2]),
            t_index=1,
            text_emb=None,
            pelvis_goal=torch.zeros(1, 3),
            scene_goal=torch.zeros(1, 3),
            object_goal=torch.zeros(1, 3),
            need_scene=torch.tensor([True]),
            need_pelvis_dir=torch.tensor([False]),
            pi=torch.tensor([0]),
            end_pi=torch.tensor([1]),
            seq_length=torch.tensor([1]),
            need_pi=torch.tensor([False]),
            is_loco=torch.tensor([False]),
            is_object=torch.tensor([False]),
            obj_bps_data=None,
            object_points=None,
            obj_rot_mat_ref=None,
            obj_rest_verts=None,
            obj_vert_normals=None,
            seq_name_dict={},
            human_dict={
                "rest_human_offsets": torch.zeros(1, 1, 24, 3),
                "transl": torch.zeros(1, 3),
                "betas": torch.zeros(1, 16),
                "gender": ["male"],
            },
            obj_rot_mat_prefix=None,
            object_only=False,
            guidance_scale=1.0,
        )

    def _run(self, path_name, enabled):
        sampler = self._sampler(enabled)
        inputs = self._inputs()
        guidance_fn = guidance_loss.select_guidance_fn(enabled, torch.tensor([False]))
        calls = []

        def fake_hsi_loss(human, global_rotations, local_rotations, *, scene_flag,
                          get_nearest_free_voxel, geometry, cfg):
            del get_nearest_free_voxel
            calls.append((human, global_rotations, local_rotations, scene_flag, geometry, cfg))
            self.assertTrue(human.requires_grad)
            self.assertEqual(cfg.hsi_guidance_sdf_proxy, "area512")
            self.assertEqual(cfg.hsi_guidance_sdf_weight, 1.0)
            return human[..., 0].pow(2).mean()

        with patch.object(infbagel, "apply_hsi_scene_guidance_loss", side_effect=fake_hsi_loss):
            torch.manual_seed(123)
            if path_name == "p_sample":
                result = sampler.p_sample(
                    model=IdentityModel(), guidance_fn=guidance_fn, **inputs
                )
            else:
                result = sampler.cm_sample(
                    model=IdentityModel(), guidance_fn=guidance_fn, w=0, **inputs
                )
        return sampler, result, calls

    def test_diffusion_and_consistency_inject_enabled_guidance_with_gradients(self):
        for path_name in ("p_sample", "cm_sample"):
            with self.subTest(path=path_name):
                sampler, enabled, calls = self._run(path_name, enabled=True)
                disabled_sampler, disabled, disabled_calls = self._run(path_name, enabled=False)

                self.assertEqual(len(calls), 1)
                self.assertEqual(disabled_calls, [])
                self.assertEqual(sampler.dataset.scene_geometry.call_count, 1)
                self.assertEqual(disabled_sampler.dataset.scene_geometry.call_count, 0)
                self.assertIs(calls[0][4], sampler.dataset.scene_geometry.return_value)
                self.assertFalse(torch.equal(enabled[0], disabled[0]))
                self.assertGreater(
                    float((enabled[0][..., :3] - disabled[0][..., :3]).abs().sum()), 0.0
                )

    def test_mixed_sampler_guidance_masks_rows_before_each_expert(self):
        sampler = self._sampler(enabled=True)
        x_start = torch.zeros(2, 1, 232, requires_grad=True)
        with torch.no_grad():
            x_start[:, :, :72] = torch.arange(144, dtype=torch.float32).reshape(2, 1, 72) / 10
        human_jnts = x_start[:, :, :72].reshape(2, 1, 24, 3)
        global_rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(2, 1, 22, 1, 1)
        local_rotations = global_rotations.reshape(2, 22, 3, 3)
        scene_flag = torch.tensor([3, 4])
        is_object = torch.tensor([False, True])
        object_calls = []

        def fake_hsi_loss(human, global_rot, local_rot, *, scene_flag, get_nearest_free_voxel, geometry, cfg):
            flags = scene_flag
            del global_rot, local_rot, get_nearest_free_voxel, cfg
            self.assertEqual(tuple(flags.tolist()), (3,))
            self.assertEqual(tuple(human.shape[:2]), (1, 1))
            self.assertIs(geometry, sampler.dataset.scene_geometry.return_value)
            return human[..., 0].pow(2).mean()

        def fake_object_loss(human, obj_verts, pred_pos, pred_rot, contacts, flags, nearest):
            del pred_pos, pred_rot, contacts, flags, nearest
            object_calls.append((human, obj_verts))
            return human[..., 1].pow(2).mean() + obj_verts.pow(2).mean()

        object_geometry = x_start[1:, :, :6].reshape(1, 1, 2, 3)
        sampler._build_object_guidance_inputs = MagicMock(
            return_value=(
                x_start[1:, :, 216:219],
                torch.eye(3).reshape(1, 1, 3, 3),
                x_start[1:, :, 228:232],
                object_geometry,
            )
        )
        with patch.object(infbagel, "apply_hsi_scene_guidance_loss", side_effect=fake_hsi_loss):
            mixed_loss = sampler._guidance_loss(
                x_start,
                human_jnts,
                global_rotations,
                local_rotations,
                torch.eye(4).reshape(1, 4, 4).repeat(2, 1, 1),
                scene_flag,
                is_object,
                None,
                None,
                None,
                {1: "sample_box"},
                None,
                fake_object_loss,
            )

        self.assertEqual(len(object_calls), 1)
        self.assertEqual(tuple(object_calls[0][0].shape[:2]), (1, 1))
        self.assertEqual(tuple(object_calls[0][1].shape[:2]), (1, 1))
        self.assertEqual(sampler._build_object_guidance_inputs.call_args.args[-1].tolist(), [1])
        self.assertEqual(sampler.dataset.scene_geometry.call_args.args[0].tolist(), [3])
        self.assertTrue(torch.isfinite(mixed_loss))
        mixed_loss.backward()
        self.assertTrue(torch.isfinite(x_start.grad).all())
        self.assertGreater(float(x_start.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(x_start.grad[1].abs().sum()), 0.0)

    def test_each_sampling_loop_resolves_scene_geometry_once_for_all_steps(self):
        guidance_fn = object()
        for loop_name in ("p_sample_loop", "cm_sample_loop"):
            with self.subTest(loop=loop_name):
                sampler = self._sampler(enabled=True)
                sampler.student_model = torch.nn.Linear(1, 1)
                sampler.channel = 232
                sampler.auto_regre_num = 0
                sampler.timesteps = 3
                sampler.cm_timesteps = 3
                inputs = self._inputs()
                loop_inputs = {
                    key: inputs[key]
                    for key in (
                        "fixed_points",
                        "mat",
                        "scene_flag",
                        "text_emb",
                        "pelvis_goal",
                        "scene_goal",
                        "object_goal",
                        "need_scene",
                        "need_pelvis_dir",
                        "pi",
                        "end_pi",
                        "seq_length",
                        "need_pi",
                        "is_loco",
                        "is_object",
                        "obj_bps_data",
                        "object_points",
                        "obj_rot_mat_ref",
                        "obj_rest_verts",
                        "obj_vert_normals",
                        "seq_name_dict",
                        "human_dict",
                    )
                }
                loop_inputs.update(
                    guidance_fn=guidance_fn,
                    guidance_scale=1.0,
                    object_only=False,
                )

                def fake_sample(*args, **kwargs):
                    del kwargs
                    return args[2], None, args[2]

                if loop_name == "p_sample_loop":
                    sampler.p_sample = MagicMock(side_effect=fake_sample)
                    sampler.p_sample_loop(**loop_inputs)
                    calls = sampler.p_sample.call_args_list
                else:
                    sampler.cm_sample = MagicMock(side_effect=fake_sample)
                    sampler.cm_sample_loop(**loop_inputs)
                    calls = sampler.cm_sample.call_args_list
                self.assertEqual(sampler.dataset.scene_geometry.call_count, 1)
                self.assertEqual(sampler.dataset.prepare_nearest_free_voxel.call_count, 1)
                self.assertEqual(len(calls), 3)
                cached_geometry = sampler.dataset.scene_geometry.return_value
                prepared_query = sampler.dataset.prepare_nearest_free_voxel.return_value
                self.assertTrue(
                    all(call.kwargs["scene_geometry"] is cached_geometry for call in calls)
                )
                self.assertTrue(
                    all(call.kwargs["nearest_free_voxel"] is prepared_query for call in calls)
                )
                sampler.dataset.scene_geometry.reset_mock()

    def test_object_guidance_vertices_match_without_normals_work(self):
        sampler = self._sampler(enabled=False)
        sampler.dataset.max_window_size = 2
        x_start = torch.zeros(1, 2, 232)
        x_start[:, :, 219:228] = torch.eye(3).reshape(1, 1, 9)
        x_start[:, :, 216:219] = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]
        )
        rest_verts = {"box": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])}
        rest_normals = {"box": torch.full((2, 3), float("nan"))}
        result = sampler._build_object_guidance_inputs(
            x_start,
            torch.eye(4).reshape(1, 4, 4),
            torch.eye(3),
            rest_verts,
            rest_normals,
            {0: "scene_box"},
            None,
            torch.tensor([0]),
        )
        expected = load_object_geometry_w_rest_geo(
            torch.eye(3).reshape(1, 3, 3).repeat(2, 1, 1),
            x_start[0, :, 216:219],
            rest_verts["box"],
        ).reshape(1, 2, 2, 3)
        torch.testing.assert_close(result[3], expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
