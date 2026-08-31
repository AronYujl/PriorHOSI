"""CPU contracts for the P17-OC SDF bank, hinge, and objective plumbing."""

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import pytorch3d.transforms as transforms
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import Sampler
from priors.hsi.penetration import SceneSDFBank
from priors.hsi.scene_field import SDF_VOXEL_SIZE, SceneGeometry
from utils import transform_points


SOURCE = (REPO / "code" / "models" / "infbagel.py").read_text()


def _geometry(name, shape=(7, 6, 5), origin=(-0.2, -0.2, -0.2), field=None):
    if field is None:
        generator = torch.Generator().manual_seed(len(name) + sum(shape))
        field = torch.randn(shape, generator=generator) * 0.2
    return SceneGeometry(
        name,
        field.numpy().astype(np.float32),
        np.asarray(origin, dtype=np.float32),
        SDF_VOXEL_SIZE,
        is_watertight=True,
    )


def _interior_points(geometry, batch, frames, joints, seed):
    low, high = geometry.bounds
    generator = torch.Generator().manual_seed(seed)
    unit = torch.rand((batch, frames, joints, 3), generator=generator)
    return low + (high - low) * (0.05 + 0.90 * unit)


class SceneSDFBankTests(unittest.TestCase):
    def test_float32_batched_gather_matches_the_audited_interpolator(self):
        first = _geometry("first", shape=(7, 6, 5))
        second = _geometry("second", shape=(5, 4, 8), origin=(-0.1, -0.16, -0.12))
        points = torch.cat(
            (
                _interior_points(first, 1, 4, 37, 1),
                _interior_points(second, 1, 4, 37, 2),
            ),
            dim=0,
        )
        flags = torch.tensor([10, 20], dtype=torch.long)
        bank = SceneSDFBank.from_geometries(
            {10: first, 20: second}, dtype=torch.float32, device="cpu"
        )

        got, out_of_bounds = bank.signed_distance(points, flags)
        expected = torch.stack(
            (
                first.signed_distance(points[0]),
                second.signed_distance(points[1]),
            ),
            dim=0,
        )
        error = (got - expected).abs()
        self.assertLessEqual(float(error.max()), 1e-4)
        self.assertLess(float(error.max()), 1e-6)
        self.assertFalse(bool(out_of_bounds.any()))
        self.assertEqual(bank.flat_field.dtype, torch.float32)

    def test_float16_storage_is_checked_separately_in_the_active_band(self):
        shape = (9, 8, 7)
        field = torch.linspace(-0.2, 0.2, steps=int(np.prod(shape))).reshape(shape)
        geometry = _geometry("band", shape=shape, field=field)
        points = _interior_points(geometry, 2, 5, 41, 3)
        flags = torch.zeros(2, dtype=torch.long)
        reference = SceneSDFBank.from_geometries(
            {0: geometry}, dtype=torch.float32, device="cpu"
        ).signed_distance(points, flags)[0]
        stored = SceneSDFBank.from_geometries(
            {0: geometry}, dtype=torch.float16, device="cpu"
        ).signed_distance(points, flags)[0]

        active_band = reference.abs() <= 0.25
        self.assertTrue(bool(active_band.all()))
        error = (stored - reference).abs()[active_band]
        self.assertLessEqual(float(error.max()), 1.5e-4)

    def test_mirror_flags_negate_only_query_x(self):
        source = _geometry("source", shape=(20, 20, 20), origin=(-0.2, -0.2, -0.2))
        points = _interior_points(source, 1, 3, 43, 4)
        # Both x and -x lie in the source field's symmetric bounds.
        points[..., 0] = points[..., 0].abs()
        points[..., 0] = points[..., 0].clamp(max=0.15)
        bank = SceneSDFBank.from_geometries(
            {0: source, 1: source},
            flag_to_name={0: "source", 1: "source_mirror"},
            dtype=torch.float32,
            device="cpu",
        )

        got, out_of_bounds = bank.signed_distance(points, torch.tensor([1]))
        flipped = points * torch.tensor([-1.0, 1.0, 1.0])
        expected = source.signed_distance(flipped)
        torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-6)
        self.assertFalse(bool(out_of_bounds.any()))

    @mock.patch("priors.hsi.penetration.SceneGeometry.from_scene")
    def test_scene_flag_loading_deduplicates_mirror_sources(self, load_scene):
        source = _geometry("source", shape=(4, 5, 6), origin=(-0.1, -0.1, -0.1))
        load_scene.return_value = source
        bank = SceneSDFBank.from_scene_flags(
            {7: "source", 8: "source_mirror"},
            dataset_root=REPO / "data" / "dataset",
            mesh_root=REPO / "mesh",
            cache_dir=REPO / ".cache" / "hsi_sdf",
            dtype="float32",
            device="cpu",
            require_cache=False,
        )

        self.assertEqual(load_scene.call_count, 1)
        self.assertEqual(load_scene.call_args.args, ("source",))
        self.assertEqual(bank.scene_names, ("source",))
        self.assertTrue(bool(bank.flag_mirror[1]))

    def test_scene_flag_loading_refuses_missing_cache_before_geometry_load(self):
        with mock.patch.object(
            SceneSDFBank,
            "_cache_path",
            return_value=Path("/definitely-missing-p17oc-cache.npz"),
        ), mock.patch("priors.hsi.penetration.SceneGeometry.from_scene") as load_scene:
            with self.assertRaisesRegex(FileNotFoundError, "source"):
                SceneSDFBank.from_scene_flags(
                    {7: "source"},
                    dataset_root=REPO / "data" / "dataset",
                    mesh_root=REPO / "mesh",
                    cache_dir=REPO / ".cache" / "hsi_sdf",
                    dtype="float32",
                    device="cpu",
                )

        load_scene.assert_not_called()

    def test_unknown_scene_flags_are_rejected_before_gather(self):
        geometry = _geometry("known", shape=(5, 5, 5))
        bank = SceneSDFBank.from_geometries(
            {10: geometry, 20: geometry}, dtype=torch.float32, device="cpu"
        )
        points = _interior_points(geometry, 3, 1, 1, 19)

        with self.assertRaisesRegex(ValueError, r"15"):
            bank.signed_distance(points, torch.tensor([10, 15, 20]))
        with self.assertRaisesRegex(ValueError, r"99"):
            bank.signed_distance(points, torch.tensor([10, 20, 99]))
        with self.assertRaisesRegex(ValueError, r"-1"):
            bank.signed_distance(points, torch.tensor([-1, 10, 20]))


class PenetrationFormTests(unittest.TestCase):
    def test_single_sided_hinge_has_exactly_zero_free_gradients(self):
        sdf = torch.tensor([-0.05, -0.03, -0.02, 0.01, 0.5], requires_grad=True)
        delta = 0.03
        d = torch.clamp(-(sdf + delta), min=0.0)
        loss = (d ** 2).mean()
        grad = torch.autograd.grad(loss, sdf)[0]

        self.assertLess(float(grad[0]), 0.0)
        self.assertTrue(torch.equal(grad[1:], torch.zeros(4)))

        # Rebuild the control as a leaf so its nonzero derivative is observable.
        control_sdf = sdf.detach().requires_grad_(True)
        control_grad = torch.autograd.grad((control_sdf ** 2).mean(), control_sdf)[0]
        self.assertTrue(bool((control_grad[1:] != 0).all()))

    def test_source_keeps_all_scorable_points_in_the_hinge_denominator(self):
        tree = ast.parse(SOURCE)
        p_losses = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "p_losses"
        )
        dump = ast.dump(p_losses)
        self.assertIn("m_scorable", dump)
        self.assertIn("pen_delta", dump)
        self.assertIn("loss_pen", dump)
        self.assertIn("d ** 2", SOURCE)
        self.assertNotIn("d[d > 0]", SOURCE)


class _NoObjectDataset:
    load_scene = True
    use_object_keypoints = False
    max_window_size = 4
    nb_voxels = (1, 1, 1)


class _OffsetModel(nn.Module):
    def forward(self, x, *args):
        return x + 0.25


class _PenetrationDataset(_NoObjectDataset):
    def denormalize_torch(self, points, is_object=False):
        return points

    def quat_ik_torch(self, rotations):
        return rotations

    def quat_fk_torch(self, rotations, positions):
        return None, positions


class _FKDataset(_PenetrationDataset):
    use_object_keypoints = True


class _FixedSDFBank:
    def __init__(self):
        self.points = None

    def signed_distance(self, points, scene_flag):
        self.points = points
        sdf = torch.full(points.shape[:-1], -0.02, dtype=points.dtype, device=points.device)
        sdf[..., 0] = -0.05
        return sdf, torch.zeros_like(sdf, dtype=torch.bool)


def _sampler_for_loss(**kwargs):
    sampler = Sampler(
        device="cpu",
        mask_ind=0,
        emb_f=None,
        batch_size=1,
        channel=232,
        auto_regre_num=2,
        timesteps=500,
        ddim_timesteps=25,
        cm_timesteps=16,
        **kwargs,
    )
    sampler.dataset = _NoObjectDataset()
    sampler.student_model = _OffsetModel()
    sampler._compute_occ = lambda *args, **kwargs: (None, None, None)
    sampler.q_sample = lambda x_start, t, noise: x_start
    return sampler


def _loss_inputs():
    generator = torch.Generator().manual_seed(17)
    x_start = torch.randn((1, 4, 232), generator=generator)
    mask = torch.zeros_like(x_start, dtype=torch.bool)
    mask[:, :2] = True
    mat = torch.eye(4).unsqueeze(0)
    return dict(
        x_start=x_start,
        joints=x_start[:, :, :84],
        mat=mat,
        scene_flag=torch.tensor([0], dtype=torch.long),
        mask=mask,
        t=torch.tensor([0], dtype=torch.long),
        text_emb=torch.zeros((1, 1)),
        pelvis_goal=torch.zeros((1, 3)),
        scene_goal=torch.zeros((1, 3)),
        object_goal=torch.zeros((1, 3)),
        need_scene=torch.ones(1, dtype=torch.bool),
        need_pelvis_dir=torch.zeros(1, dtype=torch.bool),
        pi=torch.zeros(1, dtype=torch.long),
        end_pi=torch.zeros(1, dtype=torch.long),
        seq_length=torch.tensor([4], dtype=torch.long),
        need_pi=torch.zeros(1, dtype=torch.bool),
        is_loco=torch.zeros(1, dtype=torch.bool),
        is_object=torch.zeros(1, dtype=torch.bool),
        obj_bps_data=None,
        obj_rot_mat_ref=torch.eye(3).unsqueeze(0),
        rest_pose_obj_nn_pts=torch.zeros((1, 100, 3)),
        transformed_obj_verts=None,
        rest_human_offsets=torch.zeros((1, 24, 3)),
        object_points=torch.zeros((1, 1024, 3)),
        noise=torch.zeros_like(x_start),
    )


class ObjectivePlumbingTests(unittest.TestCase):
    def test_fk_loss_is_bitwise_equal_to_the_original_inline_block(self):
        sampler = _sampler_for_loss()
        sampler.dataset = _FKDataset()
        inputs = _loss_inputs()
        result = sampler.p_losses(**inputs)

        predicted_noise = inputs["x_start"] + 0.25
        joints = inputs["joints"]
        mat = inputs["mat"]
        rest_human_offsets = inputs["rest_human_offsets"]

        # Inline copy of the pre-refactor FK block.
        global_jpos = transform_points(
            sampler.dataset.denormalize_torch(predicted_noise[:, :, :84]), mat
        ).reshape(joints.shape[0], -1, 28, 3)
        curr_seq_local_jpos = rest_human_offsets[:, None].repeat(
            1, global_jpos.shape[1], 1, 1
        )
        curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3)
        curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]
        global_jrot_6d = predicted_noise[:, :, 84:216].reshape(
            joints.shape[0], -1, 22, 6
        )
        global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d)
        global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat
        local_jrot_mat = sampler.dataset.quat_ik_torch(
            global_jrot_mat.reshape(-1, 22, 3, 3)
        )
        _, human_jnts = sampler.dataset.quat_fk_torch(
            local_jrot_mat, curr_seq_local_jpos
        )
        human_jnts = human_jnts.reshape(joints.shape[0], -1, 24, 3)

        hand_idx_28 = [20, 21, 25, 27]
        hand_idx_24 = [20, 21, 22, 23]
        foot_idx = [7, 8, 10, 11]
        gt_global_jpos = transform_points(
            sampler.dataset.denormalize_torch(joints), mat
        ).reshape(joints.shape[0], -1, 28, 3)
        mask_fk = torch.ones(
            1, sampler.dataset.max_window_size, 4, 3, dtype=torch.bool
        )
        mask_fk[:, :sampler.auto_regre_num] = False
        expected = torch.nn.functional.mse_loss(
            human_jnts[:, :, hand_idx_24, :][mask_fk],
            gt_global_jpos[:, :, hand_idx_28, :][mask_fk],
        ) + torch.nn.functional.mse_loss(
            human_jnts[:, :, foot_idx, :][mask_fk],
            gt_global_jpos[:, :, foot_idx, :][mask_fk],
        )

        torch.testing.assert_close(result["loss_fk"], expected, rtol=0.0, atol=0.0)

    def test_non_object_rows_use_finite_tensor_zero_guards(self):
        sampler = _sampler_for_loss()
        inputs = _loss_inputs()
        with mock.patch("models.infbagel.F.mse_loss", wraps=torch.nn.functional.mse_loss) as mse, \
             mock.patch("models.infbagel.F.l1_loss", wraps=torch.nn.functional.l1_loss) as l1:
            result = sampler.p_losses(**inputs)

        self.assertTrue(bool(torch.isfinite(result["loss"])))
        self.assertIsNone(result["loss_pen"])
        # Only the non-object position/rotation terms call the reducers; all three
        # object-channel reducers are skipped before they can see an empty tensor.
        self.assertEqual(mse.call_count, 1)
        self.assertEqual(l1.call_count, 1)
        method = next(
            node for node in ast.walk(ast.parse(SOURCE))
            if isinstance(node, ast.FunctionDef) and node.name == "p_losses"
        )
        zero_calls = [
            node for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "new_zeros"
        ]
        self.assertGreaterEqual(len(zero_calls), 3)

    def test_penetration_branch_is_inert_and_does_not_build_a_bank_when_off(self):
        sampler = _sampler_for_loss()
        inputs = _loss_inputs()
        with mock.patch.object(sampler, "_get_pen_sdf_bank", side_effect=AssertionError("bank built while off")):
            result = sampler.p_losses(**inputs)

        predicted = inputs["x_start"] + 0.25
        generated = ~inputs["mask"]
        expected = torch.nn.functional.mse_loss(
            inputs["x_start"][:, :, :84][generated[:, :, :84]],
            predicted[:, :, :84][generated[:, :, :84]],
        ) + torch.nn.functional.l1_loss(
            inputs["x_start"][:, :, 84:216][generated[:, :, 84:216]],
            predicted[:, :, 84:216][generated[:, :, 84:216]],
        )
        torch.testing.assert_close(result["loss"], expected, rtol=0.0, atol=0.0)
        self.assertEqual(sampler.pen_loss_weight, 0.0)
        self.assertIsNone(sampler.pen_sdf_bank)

    def test_active_penetration_uses_human_joints_and_all_scorable_points(self):
        sampler = _sampler_for_loss(pen_loss_weight=1.0)
        sampler.dataset = _PenetrationDataset()
        bank = _FixedSDFBank()
        sampler._get_pen_sdf_bank = lambda: bank
        inputs = _loss_inputs()
        inputs["x_start"] = torch.zeros((1, 4, 232))
        inputs["joints"] = inputs["x_start"][:, :, :84]
        inputs["rest_human_offsets"] = torch.full((1, 24, 3), 0.1)

        result = sampler.p_losses(**inputs)

        self.assertEqual(tuple(bank.points.shape), (1, 4, 24, 3))
        # Two generated frames x 24 joints are scorable; only joint 0 in each
        # generated frame penetrates, so the denominator is all 48 triples.
        self.assertAlmostEqual(float(result["loss_pen"]), 0.0004 * 2.0 / 48.0, places=10)

    def test_penetration_bank_restricts_lingo_flags_to_the_split(self):
        sampler = _sampler_for_loss(pen_loss_weight=1.0)
        sampler.dataset = type(
            "SplitDataset",
            (),
            {
                "split_manifest": "/does/not/exist.json",
                "split_partition": "train",
                "unified_scene_dict": {
                    "train": 10,
                    "train_mirror": 11,
                    "held_out": 12,
                    "held_out_mirror": 13,
                    "omomo_train": 14,
                },
                "unified_scene_source": {
                    10: "lingo",
                    11: "lingo",
                    12: "lingo",
                    13: "lingo",
                    14: "omomo",
                },
                "lingo_dataset": type("LingoDataset", (), {"folder": "/dataset"})(),
            },
        )()
        sentinel = object()
        manifest = '{"train": {"scenes": ["train", "train_mirror"]}}'
        with mock.patch("builtins.open", mock.mock_open(read_data=manifest)), \
             mock.patch.object(SceneSDFBank, "from_scene_flags", return_value=sentinel) as build:
            self.assertIs(sampler._get_pen_sdf_bank(), sentinel)

        build.assert_called_once()
        self.assertEqual(build.call_args.args[0], {10: "train", 11: "train_mirror"})
        self.assertTrue(build.call_args.kwargs["require_cache"])

    def test_penetration_bank_falls_back_to_all_lingo_flags_without_split(self):
        sampler = _sampler_for_loss(pen_loss_weight=1.0)
        sampler.dataset = type(
            "NoSplitDataset",
            (),
            {
                "split_manifest": None,
                "split_partition": "train",
                "unified_scene_dict": {"train": 10, "held_out": 12, "omomo": 14},
                "unified_scene_source": {10: "lingo", 12: "lingo", 14: "omomo"},
                "lingo_dataset": type("LingoDataset", (), {"folder": "/dataset"})(),
            },
        )()
        sentinel = object()
        with mock.patch.object(SceneSDFBank, "from_scene_flags", return_value=sentinel) as build:
            self.assertIs(sampler._get_pen_sdf_bank(), sentinel)

        self.assertEqual(build.call_args.args[0], {10: "train", 12: "held_out"})

    def test_p17oc_config_is_a_seven_key_override_and_sampler_resolves_all_keys(self):
        from omegaconf import OmegaConf

        config_path = REPO / "code" / "config" / "config_train_hsi_b_p17oc.yaml"
        raw = OmegaConf.load(config_path)
        self.assertEqual(list(raw.defaults), ["config_train_hsi_b_lingo_full", "_self_"])
        self.assertEqual(raw.exp_name, "hsi_b_p17oc")
        self.assertEqual(raw.pen_loss_weight, 8.36)
        self.assertEqual(raw.pen_delta, 0.03)
        self.assertEqual(raw.pen_floor_height, 0.02)
        self.assertEqual(raw.pen_sdf_dtype, "float16")
        self.assertEqual(
            sorted(raw.keys()),
            ["defaults", "exp_name", "occ_permute_fix", "pen_delta", "pen_floor_height",
             "pen_loss_weight", "pen_sdf_cache", "pen_sdf_dtype"],
        )
        self.assertIs(raw.occ_permute_fix, True)
        sampler_config = (REPO / "code" / "config" / "sampler" / "pelvis.yaml").read_text()
        for key, default in (
            ("pen_loss_weight", "0.0"),
            ("pen_delta", "0.03"),
            ("pen_floor_height", "0.02"),
            ("pen_sdf_cache", "null"),
            ("pen_sdf_dtype", "float16"),
        ):
            self.assertIn(f"{key}: ${{oc.select:{key},{default}}}", sampler_config)


if __name__ == "__main__":
    unittest.main()
