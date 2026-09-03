"""Exercise the E1 occ_list[0] layout switch on both sampler paths."""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import Sampler


class AsymmetricOccupancyDataset:
    """Small scene stub returning distinct base, goal, and temporal grids."""

    def __init__(self):
        self.load_scene = True
        self.nb_voxels = (3, 3, 3)
        self.max_window_size = 16
        self.vis = False
        values = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
        self.base = values
        self.goal = values + 100.0
        self.temporal = values + 200.0
        self.calls = 0
        self.query_centers = []

    def reset(self):
        self.calls = 0
        self.query_centers = []

    def denormalize_torch(self, value, is_object=False):
        return value

    def create_meshgrid(self, batch_size=1):
        return torch.zeros(batch_size, 27, 3, dtype=torch.float32)

    def get_occ_for_points(self, points, object_points, scene_flag):
        del object_points, scene_flag
        self.query_centers.append(points[:, 0].clone())
        grid = (self.base, self.goal, self.temporal)[min(self.calls, 2)]
        self.calls += 1
        return grid.reshape(1, -1)


def _sampler(**kwargs):
    temp_voxel_num = kwargs.pop("temp_voxel_num", 1)
    return Sampler(
        device="cpu",
        mask_ind=0,
        emb_f=0,
        batch_size=1,
        channel=232,
        auto_regre_num=1,
        timesteps=500,
        ddim_timesteps=25,
        cm_timesteps=16,
        scene_type="occ_temp",
        temp_voxel_num=temp_voxel_num,
        **kwargs,
    )


def _inputs():
    batch, frames = 1, 16
    noisy = torch.zeros(batch, frames, 232)
    noisy[:, :, 219:228] = torch.eye(3).reshape(1, 1, 9)
    x_start = noisy.clone()
    joints = torch.zeros(batch, frames, 84)
    mat = torch.eye(4).reshape(1, 4, 4)
    scene_flag = torch.zeros(batch, dtype=torch.long)
    pelvis_goal = torch.zeros(batch, 3)
    scene_goal = torch.zeros(batch, 3)
    object_goal = torch.zeros(batch, 3)
    is_loco = torch.zeros(batch, dtype=torch.bool)
    is_object = torch.zeros(batch, dtype=torch.bool)
    need_pelvis_dir = torch.ones(batch, dtype=torch.bool)
    object_points = torch.zeros(batch, 1, 3)
    obj_rot_mat_ref = torch.eye(3).reshape(1, 3, 3)
    return dict(
        noisy=noisy,
        x_start=x_start,
        joints=joints,
        mat=mat,
        scene_flag=scene_flag,
        object_points=object_points,
        pelvis_goal=pelvis_goal,
        scene_goal=scene_goal,
        object_goal=object_goal,
        is_loco=is_loco,
        is_object=is_object,
        need_pelvis_dir=need_pelvis_dir,
        obj_rot_mat_ref=obj_rot_mat_ref,
    )


def _training_occ(sampler, dataset, inputs):
    dataset.reset()
    return sampler._compute_occ(
        inputs["noisy"],
        inputs["x_start"],
        inputs["joints"],
        inputs["mat"],
        inputs["scene_flag"],
        inputs["object_points"],
        inputs["pelvis_goal"],
        inputs["scene_goal"],
        inputs["object_goal"],
        inputs["is_loco"],
        inputs["is_object"],
        inputs["need_pelvis_dir"],
        inputs["obj_rot_mat_ref"],
    )


def _inference_occ(sampler, dataset, inputs):
    dataset.reset()
    return sampler._compute_occ_sample(
        inputs["noisy"],
        inputs["x_start"],
        inputs["mat"],
        inputs["scene_flag"],
        inputs["object_points"],
        inputs["pelvis_goal"],
        inputs["scene_goal"],
        inputs["object_goal"],
        inputs["is_loco"],
        inputs["is_object"],
        inputs["need_pelvis_dir"],
        inputs["obj_rot_mat_ref"],
        False,
        None,
        None,
        None,
    )


class OccLayoutTests(unittest.TestCase):
    def test_future_crop_jitter_defaults_to_training_value_and_can_be_disabled(self):
        self.assertEqual(_sampler().hsi_future_occ_jitter_scale, 0.2)
        self.assertEqual(
            _sampler(hsi_future_occ_jitter_scale=0.0).hsi_future_occ_jitter_scale,
            0.0,
        )

    def _assert_path(self, path):
        inputs = _inputs()
        dataset = AsymmetricOccupancyDataset()
        legacy = _sampler()
        fixed = _sampler(occ_permute_fix=True)
        legacy.set_dataset_and_model(dataset, None)
        fixed.set_dataset_and_model(dataset, None)

        legacy_result = path(legacy, dataset, inputs)
        fixed_result = path(fixed, dataset, inputs)
        legacy_list = legacy_result[1]
        fixed_list = fixed_result[1]

        self.assertFalse(torch.equal(dataset.base, dataset.base.permute(1, 0, 2)))
        self.assertTrue(torch.equal(legacy_list[0], dataset.base))
        self.assertTrue(torch.equal(fixed_list[0], dataset.base.permute(1, 0, 2)))
        self.assertTrue(torch.equal(fixed_list[0], legacy_list[0].permute(1, 0, 2)))
        self.assertFalse(torch.equal(fixed_list[0], legacy_list[0]))
        self.assertTrue(torch.equal(fixed_list[1], legacy_list[1]))
        self.assertTrue(torch.equal(legacy_list[1], dataset.temporal.permute(1, 0, 2)))

        omitted = _sampler()
        omitted.set_dataset_and_model(dataset, None)
        omitted_result = path(omitted, dataset, inputs)
        self.assertFalse(omitted.occ_permute_fix)
        self.assertTrue(torch.equal(omitted_result[1], legacy_list))

    def test_training_path_both_branches_and_default(self):
        self._assert_path(_training_occ)

    def test_inference_path_both_branches_and_default(self):
        self._assert_path(_inference_occ)

    def test_future_crop_and_coordinate_sources_are_independent(self):
        inputs = _inputs()
        for index, frame in enumerate((5, 10, 15)):
            inputs["x_start"][:, frame, 0] = float(index + 1)
            inputs["x_start"][:, frame, 2] = float(10 + index)
        oracle = torch.tensor([[[21.0, 0.0, 31.0], [22.0, 0.0, 32.0], [23.0, 0.0, 33.0]]])
        expected = {
            "predicted": (False, False),
            "gt_crop": (True, False),
            "gt_coordinate": (False, True),
            "gt_both": (True, True),
        }
        for mode, (gt_crop, gt_coordinate) in expected.items():
            with self.subTest(mode=mode):
                dataset = AsymmetricOccupancyDataset()
                sampler = _sampler(temp_voxel_num=3, hsi_future_occ_mode=mode)
                sampler.set_dataset_and_model(dataset, None)
                sampler.set_hsi_future_occ_oracle(oracle)
                _occ, _occ_list, occ_pos = _inference_occ(sampler, dataset, inputs)
                predicted = inputs["x_start"][:, [5, 10, 15], :3]
                query_expected = oracle if gt_crop else predicted
                coordinate_expected = oracle[..., [0, 2]] if gt_coordinate else predicted[..., [0, 2]]
                got_queries = torch.stack(dataset.query_centers[2:5], dim=1)
                self.assertTrue(torch.equal(got_queries, query_expected))
                self.assertTrue(torch.equal(occ_pos[1:].permute(1, 0, 2), coordinate_expected))

    def test_gt_mode_requires_and_clears_window_oracle(self):
        inputs = _inputs()
        dataset = AsymmetricOccupancyDataset()
        sampler = _sampler(temp_voxel_num=3, hsi_future_occ_mode="gt_both")
        sampler.set_dataset_and_model(dataset, None)
        with self.assertRaisesRegex(RuntimeError, "requires a window-scoped oracle"):
            _inference_occ(sampler, dataset, inputs)
        oracle = torch.zeros(1, 3, 3)
        sampler.set_hsi_future_occ_oracle(oracle)
        with self.assertRaisesRegex(RuntimeError, "not cleared"):
            sampler.set_hsi_future_occ_oracle(oracle)
        sampler.clear_hsi_future_occ_oracle()


if __name__ == "__main__":
    unittest.main()
