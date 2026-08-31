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

    def reset(self):
        self.calls = 0

    def denormalize_torch(self, value, is_object=False):
        return value

    def create_meshgrid(self, batch_size=1):
        return torch.zeros(batch_size, 27, 3, dtype=torch.float32)

    def get_occ_for_points(self, points, object_points, scene_flag):
        del points, object_points, scene_flag
        grid = (self.base, self.goal, self.temporal)[min(self.calls, 2)]
        self.calls += 1
        return grid.reshape(1, -1)


def _sampler(**kwargs):
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
        temp_voxel_num=1,
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


if __name__ == "__main__":
    unittest.main()
