"""Bitwise and call-count gates for the R2-CG inference engineering path."""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code'))

from models.infbagel import Sampler  # noqa: E402


class CountingSceneDataset:
    load_scene = True
    max_window_size = 16
    nb_voxels = [2, 3, 4]
    vis = False

    def __init__(self):
        self.meshgrid_calls = 0
        self.occupancy_calls = 0

    def create_meshgrid(self, batch_size=1):
        self.meshgrid_calls += 1
        grid = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)
        return grid.repeat(batch_size, 1, 1) / 100.0

    @staticmethod
    def denormalize_torch(value, is_object=False):
        del is_object
        return value

    def get_occ_for_points(self, points, object_points, scene_flag):
        del object_points, scene_flag
        self.occupancy_calls += 1
        return (points[..., :1] > 0.25).to(torch.int8)


def _sampler_and_dataset():
    sampler = Sampler(
        device='cpu', mask_ind=0, emb_f=0, batch_size=1, channel=232,
        auto_regre_num=2, timesteps=500, ddim_timesteps=25, cm_timesteps=16,
        scene_type='occ_temp', temp_voxel_num=3,
        occ_list_layout_repaired=True,
    )
    dataset = CountingSceneDataset()
    sampler.set_dataset_and_model(dataset, torch.nn.Linear(1, 1))
    return sampler, dataset


def _arguments():
    torch.manual_seed(7)
    x = torch.randn(1, 16, 232)
    x0 = torch.randn(1, 16, 232)
    identity = torch.eye(3).reshape(1, 1, 9)
    x[..., 219:228] = identity
    return (
        x, x0, torch.eye(4).reshape(1, 4, 4),
        torch.zeros(1, dtype=torch.long), torch.randn(1, 5, 3),
        torch.tensor([[0.3, 0.0, 0.4]]), torch.tensor([[0.2, 0.0, 0.1]]),
        torch.tensor([[0.1, 0.0, 0.2]]), torch.zeros(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
        torch.eye(3).reshape(1, 3, 3), False, {}, {0: 'suitcase'}, None,
    )


class OccupancyEngineeringTests(unittest.TestCase):
    def test_only_the_window_static_goal_query_is_cached(self):
        arguments = _arguments()

        baseline, baseline_dataset = _sampler_and_dataset()
        baseline_outputs = [
            baseline._compute_occ_sample(*arguments),
            baseline._compute_occ_sample(*arguments),
        ]

        optimized, optimized_dataset = _sampler_and_dataset()
        static_cache = {}
        optimized_outputs = [
            optimized._compute_occ_sample(
                *arguments, inference_engineering=True,
                static_cache=static_cache,
            ),
            optimized._compute_occ_sample(
                *arguments, inference_engineering=True,
                static_cache=static_cache,
            ),
        ]

        for baseline_call, optimized_call in zip(
            baseline_outputs, optimized_outputs,
        ):
            for baseline_tensor, optimized_tensor in zip(
                baseline_call, optimized_call,
            ):
                self.assertTrue(torch.equal(baseline_tensor, optimized_tensor))

        # Five queries per baseline step: anchor, goal and three temporal.
        self.assertEqual(baseline_dataset.occupancy_calls, 10)
        # The optimized second step skips only goal; four dynamic queries remain.
        self.assertEqual(optimized_dataset.occupancy_calls, 9)
        # set_dataset_and_model creates the reusable grid once. Baseline then
        # recreates it per step; the optimized path does not.
        self.assertEqual(baseline_dataset.meshgrid_calls, 3)
        self.assertEqual(optimized_dataset.meshgrid_calls, 1)
        self.assertIn('goal', static_cache)


if __name__ == '__main__':
    unittest.main()
