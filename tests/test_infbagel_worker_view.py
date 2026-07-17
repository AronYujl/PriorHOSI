import sys
import unittest
from pathlib import Path

import torch
CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from datasets.infbagel import InfBaGelDataset  # noqa: E402


class InfBaGelWorkerViewTest(unittest.TestCase):
    def test_worker_view_drops_rank_gpu_state_without_mutating_main_dataset(self):
        dataset = object.__new__(InfBaGelDataset)
        dataset.device = 'cuda:3'
        dataset.scene_occ = object()
        dataset.scene_occ_ref = object()
        dataset.scene_grid_torch = object()
        dataset.batch_id = object()
        dataset.batch_id_obj = object()
        dataset.min_torch = object()
        dataset.max_torch = object()
        dataset.obj_min_torch = object()
        dataset.obj_max_torch = object()
        dataset.cpu_value = object()

        worker_dataset = dataset.cpu_worker_view()

        self.assertIsNot(worker_dataset, dataset)
        self.assertEqual(worker_dataset.device, 'cpu')
        self.assertIs(worker_dataset.cpu_value, dataset.cpu_value)
        for name in (
                'scene_occ', 'scene_occ_ref', 'scene_grid_torch',
                'batch_id', 'batch_id_obj', 'min_torch', 'max_torch',
                'obj_min_torch', 'obj_max_torch'):
            self.assertFalse(hasattr(worker_dataset, name), name)
            self.assertTrue(hasattr(dataset, name), name)

    def test_direct_training_occupancy_matches_full_grid_lookup(self):
        dataset = object.__new__(InfBaGelDataset)
        dataset.device = 'cpu'
        dataset.train = True
        dataset.vis = False
        dataset.load_object_goal = True
        dataset.scene_grid_torch = torch.tensor(
            [0., 0., 0., 4., 4., 4., 4., 4., 4.])
        dataset.scene_occ = torch.zeros((2, 4, 4, 4), dtype=torch.int8)
        dataset.scene_occ[0, 1, 1, 1] = 1
        dataset.scene_occ[1, 2, 2, 2] = 1
        dataset.batch_id_obj = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)

        points = torch.tensor([
            [[1.1, 1.1, 1.1], [0.1, 0.1, 0.1], [-1., 0., 0.]],
            [[2.1, 2.1, 2.1], [3.1, 3.1, 3.1], [4.1, 0., 0.]],
        ])
        object_points = torch.tensor([
            [[0.1, 0.1, 0.1], [3.1, 3.1, 3.1]],
            [[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]],
        ])
        scene_flag = torch.tensor([0, 1], dtype=torch.long)

        actual = dataset.get_occ_for_points(points, object_points, scene_flag)

        voxel_size = torch.ones(3)
        voxel = ((points.reshape(-1, 3) - 0.) / voxel_size).long()
        in_bound = torch.all((voxel >= 0) & (voxel < 4), dim=-1)
        voxel[~in_bound] = 0
        full_occ = dataset.scene_occ[scene_flag].clone()
        object_voxel = ((object_points.reshape(-1, 3) - 0.) / voxel_size).long()
        object_voxel[~torch.all((object_voxel >= 0) & (object_voxel < 4), dim=-1)] = 0
        full_occ[dataset.batch_id_obj[:, 0], object_voxel[:, 0], object_voxel[:, 1], object_voxel[:, 2]] = 2
        batch_index = torch.arange(2).view(-1, 1).expand(-1, 3).reshape(-1)
        expected = full_occ[batch_index, voxel[:, 0], voxel[:, 1], voxel[:, 2]]
        expected[~in_bound] = 1
        expected = expected.reshape(2, 3)

        self.assertTrue(torch.equal(actual, expected), (actual, expected))

if __name__ == '__main__':
    unittest.main()
