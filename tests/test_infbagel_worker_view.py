import sys
import unittest
from pathlib import Path

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

if __name__ == '__main__':
    unittest.main()
