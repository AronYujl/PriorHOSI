"""Call ``InfBaGelMixDataset.get_nearest_free_voxel`` for real.

Guided HSI evaluation reaches this method once per guidance application, through
``guidance_loss.apply_hsi_guidance_loss``.  ``InfBaGelMixDataset`` subclasses
``torch.utils.data.Dataset``, not ``InfBaGelDataset``, so it borrows the body
unbound -- and therefore any dispatch *inside* that body resolves against the
mix instance, where the private implementations do not exist.  A ``hasattr``
check cannot see this; only a call can.  ``tests/hsi/test_guidance_routing.py``
covers which guidance function is selected, never whether its occupancy query
runs, which is why the dispatcher regression reached a launch-ready state.

The scene here is synthetic and 4x4x4, but the class, the method, the
``LazyOccRef`` displacement cache and the whole query body are the real ones.
"""

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from datasets.infbagel import InfBaGelDataset, LazyOccRef
from datasets.infbagel_mix import InfBaGelMixDataset

GRID_DIM = 4
OCCUPIED_VOXELS = {0: (1, 1, 1), 1: (2, 3, 0)}


def _mix_dataset_with_synthetic_scene(lazy=True):
    """A real ``InfBaGelMixDataset`` carrying only the attributes the query reads.

    ``__init__`` loads OMOMO and LINGO from disk, which no unit test can afford;
    ``__new__`` keeps the real type, the real MRO and the real bound method while
    supplying the three attributes ``infbagel_mix.py`` sets at lines 257-262.
    """
    dataset = InfBaGelMixDataset.__new__(InfBaGelMixDataset)
    occ = torch.zeros((2, GRID_DIM, GRID_DIM, GRID_DIM), dtype=torch.uint8)
    for scene_id, voxel in OCCUPIED_VOXELS.items():
        occ[(scene_id,) + voxel] = 1
    dataset.scene_occ = occ
    dataset.scene_occ_ref = LazyOccRef(occ) if lazy else torch.stack(
        [LazyOccRef(occ)[int(scene_id)] for scene_id in range(occ.shape[0])]
    )
    dataset.scene_grid_torch = torch.tensor(
        [0, 0, 0, 1, 1, 1, GRID_DIM, GRID_DIM, GRID_DIM], dtype=torch.int64
    )
    return dataset


def _voxel_center(voxel):
    return [(index + 0.5) / GRID_DIM for index in voxel]


def _probe_points():
    """[B=2, T=1, N=4, 3]: the occupied voxel of each scene, a free voxel, and
    two out-of-bounds points on either side of the grid."""
    per_scene = []
    for scene_id in range(2):
        per_scene.append(
            [
                _voxel_center(OCCUPIED_VOXELS[scene_id]),
                _voxel_center((0, 0, 0)),
                [-5.0, -5.0, -5.0],
                [5.0, 5.0, 5.0],
            ]
        )
    return torch.tensor(per_scene, dtype=torch.float32).reshape(2, 1, 4, 3)


class NearestFreeVoxelCallTests(unittest.TestCase):
    def test_mix_dataset_does_not_inherit_the_private_implementations(self):
        # The premise of the bug: nothing in the MRO provides the private bodies,
        # so the borrowed body must not dispatch through ``self``.
        self.assertNotIn(InfBaGelDataset, InfBaGelMixDataset.__mro__)
        self.assertFalse(hasattr(InfBaGelMixDataset, "_get_nearest_free_voxel_direct"))
        self.assertFalse(hasattr(InfBaGelMixDataset, "_get_nearest_free_voxel_materialized"))
        self.assertTrue(hasattr(InfBaGelMixDataset, "get_nearest_free_voxel"))

    def test_call_returns_documented_shapes_and_dtypes(self):
        dataset = _mix_dataset_with_synthetic_scene()
        points = _probe_points()
        scene_flag = torch.tensor([0, 1], dtype=torch.long)

        is_penetrating, nearest_free_points = dataset.get_nearest_free_voxel(points, scene_flag)

        self.assertEqual(tuple(is_penetrating.shape), (2, 1, 4))
        self.assertIs(is_penetrating.dtype, torch.bool)
        self.assertEqual(tuple(nearest_free_points.shape), (2, 1, 4, 3))
        self.assertIs(nearest_free_points.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(nearest_free_points).all()))

    def test_call_flags_the_occupied_voxel_of_each_scene(self):
        dataset = _mix_dataset_with_synthetic_scene()
        points = _probe_points()
        scene_flag = torch.tensor([0, 1], dtype=torch.long)

        is_penetrating, nearest_free_points = dataset.get_nearest_free_voxel(points, scene_flag)

        # Column 0 is that scene's occupied voxel; column 1 is a free voxel.
        self.assertTrue(bool(is_penetrating[0, 0, 0]))
        self.assertTrue(bool(is_penetrating[1, 0, 0]))
        self.assertFalse(bool(is_penetrating[0, 0, 1]))
        self.assertFalse(bool(is_penetrating[1, 0, 1]))
        # Non-penetrating points must come back untouched.
        torch.testing.assert_close(nearest_free_points[:, :, 1:, :], points[:, :, 1:, :])
        # The displaced point must land in a genuinely free voxel of its scene.
        voxel_size = 1.0 / GRID_DIM
        for batch in range(2):
            moved = nearest_free_points[batch, 0, 0]
            voxel = torch.div(moved, voxel_size).long().clamp(0, GRID_DIM - 1)
            self.assertEqual(int(dataset.scene_occ[batch][tuple(voxel.tolist())]), 0)
            self.assertGreater(float(torch.linalg.vector_norm(moved - points[batch, 0, 0])), 0.0)

    def test_scene_flag_selects_per_scene_occupancy_not_a_shared_grid(self):
        # Swapping the flags must move the penetration to the other column set,
        # otherwise the query is ignoring scene identity.
        dataset = _mix_dataset_with_synthetic_scene()
        points = _probe_points()

        straight, _ = dataset.get_nearest_free_voxel(points, torch.tensor([0, 1]))
        swapped, _ = dataset.get_nearest_free_voxel(points, torch.tensor([1, 0]))

        self.assertTrue(bool(straight[0, 0, 0]))
        self.assertFalse(bool(swapped[0, 0, 0]))
        self.assertTrue(bool(straight[1, 0, 0]))
        self.assertFalse(bool(swapped[1, 0, 0]))

    def test_out_of_bounds_points_are_reported_as_penetrating_and_not_moved(self):
        dataset = _mix_dataset_with_synthetic_scene()
        points = _probe_points()
        scene_flag = torch.tensor([0, 1], dtype=torch.long)

        is_penetrating, nearest_free_points = dataset.get_nearest_free_voxel(points, scene_flag)

        # Columns 2 and 3 are outside the grid: the body clamps them to voxel 0
        # and, because ``valid_mask`` is false, never displaces them.
        torch.testing.assert_close(nearest_free_points[:, :, 2:, :], points[:, :, 2:, :])
        self.assertEqual(tuple(is_penetrating[:, :, 2:].shape), (2, 1, 2))


class DirectVersusMaterializedTests(unittest.TestCase):
    """087848f changed how occupancy is indexed.  If the two bodies disagree,
    guidance geometry changed silently."""

    def test_the_two_bodies_agree_numerically_on_a_mix_dataset(self):
        points = _probe_points()
        scene_flag = torch.tensor([0, 1], dtype=torch.long)

        lazy = _mix_dataset_with_synthetic_scene(lazy=True)
        direct = InfBaGelDataset._get_nearest_free_voxel_direct(lazy, points, scene_flag)
        materialized = InfBaGelDataset._get_nearest_free_voxel_materialized(
            lazy, points, scene_flag
        )

        torch.testing.assert_close(direct[0], materialized[0], rtol=0, atol=0)
        torch.testing.assert_close(direct[1], materialized[1], rtol=0, atol=0)

    def test_lazy_and_eager_occ_ref_agree(self):
        points = _probe_points()
        scene_flag = torch.tensor([0, 1], dtype=torch.long)

        lazy = _mix_dataset_with_synthetic_scene(lazy=True)
        eager = _mix_dataset_with_synthetic_scene(lazy=False)

        for left, right in zip(
            lazy.get_nearest_free_voxel(points, scene_flag),
            eager.get_nearest_free_voxel(points, scene_flag),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_prepared_query_resolves_scene_ids_once_and_matches_direct(self):
        dataset = _mix_dataset_with_synthetic_scene(lazy=False)
        points = _probe_points()
        scene_flag = torch.tensor([0, 1], dtype=torch.long)
        prepared = dataset.prepare_nearest_free_voxel(scene_flag)
        self.assertEqual(prepared.scene_ids, (0, 1))

        with patch.object(
            torch.Tensor, "tolist", wraps=torch.Tensor.tolist
        ) as tolist:
            before = tolist.call_count
            optimized = prepared(points, scene_flag)
            self.assertEqual(tolist.call_count, before)
        direct = InfBaGelDataset._get_nearest_free_voxel_direct(
            dataset, points, scene_flag
        )
        torch.testing.assert_close(optimized[0], direct[0], rtol=0, atol=0)
        torch.testing.assert_close(optimized[1], direct[1], rtol=0, atol=0)

    def test_direct_source_has_no_unique_scene_flag_cpu_conversion(self):
        source = (REPO / "code" / "datasets" / "infbagel.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch.unique(penetrating_scene_flags).tolist()", source)
        self.assertIn("prepared_scene_ids", source)


class BorrowedCallSiteGuard(unittest.TestCase):
    """Numerical agreement means only a structural check can pin *which* body the
    borrowed call site names."""

    def test_mix_dataset_names_the_direct_implementation_explicitly(self):
        source = (REPO / "code" / "datasets" / "infbagel_mix.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_nearest_free_voxel"
            for node in ast.walk(node)
            if isinstance(node, ast.Call)
        ]
        named = [
            node.func.attr
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "InfBaGelDataset"
        ]
        self.assertEqual(named, ["_get_nearest_free_voxel_direct"])


class PreflightTests(unittest.TestCase):
    """The evaluator's preflight must call the method, not merely find it."""

    def _evaluator(self):
        import test_infbagel_lingo_hsi as evaluator

        return evaluator

    def test_preflight_passes_on_a_working_mix_dataset(self):
        self._evaluator()._preflight_nearest_free_voxel(_mix_dataset_with_synthetic_scene())

    def test_preflight_consumes_no_random_numbers(self):
        # It runs after seed_everything and before the first episode, so any RNG
        # draw inside it would shift the unguided gate column.
        evaluator = self._evaluator()
        dataset = _mix_dataset_with_synthetic_scene()
        torch.manual_seed(4242)
        before = evaluator._capture_rng_state()
        evaluator._preflight_nearest_free_voxel(dataset)
        after = evaluator._capture_rng_state()
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(str(before["numpy"]), str(after["numpy"]))

    def test_preflight_reports_the_pre_fix_dispatch_failure_clearly(self):
        # Reproduce the shipped defect on one instance: the borrowed body
        # dispatching through self, which has no private implementation.
        evaluator = self._evaluator()
        dataset = _mix_dataset_with_synthetic_scene()
        dataset.get_nearest_free_voxel = (
            lambda points, scene_flag: InfBaGelDataset.get_nearest_free_voxel(
                dataset, points, scene_flag
            )
        )
        with self.assertRaises(RuntimeError) as raised:
            evaluator._preflight_nearest_free_voxel(dataset)
        message = str(raised.exception)
        self.assertIn("InfBaGelMixDataset", message)
        self.assertIn("AttributeError", message)
        self.assertIn("_get_nearest_free_voxel_direct", message)
        self.assertIsInstance(raised.exception.__cause__, AttributeError)


class RdsGatingTests(unittest.TestCase):
    """The guided cell skips the null-scene pass; the unguided cell must not shift."""

    def _evaluator(self):
        import test_infbagel_lingo_hsi as evaluator

        return evaluator

    def test_skipping_the_rewound_block_leaves_the_stream_where_running_it_does(self):
        evaluator = self._evaluator()

        def draw(run_block):
            torch.manual_seed(99)
            pre = evaluator._capture_rng_state()
            first = torch.randn(4)
            if run_block:
                with evaluator._rng_rewound(pre):
                    torch.randn(4)
            return first, torch.randn(4)

        ran, skipped = draw(True), draw(False)
        self.assertTrue(torch.equal(ran[0], skipped[0]))
        self.assertTrue(torch.equal(ran[1], skipped[1]))

    def test_scene_aggregation_tolerates_a_null_rds(self):
        evaluator = self._evaluator()
        records = {
            "s:0": {"scene_name": "s", "rds": None, "rds_max": None, "rds_available": False, "x": 2.0},
            "s:1": {"scene_name": "s", "rds": None, "rds_max": None, "rds_available": False, "x": 4.0},
        }
        summary = evaluator._aggregate_by_scene(records)
        means = summary["s"]["metrics_mean"]
        self.assertNotIn("rds", means)
        self.assertNotIn("rds_max", means)
        self.assertNotIn("rds_available", means)
        self.assertEqual(means["x"], 3.0)

    def test_scene_aggregation_still_averages_a_present_rds(self):
        evaluator = self._evaluator()
        records = {
            "s:0": {"scene_name": "s", "rds": 0.2, "rds_available": True},
            "s:1": {"scene_name": "s", "rds": 0.4, "rds_available": True},
        }
        means = evaluator._aggregate_by_scene(records)["s"]["metrics_mean"]
        self.assertAlmostEqual(means["rds"], 0.3, places=9)


if __name__ == "__main__":
    unittest.main()
