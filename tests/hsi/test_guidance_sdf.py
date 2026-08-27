"""Tests for the P16-GQ floor split and mesh-SDF guidance routing."""

import gc
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import guidance_loss  # noqa: E402
from datasets.infbagel_mix import InfBaGelMixDataset  # noqa: E402
from priors.hsi.scene_field import SceneGeometry  # noqa: E402
from priors.hsi import metrics as sealed_metrics  # noqa: E402


class PlaneGeometry:
    def __init__(self):
        self.calls = []

    def signed_distance(self, points):
        self.calls.append(tuple(points.shape))
        # Negative below y=0.5, positive above it.
        return points[..., 1] - 0.5


class GuidanceSdfTests(unittest.TestCase):
    def test_frozen_floor_and_margin_constants(self):
        self.assertEqual(guidance_loss.FLOOR_EXCLUSION_HEIGHT_M, 0.02)
        self.assertEqual(guidance_loss.SDF_MARGIN_M, 0.0)
        self.assertEqual(
            guidance_loss.FLOOR_EXCLUSION_HEIGHT_M,
            sealed_metrics.FLOOR_EXCLUSION_HEIGHT_M,
        )

    def _nearest(self, points, flags):
        return torch.ones(points.shape[:-1], dtype=torch.bool), points + torch.tensor(
            [0.0, 0.1, 0.0], dtype=points.dtype
        )

    def test_mesh_term_is_above_floor_and_original_voxel_term_is_below(self):
        human = torch.zeros(1, 1, 24, 3)
        global_rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 1, 22, 1, 1)
        local_rotations = global_rotations.reshape(1, 22, 3, 3)
        flags = torch.tensor([0])
        proxy = torch.tensor([[[[0.0, 0.01, 0.0], [0.0, 0.40, 0.0]]]])
        cfg = SimpleNamespace(hsi_guidance_sdf_proxy="area512", hsi_guidance_sdf_weight=4879)

        geometry = PlaneGeometry()
        with patch.object(guidance_loss, "proxy_points", return_value=proxy) as make_proxy:
            loss = guidance_loss.apply_hsi_mesh_guidance_loss(
                human,
                global_rotations,
                local_rotations,
                scene_flag=flags,
                get_nearest_free_voxel=self._nearest,
                geometry=geometry,
                proxy="area512",
                sdf_weight=cfg.hsi_guidance_sdf_weight,
            )

        # Every joint is below 2 cm, so the voxel term is
        # 20000 * (24 * 0.1^2) / (24 * 3) = 66.6667.  Only the second proxy point is
        # above the split and is inside the plane by 0.1 m:
        # 4879 * (0.1^2 / 2) = 24.395.
        self.assertTrue(torch.allclose(loss, torch.tensor(91.061666), atol=1e-5, rtol=0.0))
        make_proxy.assert_called_once_with(
            human, global_rotations, local_rotations, proxy="area512"
        )
        self.assertEqual(geometry.calls, [(1, 1, 1, 3)])

    def test_batch_two_preserves_legacy_voxel_and_mesh_mean_normalization(self):
        human = torch.zeros(2, 1, 24, 3)
        flags = torch.tensor([0, 1])
        nearest_points = human + torch.tensor([0.0, 0.1, 0.0])
        global_rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(2, 1, 22, 1, 1)
        local_rotations = global_rotations.reshape(2, 22, 3, 3)
        proxy = torch.tensor(
            [
                [[[0.0, 0.01, 0.0], [0.0, 0.40, 0.0]]],
                [[[0.0, 0.01, 0.0], [0.0, 0.40, 0.0]]],
            ]
        )
        cfg = SimpleNamespace(hsi_guidance_sdf_proxy="area512", hsi_guidance_sdf_weight=4879)
        expected_legacy = (human - nearest_points).pow(2).mean() * 20000.0

        legacy = guidance_loss.apply_hsi_guidance_loss(human, flags, self._nearest)
        disabled = guidance_loss.apply_hsi_scene_guidance_loss(
            human,
            global_rotations,
            local_rotations,
            scene_flag=flags,
            get_nearest_free_voxel=self._nearest,
            geometry=PlaneGeometry(),
            cfg=SimpleNamespace(hsi_guidance_sdf_proxy=None, hsi_guidance_sdf_weight=0),
        )
        self.assertTrue(torch.equal(legacy, disabled))
        self.assertTrue(torch.allclose(disabled, expected_legacy, atol=0.0, rtol=0.0))

        with patch.object(guidance_loss, "proxy_points", return_value=proxy):
            geometry = PlaneGeometry()
            enabled = guidance_loss.apply_hsi_mesh_guidance_loss(
                human,
                global_rotations,
                local_rotations,
                scene_flag=flags,
                get_nearest_free_voxel=self._nearest,
                geometry=geometry,
                proxy="area512",
                sdf_weight=cfg.hsi_guidance_sdf_weight,
            )

        expected_ground = 20000.0 * (human - nearest_points).pow(2).sum() / human.numel()
        expected_mesh = torch.tensor(4879.0 * (0.1**2 / 2.0))
        self.assertTrue(
            torch.allclose(enabled, expected_ground + expected_mesh, atol=0.0, rtol=0.0)
        )
        self.assertEqual(geometry.calls, [(1, 1, 1, 3), (1, 1, 1, 3)])

    def test_mesh_skips_an_all_below_floor_proxy_without_changing_zero_term(self):
        human = torch.zeros(1, 1, 24, 3, requires_grad=True)
        rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 1, 22, 1, 1)
        local = rotations.reshape(1, 22, 3, 3)
        proxy = torch.zeros(1, 1, 3, 3)
        proxy[..., 1] = 0.01
        geometry = PlaneGeometry()
        with patch.object(guidance_loss, "proxy_points", return_value=proxy):
            loss = guidance_loss.apply_hsi_mesh_guidance_loss(
                human,
                rotations,
                local,
                scene_flag=torch.tensor([0]),
                get_nearest_free_voxel=self._nearest,
                geometry=geometry,
                sdf_weight=4879,
            )
        self.assertEqual(geometry.calls, [])
        self.assertAlmostEqual(float(loss), 66.66666666666667, places=5)
        loss.backward()
        self.assertTrue(torch.equal(human.grad, torch.zeros_like(human)))

    def test_disabled_mesh_config_is_bitwise_legacy(self):
        human = torch.zeros(1, 1, 24, 3)
        human[..., 1] = 0.5
        flags = torch.tensor([2])
        legacy = guidance_loss.apply_hsi_guidance_loss(human, flags, self._nearest)
        explicit = guidance_loss.apply_hsi_scene_guidance_loss(
            human,
            torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 1, 22, 1, 1),
            torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 22, 1, 1),
            scene_flag=flags,
            get_nearest_free_voxel=self._nearest,
            geometry=PlaneGeometry(),
            cfg=SimpleNamespace(hsi_guidance_sdf_proxy=None, hsi_guidance_sdf_weight=0),
        )
        self.assertTrue(torch.equal(legacy, explicit))


class NearFloorGuidanceEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _reference(human_jnts, scene_flag, nearest):
        _, nearest_free_points = nearest(human_jnts, scene_flag)
        near_floor = (
            human_jnts[..., 1] < guidance_loss.FLOOR_EXCLUSION_HEIGHT_M
        ).detach().unsqueeze(-1)
        delta = torch.where(
            near_floor,
            human_jnts - nearest_free_points,
            torch.zeros_like(human_jnts),
        )
        return 20000.0 * delta.pow(2).sum() / human_jnts.numel()

    @staticmethod
    def _nearest_factory(calls):
        def nearest(points, flags):
            calls.append((tuple(points.shape), tuple(flags.reshape(-1).tolist())))
            penetrating = points[..., 0] > 0
            offset = torch.tensor([0.03, 0.04, -0.02], dtype=points.dtype)
            target = torch.where(
                penetrating.unsqueeze(-1), points + offset, points
            )
            return penetrating, target

        return nearest

    def test_no_eligible_points_skips_query_and_has_zero_gradient(self):
        human = torch.zeros(2, 2, 4, 3)
        human[..., 1] = 0.5
        human[0, 0, 0, 0] = float("nan")
        human.requires_grad_()
        flags = torch.tensor([7, 8])
        calls = []
        nearest = self._nearest_factory(calls)
        reference = self._reference(human.detach().clone().requires_grad_(), flags, nearest)
        calls.clear()
        optimized = guidance_loss._apply_hsi_near_floor_voxel_guidance_loss(
            human, flags, nearest
        )
        self.assertEqual(calls, [])
        self.assertEqual(float(optimized), 0.0)
        optimized.backward()
        self.assertTrue(torch.equal(human.grad, torch.zeros_like(human)))
        self.assertEqual(float(reference), 0.0)

    def test_mixed_heights_penetration_and_batch_alignment_match_reference(self):
        base = torch.zeros(2, 2, 4, 3)
        base[..., 1] = 0.5
        base[0, 0, 0, 1] = -0.01
        base[0, 0, 0, 0] = 0.25  # eligible and penetrating
        base[0, 0, 1, 1] = -0.01
        base[0, 0, 1, 0] = -0.25  # eligible and non-penetrating
        base[1, 1, 2, 1] = 0.01
        base[1, 1, 2, 0] = 0.4  # second batch row, eligible and penetrating
        reference_input = base.clone().requires_grad_()
        optimized_input = base.clone().requires_grad_()
        flags = torch.tensor([7, 8])

        reference_calls = []
        reference = self._reference(
            reference_input, flags, self._nearest_factory(reference_calls)
        )
        optimized_calls = []
        optimized = guidance_loss._apply_hsi_near_floor_voxel_guidance_loss(
            optimized_input, flags, self._nearest_factory(optimized_calls)
        )
        self.assertEqual(reference_calls, [((2, 2, 4, 3), (7, 8))])
        self.assertEqual(
            optimized_calls,
            [((1, 1, 2, 3), (7,)), ((1, 1, 1, 3), (8,))],
        )
        self.assertTrue(torch.equal(reference, optimized))
        reference.backward()
        optimized.backward()
        self.assertTrue(torch.equal(reference_input.grad, optimized_input.grad))


class SceneGeometryAccessorTests(unittest.TestCase):
    def setUp(self):
        self.dataset = InfBaGelMixDataset.__new__(InfBaGelMixDataset)
        self.dataset.hsi_mesh_root = Path("/checkout/mesh")
        self.dataset.lingo_dataset = SimpleNamespace(folder="/checkout/data/dataset")
        self.dataset.unified_scene_dict = {"010": 0, "018-1": 1}
        self.dataset._scene_name_by_flag = {0: "010", 1: "018-1"}
        self.dataset.unified_scene_source = {0: "lingo", 1: "lingo"}

    def test_accessor_delegates_ownership_to_scene_geometry_lru(self):
        first = MagicMock(name="first_geometry")
        second = MagicMock(name="second_geometry")
        with patch(
            "priors.hsi.scene_field.SceneGeometry.from_scene",
            side_effect=[first, second, first, second],
        ) as from_scene:
            self.assertIs(self.dataset.scene_geometry(torch.tensor([0, 0])), first)
            self.assertIs(self.dataset.scene_geometry(torch.tensor([0])), second)
            mixed = self.dataset.scene_geometry(torch.tensor([0, 1]))

        self.assertEqual(mixed, (first, second))
        self.assertEqual(from_scene.call_count, 4)
        self.assertFalse(hasattr(self.dataset, "_hsi_scene_geometry"))
        from_scene.assert_any_call(
            "010",
            dataset_root=Path("/checkout/data/dataset"),
            mesh_root=Path("/checkout/mesh"),
            cache_dir=unittest.mock.ANY,
        )

    def test_dataset_does_not_retain_evicted_scene_geometry(self):
        """The dataset must not extend the four-scene process-wide LRU lifetime."""
        from priors.hsi import scene_field

        previous = SceneGeometry.cache_info()
        SceneGeometry.cache_clear()
        SceneGeometry.configure_cache(4)
        first_ref = None
        try:
            with tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                dataset_root = root / "dataset"
                mesh_root = root / "mesh"
                (dataset_root / "Scene").mkdir(parents=True)
                for index in range(5):
                    name = "scene-%d" % index
                    (mesh_root / name).mkdir(parents=True)
                    (mesh_root / name / "mesh_low.obj").write_text("", encoding="utf-8")
                    np.save(
                        dataset_root / "Scene" / (name + ".npy"),
                        np.zeros((1, 1, 1), dtype=np.bool_),
                    )

                dataset = InfBaGelMixDataset.__new__(InfBaGelMixDataset)
                dataset.hsi_mesh_root = mesh_root
                dataset.lingo_dataset = SimpleNamespace(folder=str(dataset_root))
                dataset.unified_scene_dict = {
                    "scene-%d" % index: index for index in range(5)
                }
                dataset._scene_name_by_flag = {
                    index: "scene-%d" % index for index in range(5)
                }
                dataset.unified_scene_source = {index: "lingo" for index in range(5)}

                field = np.zeros((2, 2, 2), dtype=np.float32)
                origin = np.zeros(3, dtype=np.float64)
                stats = {"watertight": 1.0}
                mesh_info = (
                    np.zeros((3, 3), dtype=np.float32),
                    np.zeros((1, 3), dtype=np.int64),
                    {"watertight": 1.0},
                )
                with patch.object(scene_field, "_validate_occupancy"), patch.object(
                    scene_field, "_mesh_sha256", side_effect=lambda path: path.parent.name
                ), patch.object(
                    scene_field, "_load_mesh", return_value=mesh_info
                ), patch.object(
                    scene_field, "_build_field", return_value=(field, origin, stats)
                ), patch.object(
                    SceneGeometry, "_read_cache", return_value=None
                ), patch.object(SceneGeometry, "_write_cache"):
                    first_ref = weakref.ref(dataset.scene_geometry(torch.tensor([0])))
                    for index in range(1, 5):
                        dataset.scene_geometry(torch.tensor([index]))

            gc.collect()
            self.assertEqual(SceneGeometry.cache_info()["size"], 4)
            self.assertIsNone(first_ref())
        finally:
            SceneGeometry.cache_clear()
            SceneGeometry.configure_cache(previous["maxsize"])


if __name__ == "__main__":
    unittest.main()
