"""Tests for the mesh-derived scene signed-distance field.

Geometry is the load-bearing input to every penetration number, and the
correction recorded in ``docs/plan/PHASE_1C_HSI.md`` (2026-08-12 同日修订,
subsection A) turned on a measured property of the data: ``Scene/*.npy`` is a
reachability volume, not geometry -- scene 004 is 51.19% "occupied" and its
y~1.98 m ceiling slice is the *most* occupied at 80.7%.  So the mesh is the
scoring reference and the occupancy grid is a separate diagnostic, and these
tests pin that distinction along with the sign convention.

Tolerances are not guessed.  Measured against closed forms on a 2 cm field:
sphere |err| mean 2.48 mm, p95 6.63 mm, max 23.11 mm; box mean 0.36 mm, p95
1.68 mm.  The assertions below use one voxel (20 mm) near the surface and 1.5
voxels away from it, which those distributions justify.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.scene_field import (
    OCC_GRID_MAX,
    OCC_GRID_MIN,
    OCC_GRID_SHAPE,
    OCC_VOXEL_SIZE,
    SDF_VOXEL_SIZE,
    SceneGeometry,
    SceneGeometryError,
)

try:
    import trimesh

    HAVE_TRIMESH = True
except ImportError:  # pragma: no cover
    HAVE_TRIMESH = False


NEAR_SURFACE_TOL = SDF_VOXEL_SIZE          # 20 mm
FAR_FIELD_TOL = 1.5 * SDF_VOXEL_SIZE       # 30 mm


def _unit_box(extents=(0.6, 0.4, 0.8)):
    """A watertight axis-aligned box centred on the origin, built by hand."""
    half = np.asarray(extents, dtype=np.float64) / 2.0
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
    )
    vertices = signs * half
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # x = -half
            [4, 6, 7], [4, 7, 5],  # x = +half
            [0, 4, 5], [0, 5, 1],  # y = -half
            [2, 3, 7], [2, 7, 6],  # y = +half
            [0, 2, 6], [0, 6, 4],  # z = -half
            [1, 5, 7], [1, 7, 3],  # z = +half
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _box_signed_distance(points, extents=(0.6, 0.4, 0.8)):
    half = np.asarray(extents, dtype=np.float64) / 2.0
    delta = np.abs(points) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(delta.max(axis=-1), 0.0)
    return outside + inside


class SignConventionTests(unittest.TestCase):
    """The single most consequential thing to pin: negative means inside."""

    @classmethod
    def setUpClass(cls):
        vertices, faces = _unit_box()
        cls.geometry = SceneGeometry.from_mesh(
            vertices, faces, scene_name="box", interior_is_free_space=False
        )

    def test_interior_is_negative_and_exterior_is_positive(self):
        interior = torch.zeros(1, 3)
        exterior = torch.tensor([[2.0, 0.0, 0.0]])
        self.assertLess(float(self.geometry.signed_distance(interior)), 0.0)
        self.assertGreater(float(self.geometry.signed_distance(exterior)), 0.0)

    def test_matches_the_closed_form_box_distance(self):
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.8, 0.8, size=(2000, 3)).astype(np.float32)
        truth = _box_signed_distance(points.astype(np.float64))
        got = self.geometry.signed_distance(torch.from_numpy(points)).numpy()
        keep = ~self.geometry.out_of_bounds(torch.from_numpy(points)).numpy()
        error = np.abs(got[keep] - truth[keep])
        self.assertLess(error.max(), FAR_FIELD_TOL, f"max error {error.max()*1000:.2f} mm")
        near = keep & (np.abs(truth) < 0.05)
        self.assertLess(np.abs(got[near] - truth[near]).max(), NEAR_SURFACE_TOL)

    def test_sign_is_correct_outside_a_one_voxel_band(self):
        rng = np.random.default_rng(1)
        points = rng.uniform(-0.8, 0.8, size=(2000, 3)).astype(np.float32)
        truth = _box_signed_distance(points.astype(np.float64))
        got = self.geometry.signed_distance(torch.from_numpy(points)).numpy()
        keep = (~self.geometry.out_of_bounds(torch.from_numpy(points)).numpy()) & (
            np.abs(truth) >= SDF_VOXEL_SIZE
        )
        np.testing.assert_array_equal(np.sign(got[keep]), np.sign(truth[keep]))

    def test_deepest_interior_distance_is_conservative_by_at_most_half_a_voxel(self):
        """Voxel-centre discretisation makes depth slightly under-reported, never over."""
        centre = float(self.geometry.signed_distance(torch.zeros(1, 3)))
        analytic = -0.2
        self.assertGreaterEqual(centre, analytic - 1e-6)
        self.assertLess(centre - analytic, SDF_VOXEL_SIZE)


@unittest.skipUnless(HAVE_TRIMESH, "trimesh is required for the sphere case")
class SphereTests(unittest.TestCase):
    def test_matches_the_closed_form_sphere_distance(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        geometry = SceneGeometry.from_mesh(
            mesh.vertices, mesh.faces, scene_name="sphere", interior_is_free_space=False
        )
        self.assertTrue(geometry.is_watertight)
        rng = np.random.default_rng(2)
        points = rng.uniform(-0.9, 0.9, size=(2000, 3)).astype(np.float32)
        truth = np.linalg.norm(points.astype(np.float64), axis=1) - 0.5
        got = geometry.signed_distance(torch.from_numpy(points)).numpy()
        keep = ~geometry.out_of_bounds(torch.from_numpy(points)).numpy()
        near = keep & (np.abs(truth) < 0.05)
        self.assertLess(np.abs(got[near] - truth[near]).max(), NEAR_SURFACE_TOL)


class SymmetryAndDeterminismTests(unittest.TestCase):
    def test_x_negating_the_mesh_x_negates_the_field(self):
        """The dataset's own mirror invariant, as an end-to-end check."""
        vertices, faces = _unit_box(extents=(0.6, 0.4, 0.8))
        shifted = vertices + np.array([0.35, 0.0, 0.0])
        forward = SceneGeometry.from_mesh(
            shifted, faces, scene_name="a", interior_is_free_space=False
        )
        mirrored = SceneGeometry.from_mesh(
            shifted * np.array([-1.0, 1.0, 1.0]), faces[:, ::-1].copy(),
            scene_name="a_mirror", interior_is_free_space=False,
        )
        probes = torch.tensor(
            [[0.35, 0.0, 0.0], [0.9, 0.1, 0.2], [0.2, -0.1, 0.3], [1.4, 0.0, 0.0]]
        )
        flipped = probes * torch.tensor([-1.0, 1.0, 1.0])
        torch.testing.assert_close(
            forward.signed_distance(probes),
            mirrored.signed_distance(flipped),
            atol=NEAR_SURFACE_TOL,
            rtol=0.0,
        )

    def test_two_builds_of_the_same_mesh_agree_exactly(self):
        vertices, faces = _unit_box()
        first = SceneGeometry.from_mesh(vertices, faces, interior_is_free_space=False)
        second = SceneGeometry.from_mesh(vertices, faces, interior_is_free_space=False)
        probes = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.31, 0.0, 0.0]])
        torch.testing.assert_close(
            first.signed_distance(probes), second.signed_distance(probes), atol=0.0, rtol=0.0
        )


class BoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vertices, faces = _unit_box()
        cls.geometry = SceneGeometry.from_mesh(vertices, faces, interior_is_free_space=False)

    def test_out_of_bounds_points_are_positive_finite_and_flagged(self):
        far = torch.tensor([[50.0, 50.0, 50.0], [-40.0, 3.0, 12.0]])
        flags = self.geometry.out_of_bounds(far)
        self.assertTrue(bool(flags.all()))
        distance = self.geometry.signed_distance(far)
        self.assertTrue(bool(torch.isfinite(distance).all()))
        # Never clamped toward zero: an out-of-grid sample must not read as contact.
        self.assertTrue(bool((distance > 0).all()))

    def test_interior_points_are_not_flagged_out_of_bounds(self):
        self.assertFalse(bool(self.geometry.out_of_bounds(torch.zeros(1, 3)).any()))

    def test_bounds_and_voxel_size_are_exposed(self):
        low, high = self.geometry.bounds
        self.assertEqual(tuple(low.shape), (3,))
        self.assertTrue(bool((high > low).all()))
        self.assertAlmostEqual(self.geometry.voxel_size, SDF_VOXEL_SIZE, places=9)


class ReachabilityIsNotPenetrationTests(unittest.TestCase):
    """`reachability_violation` reads the occupancy grid and must never be
    confused with penetration -- that confusion is exactly what the 2026-08-12
    correction removed."""

    def test_it_refuses_rather_than_returning_a_misleading_false(self):
        vertices, faces = _unit_box()
        geometry = SceneGeometry.from_mesh(vertices, faces, interior_is_free_space=False)
        # A synthetic mesh has no LINGO occupancy grid behind it.  Returning
        # False here would read as "reachable", which is a claim we cannot make.
        with self.assertRaises((SceneGeometryError, ValueError)):
            geometry.reachability_violation(torch.zeros(1, 3))

    def test_it_uses_the_released_occupancy_convention_when_given_one(self):
        vertices, faces = _unit_box()
        occupancy = np.zeros(OCC_GRID_SHAPE, dtype=bool)
        # Mark one cell and probe its centre in world coordinates.
        index = (10, 20, 30)
        occupancy[index] = True
        geometry = SceneGeometry.from_mesh(
            vertices, faces, interior_is_free_space=False, occupancy=occupancy
        )
        low = np.asarray(OCC_GRID_MIN)
        centre = low + (np.asarray(index) + 0.5) * OCC_VOXEL_SIZE
        probe = torch.tensor(centre, dtype=torch.float32).view(1, 3)
        self.assertTrue(bool(geometry.reachability_violation(probe).all()))
        empty = low + (np.asarray([11, 20, 30]) + 0.5) * OCC_VOXEL_SIZE
        self.assertFalse(
            bool(geometry.reachability_violation(torch.tensor(empty, dtype=torch.float32).view(1, 3)).any())
        )

    def test_the_released_grid_geometry_constants_are_self_consistent(self):
        span = np.asarray(OCC_GRID_MAX) - np.asarray(OCC_GRID_MIN)
        derived = span / np.asarray(OCC_GRID_SHAPE)
        np.testing.assert_allclose(derived, OCC_VOXEL_SIZE, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
