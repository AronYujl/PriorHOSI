"""Unit tests for the reduced exact SMPL-X body-proxy arithmetic."""

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.body_proxy import (  # noqa: E402
    AREA512_COUNT,
    BODY_PROXY_ASSET_SHA256,
    BODY_PROXY_ASSET_SIZE_BYTES,
    AREA512_INDEX_RAW_INT64_SHA256,
    AREA512_INDEX_SHA256,
    load_proxy_tables,
    proxy_points_from_tables,
)


class ExactProxyArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.joints = torch.zeros(1, 1, 22, 3)
        self.joints[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
        self.joints[0, 0, 1] = torch.tensor([4.0, 5.0, 6.0])
        self.rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 1, 22, 1, 1)
        self.weights = torch.zeros(2, 22)
        self.weights[0, 0] = 1.0
        self.weights[1, 0] = 0.5
        self.weights[1, 1] = 0.5
        self.offsets = torch.zeros(2, 22, 3)
        self.offsets[0, 0] = torch.tensor([0.1, 0.2, 0.3])
        self.offsets[1, 0] = torch.tensor([0.2, 0.0, 0.0])
        self.offsets[1, 1] = torch.tensor([0.0, 0.4, 0.0])
        self.posedirs = torch.zeros(2, 3, 189)

    def test_reduced_lbs_matches_weighted_joint_transform(self):
        points = proxy_points_from_tables(
            self.joints,
            self.rotations,
            None,
            self.weights,
            self.offsets,
            self.posedirs,
        )
        expected = torch.tensor([[[[1.1, 2.2, 3.3], [2.6, 3.7, 4.5]]]])
        self.assertTrue(torch.allclose(points, expected, atol=1e-6, rtol=0.0))

    def test_pose_blend_shapes_are_added_after_lbs(self):
        local = self.rotations.clone()
        local[0, 0, 1, 0, 0] = 1.2
        self.posedirs[0, 0, 0] = 0.25
        points = proxy_points_from_tables(
            self.joints,
            self.rotations,
            local.reshape(1, 22, 3, 3),
            self.weights,
            self.offsets,
            self.posedirs,
        )
        expected = torch.tensor([[[[1.15, 2.2, 3.3], [2.6, 3.7, 4.5]]]])
        self.assertTrue(torch.allclose(points, expected, atol=1e-6, rtol=0.0))

    @staticmethod
    def _materialized_reference(joints, global_rotations, local_rotations, weights, offsets, posedirs):
        batch, steps = joints.shape[:2]
        points = joints[..., :22, :]
        weighted_offsets = weights[:, :, None] * offsets
        base = torch.einsum("nk,btkj->btnj", weights, points)
        base = base + torch.einsum("btkij,nkj->btni", global_rotations, weighted_offsets)
        eye = torch.eye(3, dtype=joints.dtype).reshape(1, 1, 1, 3, 3)
        theta = (local_rotations[:, :, 1:22] - eye).reshape(batch, steps, 189)
        blend_delta = torch.einsum("nij,btj->btni", posedirs, theta)
        # Faithful pre-remediation reference: retain the joint axis in the
        # [B,T,N,22,3,3] weighted rotation table before applying the blend delta.
        rotated_weights = (
            weights.reshape(1, 1, weights.shape[0], weights.shape[1], 1, 1)
            * global_rotations.unsqueeze(2)
        )
        rotated_delta = torch.matmul(
            rotated_weights,
            blend_delta.unsqueeze(3).unsqueeze(-1),
        ).squeeze(-1)
        return base + rotated_delta.sum(dim=3)

    def test_algebraic_contraction_matches_materialized_forward_and_gradients(self):
        tables = load_proxy_tables()
        weights = torch.from_numpy(tables.weights)
        offsets = torch.from_numpy(tables.offsets)
        posedirs = torch.from_numpy(tables.posedirs)
        generator = torch.Generator().manual_seed(20260827)
        shape = (2, 3)
        joints_old = torch.randn(*shape, 24, 3, generator=generator, requires_grad=True)
        global_old = torch.randn(*shape, 22, 3, 3, generator=generator, requires_grad=True)
        local_old = (
            torch.eye(3).reshape(1, 1, 1, 3, 3)
            + 0.2 * torch.randn(*shape, 22, 3, 3, generator=generator)
        ).requires_grad_()
        joints_new = joints_old.detach().clone().requires_grad_()
        global_new = global_old.detach().clone().requires_grad_()
        local_new = local_old.detach().clone().requires_grad_()

        reference = self._materialized_reference(
            joints_old, global_old, local_old, weights, offsets, posedirs
        )
        optimized = proxy_points_from_tables(
            joints_new, global_new, local_new, weights, offsets, posedirs
        )
        reference_loss = reference.square().mean()
        optimized_loss = optimized.square().mean()
        reference_grads = torch.autograd.grad(
            reference_loss, (joints_old, global_old, local_old)
        )
        optimized_grads = torch.autograd.grad(
            optimized_loss, (joints_new, global_new, local_new)
        )
        forward_max = float((reference - optimized).abs().max())
        gradient_max = max(
            float((old - new).abs().max())
            for old, new in zip(reference_grads, optimized_grads)
        )
        self.assertLessEqual(forward_max, 2e-6, "forward max deviation=%g" % forward_max)
        self.assertLessEqual(gradient_max, 2e-6, "gradient max deviation=%g" % gradient_max)

    def test_runtime_uses_the_frozen_derived_asset_contract(self):
        path = REPO / "code" / "priors" / "hsi" / "assets" / "body_proxy_area512.npz"
        self.assertEqual(path.stat().st_size, BODY_PROXY_ASSET_SIZE_BYTES)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), BODY_PROXY_ASSET_SHA256)
        tables = load_proxy_tables()
        self.assertEqual(tables.weights.shape, (512, 22))
        self.assertEqual(tables.offsets.shape, (512, 22, 3))
        self.assertEqual(tables.posedirs.shape, (512, 3, 189))
        self.assertFalse(hasattr(tables, "indices"))
        with self.assertRaises(ValueError):
            load_proxy_tables(gender="female")


class FrozenArea512AssetTests(unittest.TestCase):
    def test_index_is_the_frozen_sorted_int64_asset(self):
        path = REPO / "code" / "priors" / "hsi" / "assets" / "idx_area512.npy"
        self.assertEqual(path.stat().st_size, 4224)
        import hashlib

        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), AREA512_INDEX_SHA256)
        index = np.load(path, allow_pickle=False)
        self.assertEqual(index.shape, (AREA512_COUNT,))
        self.assertEqual(index.dtype, np.int64)
        self.assertTrue(np.all(index[:-1] < index[1:]))
        self.assertEqual(
            hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest(),
            AREA512_INDEX_RAW_INT64_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
