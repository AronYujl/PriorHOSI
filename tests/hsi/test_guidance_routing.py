"""Pin HSI guidance routing without importing the asset-heavy evaluator.

Scene-only batches must select a seven-argument adapter that ignores every
object input, while mixed batches must route each row independently.
The AST guard pins the lightweight selection at the evaluator call site.
"""

import ast
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import guidance_loss


class GuidanceSelectionTests(unittest.TestCase):
    def test_guidance_off_returns_none_for_all_batch_compositions(self):
        self.assertIsNone(guidance_loss.select_guidance_fn(False, torch.tensor([False])))
        self.assertIsNone(guidance_loss.select_guidance_fn(False, torch.tensor([True])))
        self.assertIsNone(
            guidance_loss.select_guidance_fn(False, torch.tensor([False, True]))
        )

    def test_scene_only_selects_hsi_adapter_not_hosi(self):
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([False, False]))
        self.assertIs(selected, guidance_loss.apply_hsi_guidance_fn)
        self.assertIsNot(selected, guidance_loss.apply_hosi_guidance_loss)

    def test_mesh_route_is_explicit_and_legacy_signature_rejects_misordered_args(self):
        human = torch.zeros(1, 1, 24, 3)
        rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 1, 22, 1, 1)
        local = rotations.reshape(1, 22, 3, 3)
        with self.assertRaises(TypeError):
            guidance_loss.apply_hsi_guidance_loss(
                human,
                rotations,
                local,
                torch.tensor([0]),
                lambda points, flags: (None, points),
                object(),
                object(),
            )
        with self.assertRaises(TypeError):
            guidance_loss.apply_hsi_mesh_guidance_loss(
                human,
                rotations,
                local,
                torch.tensor([0]),
                lambda points, flags: (None, points),
                object(),
            )

    def test_object_batch_preserves_hosi_route(self):
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([True, True]))
        self.assertIs(selected, guidance_loss.apply_hosi_guidance_loss)

    def test_mixed_batch_selects_a_per_row_route(self):
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([False, True]))
        self.assertIsNot(selected, guidance_loss.apply_hsi_guidance_fn)
        self.assertIsNot(selected, guidance_loss.apply_hosi_guidance_loss)

    def test_mixed_batch_routes_finite_rows_and_gradients_independently(self):
        human_jnts = torch.zeros(2, 1, 24, 3)
        human_jnts[0, 0, 22] = torch.tensor([0.2, 0.5, 0.0])
        human_jnts[0, 0, 23] = torch.tensor([0.3, 0.5, 0.0])
        human_jnts[1, 0, 22] = torch.tensor([0.4, 0.5, 0.0])
        human_jnts[1, 0, 23] = torch.tensor([0.5, 0.5, 0.0])
        human_jnts.requires_grad_()
        obj_verts = torch.tensor(
            [
                [
                    [[float("nan"), float("nan"), float("nan")],
                     [float("nan"), float("nan"), float("nan")]]
                ],
                [[[0.0, 0.3, 0.0], [0.1, 0.3, 0.0]]],
            ],
            requires_grad=True,
        )
        pred_seq_com_pos = torch.tensor(
            [[[float("nan"), float("nan"), float("nan")]], [[0.0, 0.0, 0.0]]]
        )
        pred_obj_rot_mat = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)
        pred_obj_rot_mat[0] = float("nan")
        contact_labels = torch.tensor(
            [[[float("nan")] * 4], [[0.0, 0.0, 0.0, 0.0]]]
        )
        scene_flag = torch.tensor([3, 4])

        def nearest_free_voxel(points, flags):
            del flags
            return torch.zeros(points.shape[:-1], dtype=torch.bool), torch.zeros_like(points)

        selected = guidance_loss.select_guidance_fn(True, torch.tensor([False, True]))
        mixed_loss = selected(
            human_jnts,
            obj_verts,
            pred_seq_com_pos,
            pred_obj_rot_mat,
            contact_labels,
            scene_flag,
            nearest_free_voxel,
        )
        expected = guidance_loss.apply_hsi_guidance_loss(
            human_jnts.detach()[:1], scene_flag[:1], nearest_free_voxel
        ) + guidance_loss.apply_hosi_guidance_loss(
            human_jnts.detach()[1:],
            obj_verts.detach()[1:],
            pred_seq_com_pos[1:],
            pred_obj_rot_mat[1:],
            contact_labels[1:],
            scene_flag[1:],
            nearest_free_voxel,
        )
        self.assertTrue(torch.isfinite(mixed_loss))
        self.assertTrue(
            torch.allclose(mixed_loss.detach(), expected, atol=0.0, rtol=0.0)
        )

        mixed_loss.backward()
        self.assertTrue(torch.isfinite(human_jnts.grad).all())
        self.assertGreater(float(human_jnts.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(human_jnts.grad[1].abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(obj_verts.grad[1]).all())
        self.assertTrue(
            torch.equal(obj_verts.grad[0], torch.zeros_like(obj_verts.grad[0]))
        )

    def test_scene_adapter_ignores_poisoned_object_arguments(self):
        human_jnts = torch.zeros(1, 2, 24, 3)
        human_jnts[..., 1] = 0.5
        scene_flag = torch.tensor([3])

        def nearest_free_voxel(points, flags):
            nearest = points.clone()
            nearest[..., 1] += flags.to(points.dtype).reshape(-1, 1, 1) / 10
            return torch.ones(points.shape[:-1], dtype=torch.bool), nearest

        nan = float("nan")
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([False]))
        adapted = selected(
            human_jnts,
            torch.full((1, 2, 4, 3), nan),
            torch.full((1, 2, 3), nan),
            torch.full((1, 2, 3, 3), nan),
            torch.full((1, 2, 4), nan),
            scene_flag,
            nearest_free_voxel,
        )
        core = guidance_loss.apply_hsi_guidance_loss(
            human_jnts, scene_flag, nearest_free_voxel
        )
        self.assertTrue(torch.isfinite(adapted))
        self.assertTrue(torch.equal(adapted, core))


class GuidanceCallSiteTests(unittest.TestCase):
    def test_sample_step_selects_guidance_instead_of_hardcoding_hosi(self):
        source = (REPO / "code" / "test_infbagel_hosi.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        sample_step = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sample_step"
        )
        assignments = [
            node
            for node in ast.walk(sample_step)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "guidance_fn" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Name)
        self.assertEqual(value.func.id, "select_guidance_fn")


if __name__ == "__main__":
    unittest.main()
