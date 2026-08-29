"""Pin HSI guidance routing without importing the asset-heavy evaluator.

Scene-only batches must select a seven-argument adapter that ignores every
object input, while object-bearing and mixed batches preserve the HOSI route.
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
    def test_guidance_off_returns_none_for_scene_and_object_batches(self):
        self.assertIsNone(guidance_loss.select_guidance_fn(False, torch.tensor([False])))
        self.assertIsNone(guidance_loss.select_guidance_fn(False, torch.tensor([True])))

    def test_scene_only_selects_hsi_adapter_not_hosi(self):
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([False, False]))
        self.assertIs(selected, guidance_loss.apply_hsi_guidance_fn)
        self.assertIsNot(selected, guidance_loss.apply_hosi_guidance_loss)

    def test_object_batch_preserves_hosi_route(self):
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([True, True]))
        self.assertIs(selected, guidance_loss.apply_hosi_guidance_loss)

    def test_mixed_batch_matches_model_any_gate(self):
        selected = guidance_loss.select_guidance_fn(True, torch.tensor([False, True]))
        self.assertIs(selected, guidance_loss.apply_hosi_guidance_loss)

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
