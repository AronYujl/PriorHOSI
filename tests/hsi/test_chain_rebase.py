"""Mechanism checks for the preregistered D4-B chain-history rebase."""

import ast
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import rebase_model_output


class ChainRebaseArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros(1, 16, 232)
        self.x[:, 0, :84] = 1.0
        self.x[:, 1, :84] = 3.0
        self.x[:, 0, 84:216] = 2.0
        self.x[:, 1, 84:216] = 5.0
        self.output = torch.randn(1, 16, 232, generator=torch.Generator().manual_seed(4))

    def test_off_returns_the_same_object(self):
        self.assertIs(rebase_model_output(self.output, self.x, "off"), self.output)

    def test_c1_uses_fixed_history_and_preserves_future_differences(self):
        result = rebase_model_output(self.output, self.x, "c1")
        self.assertTrue(torch.allclose(result[:, 2, :84], torch.full((1, 84), 5.0)))
        self.assertTrue(
            torch.allclose(
                result[:, 3:, :84] - result[:, 2:-1, :84],
                self.output[:, 3:, :84] - self.output[:, 2:-1, :84],
                atol=1e-6,
            )
        )
        self.assertTrue(torch.equal(result[:, :, 84:], self.output[:, :, 84:]))

    def test_c2_uses_the_oracle_position(self):
        oracle = torch.full((1, 84), -7.0)
        result = rebase_model_output(self.output, self.x, "c2", oracle)
        self.assertTrue(torch.allclose(result[:, 2, :84], oracle, atol=1e-6))

    def test_c3_adds_the_rotation_rebase(self):
        result = rebase_model_output(self.output, self.x, "c3")
        self.assertTrue(torch.allclose(result[:, 2, :84], torch.full((1, 84), 5.0)))
        self.assertTrue(
            torch.allclose(result[:, 2, 84:216], torch.full((1, 132), 8.0), atol=1e-6)
        )
        self.assertTrue(torch.equal(result[:, :, 216:], self.output[:, :, 216:]))


class ChainRebaseCallSiteTests(unittest.TestCase):
    def test_diffusion_path_applies_rebase_before_trace_and_posterior(self):
        tree = ast.parse((REPO / "code" / "models" / "infbagel.py").read_text())
        sampler = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Sampler"
        )
        methods = {
            node.name: node for node in sampler.body if isinstance(node, ast.FunctionDef)
        }
        p_sample = methods["p_sample"]
        calls = [
            node.lineno for node in ast.walk(p_sample)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rebase_model_output"
        ]
        posterior = [
            node.lineno for node in ast.walk(p_sample)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "model_mean" for target in node.targets)
        ]
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0], posterior[0])
        self.assertNotIn("rebase_model_output", ast.dump(methods["cm_sample"]))


if __name__ == "__main__":
    unittest.main()
