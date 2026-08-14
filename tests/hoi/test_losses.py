"""HOIPrior objective and registered-gradient-probe contracts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hoi import diagnostics, losses as loss_module
from priors.hoi.diffusion import GaussianDiffusion
from priors.hoi.losses import hoi_training_losses


def _synthetic_inputs(batch_size=2):
    torch.manual_seed(42)
    prediction = (torch.randn(batch_size, 16, 232) * 0.15).requires_grad_()
    target = torch.randn(batch_size, 16, 232) * 0.15
    # Engage both semantic hand-contact channels on active frames.
    target[:, 2:10, 228] = 1.0
    target[:, 6:, 229] = 1.0
    rest_offsets = torch.randn(batch_size, 24, 3) * 0.08
    rest_offsets[:, 0] = 0.0
    parents = torch.zeros(24, dtype=torch.long)
    parents[0] = -1
    identity = torch.eye(3).expand(batch_size, 3, 3).clone()
    values = {
        "prediction": prediction,
        "target": target,
        "goals": torch.randn(batch_size, 9) * 0.1,
        "rest_human_offsets": rest_offsets,
        "parents_24": parents,
        "position_minimum": torch.tensor([-1.5, -1.0, -1.5]),
        "position_maximum": torch.tensor([1.5, 2.0, 1.5]),
        "object_minimum": torch.tensor([-1.0, -1.0, -1.0]),
        "object_maximum": torch.tensor([1.0, 1.0, 1.0]),
        "terminal_window": torch.tensor([1.0, 0.0])[:batch_size],
        "rest_object_points": torch.randn(batch_size, 12, 3) * 0.12,
        "world_to_local_rotation": identity,
        "object_rotation_reference": identity.clone(),
    }
    return values


def _call(values, detach_root, weight=3.0):
    return hoi_training_losses(
        values["prediction"],
        values["target"],
        values["goals"],
        values["rest_human_offsets"],
        values["parents_24"],
        values["position_minimum"],
        values["position_maximum"],
        values["object_minimum"],
        values["object_maximum"],
        values["terminal_window"],
        values["rest_object_points"],
        values["world_to_local_rotation"],
        values["object_rotation_reference"],
        hand_object_contact_weight=weight,
        hand_object_contact_detach_root=detach_root,
    )


class RootDetachLossTests(unittest.TestCase):
    def test_forward_values_and_rotation_gradients_are_bit_identical(self):
        values = _synthetic_inputs()
        attached = _call(values, False)
        detached = _call(values, True)
        self.assertTrue(torch.equal(attached["hand_object_contact_geometry"],
                                    detached["hand_object_contact_geometry"]))
        self.assertTrue(torch.equal(attached["total"], detached["total"]))
        attached_gradient, = torch.autograd.grad(
            attached["hand_object_contact_geometry"], values["prediction"],
            retain_graph=True,
        )
        detached_gradient, = torch.autograd.grad(
            detached["hand_object_contact_geometry"], values["prediction"],
            retain_graph=True,
        )
        self.assertGreater(float(attached_gradient[..., 0:3].norm()), 0.0)
        self.assertEqual(int(torch.count_nonzero(detached_gradient[..., 0:3])), 0)
        self.assertTrue(torch.equal(attached_gradient[..., 84:216],
                                    detached_gradient[..., 84:216]))
        self.assertGreater(float(attached_gradient[..., 84:216].norm()), 0.0)

    def test_default_path_has_one_fk_pass_and_detached_path_has_two(self):
        values = _synthetic_inputs()
        original = loss_module._fk_positions
        with mock.patch.object(loss_module, "_fk_positions", wraps=original) as wrapped:
            _call(values, False)
            self.assertEqual(wrapped.call_count, 1)
        with mock.patch.object(loss_module, "_fk_positions", wraps=original) as wrapped:
            _call(values, True)
            self.assertEqual(wrapped.call_count, 2)

    def test_fk_value_and_root_gradient_remain_attached(self):
        values = _synthetic_inputs()
        attached = _call(values, False)
        detached = _call(values, True)
        self.assertTrue(torch.equal(attached["fk"], detached["fk"]))
        attached_gradient, = torch.autograd.grad(
            attached["fk"], values["prediction"], retain_graph=True,
        )
        detached_gradient, = torch.autograd.grad(
            detached["fk"], values["prediction"], retain_graph=True,
        )
        self.assertGreater(float(attached_gradient[..., 0:3].norm()), 0.0)
        self.assertGreater(float(detached_gradient[..., 0:3].norm()), 0.0)
        self.assertTrue(torch.equal(attached_gradient, detached_gradient))

    def test_inert_root_detach_is_rejected_but_default_zero_weight_is_valid(self):
        values = _synthetic_inputs()
        with self.assertRaisesRegex(ValueError, "requires non-zero"):
            _call(values, True, weight=0.0)
        result = _call(values, False, weight=0.0)
        self.assertNotIn("hand_object_contact_geometry", result)


class _SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, noisy, timesteps, text_embedding, object_bps, goals, progress):
        del timesteps, text_embedding, object_bps, goals, progress
        return noisy * 0.05 + self.bias


def _probe_fixture():
    values = _synthetic_inputs()
    batch = {
        "x": values["target"],
        "text_embedding": torch.randn(2, 4),
        "object_bps": torch.randn(2, 8),
        "goals": values["goals"],
        "progress": torch.tensor([[0.0, 1.0, 10.0], [1.0, 2.0, 10.0]]),
        "rest_human_offsets": values["rest_human_offsets"],
        "terminal_window": values["terminal_window"],
        "rest_object_points": values["rest_object_points"],
        "world_to_local_rotation": values["world_to_local_rotation"],
        "object_rotation_reference": values["object_rotation_reference"],
    }
    cfg = OmegaConf.create({
        "fk_weight": 0.3569973401779424,
        "object_surface_weight": 0.4772322188400037,
        "velocity_weight": 0.1,
        "goal_weight": 1.0,
        "hand_object_contact_weight": 3.0,
        "hand_object_contact_hinge": 0.0,
        "hand_object_contact_detach_object": False,
        "hand_object_contact_detach_root": False,
        "fk_foot_temporal_routing": True,
        "routed_foot_residual_multiplier": 1.0,
        "d2ai_full_budget": True,
    })
    return values, batch, cfg


class RootGradientShareProbeTests(unittest.TestCase):
    def _run(self, directory):
        values, batch, cfg = _probe_fixture()
        checkpoint = Path(directory) / "sealed_w3.pth"
        checkpoint.write_bytes(b"synthetic-checkpoint")
        output = Path(directory) / "probe.json"
        result = diagnostics.root_gradient_share_probe(
            _SyntheticModel(),
            GaussianDiffusion(500),
            [batch],
            values["parents_24"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
            cfg,
            checkpoint_path=checkpoint,
            output_path=output,
            window_count=2,
        )
        return result, output

    def test_probe_reports_training_gradients_and_passes_exact_self_check(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run(directory)
            self.assertEqual(result["window_count"], 2)
            self.assertGreater(result["geometry_gradient_l2"]["root_translation"], 0.0)
            self.assertGreater(result["geometry_gradient_l2"]["rotations"], 0.0)
            self.assertGreater(result["root_gradient_share"], 0.0)
            self.assertTrue(result["self_check"]["detached_root_gradient_exactly_zero"])
            self.assertTrue(result["self_check"]["detached_rotation_gradient_bitwise_equal"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_probe_self_check_asserts_on_any_rotation_gradient_change(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(diagnostics.torch, "equal", return_value=False):
                with self.assertRaisesRegex(AssertionError, "rotation gradient"):
                    self._run(directory)


if __name__ == "__main__":
    unittest.main()
