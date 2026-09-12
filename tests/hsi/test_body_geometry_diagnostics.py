"""Observation coverage and zero-update calibration for body scene geometry."""

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.diagnostics import (
    body_geometry_calibrated_weight,
    body_geometry_coverage_metrics,
    body_geometry_gradient_calibration,
)


class BodyGeometryCoverageTests(unittest.TestCase):
    def test_height_coverage_counts_valid_near_geometry_and_exposes_missing_stencil(self):
        joints = torch.zeros(1, 2, 24, 3)
        joints[..., 1] = 0.8
        joints[:, :, :8, 1] = torch.tensor([1.21, 0.09, 2.0, 2.0, 1.5, 0.05, 0.1, 1.2])
        distance = torch.full((1, 2, 24), 0.1)
        distance[:, :, 4:7] = torch.tensor([0.3, -0.3, -0.02])
        valid = torch.ones_like(distance, dtype=torch.bool)
        valid[:, :, 2:4] = False
        center_oob = torch.zeros_like(valid)
        center_oob[:, :, 2] = True
        query = dict(signed_distance=distance, valid=valid, center_out_of_bounds=center_oob,
                     gradient_world=torch.ones_like(joints))

        result = body_geometry_coverage_metrics(joints, query)

        self.assertEqual(result["query_count"], 48)
        self.assertEqual(result["valid_count"], 44)
        self.assertEqual(result["invalid_count"], 4)
        self.assertEqual(result["center_out_of_bounds_count"], 2)
        self.assertEqual(result["center_in_bounds_stencil_invalid_count"], 2)
        self.assertEqual(result["valid_near_surface_count"], 40)
        self.assertEqual(result["valid_near_surface_outside_old_crop_count"], 4)
        self.assertEqual(result["valid_near_surface_above_old_crop_count"], 2)
        self.assertEqual(result["valid_near_surface_below_old_crop_count"], 2)
        self.assertEqual(result["per_joint"][0]["joint_name"], "pelvis")
        self.assertEqual(result["per_joint"][0]["valid_near_surface_above_old_crop_count"], 2)
        self.assertEqual(result["per_joint"][6]["valid_near_surface_below_old_crop_count"], 0)
        self.assertEqual(result["per_joint"][7]["valid_near_surface_above_old_crop_count"], 0)
        self.assertIsNone(result["per_joint"][2]["valid_signed_distance_mean_m"])
        self.assertIsNone(result["per_joint"][3]["valid_gradient_norm_mean"])


class _CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Sequential(nn.Linear(4, 5), nn.Tanh())
        self.out = nn.Linear(5, 232)
        self.body_geometry_refiner = nn.Module()
        self.body_geometry_refiner.hidden = nn.Linear(5, 5)
        self.body_geometry_refiner.output = nn.Linear(5, 216)
        nn.init.zeros_(self.body_geometry_refiner.output.weight)
        nn.init.zeros_(self.body_geometry_refiner.output.bias)

    def forward(self, x):
        hidden = self.transformer(x)
        geometry = self.body_geometry_refiner.hidden(hidden).tanh()
        correction = self.body_geometry_refiner.output(geometry)
        coarse = self.out(hidden)
        return coarse + torch.cat((correction, torch.zeros_like(coarse[:, 216:])), dim=-1)


class BodyGeometryCalibrationTests(unittest.TestCase):
    def test_zero_output_layer_learns_and_original_objective_excludes_added_weight(self):
        torch.manual_seed(42)
        model = _CalibrationModel()
        prediction = model(torch.randn(3, 4))
        geometry = (prediction[:, :216] - 0.2).square().mean()
        jpos = prediction[:, :84].square().mean()
        jrot = (prediction[:, 84:216] + 0.1).square().mean()
        fk = (prediction[:, :216].mean(-1) - 0.3).square().mean()
        seam = (prediction[1:, :216] - prediction[:-1, :216]).square().mean()
        base = jpos + jrot + 0.5 * seam + 3.0 * fk
        weight = 0.7
        losses = dict(loss=base - 3.0 * fk + weight * geometry,
                      loss_body_geometry=geometry, loss_jpos=jpos, loss_jrot=jrot,
                      loss_fk=fk, loss_fullbody_seam=seam,
                      body_geometry_active_fraction=torch.tensor(0.5),
                      body_geometry_invalid_fraction=torch.tensor(0.1))
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}

        records = body_geometry_gradient_calibration(model, losses, 3.0, geometry_weight=weight)

        self.assertAlmostEqual(records["r2_total"]["raw_loss"], float(base.detach()), places=6)
        self.assertGreater(records["body_geometry"]["refiner_output_gradient_norm"], 0.0)
        self.assertGreater(records["body_geometry"]["rotation_head_gradient_norm"], 0.0)
        for record in records.values():
            self.assertTrue(all(math.isfinite(value) for value in record.values()))
        hidden_gradient = torch.autograd.grad(geometry, model.body_geometry_refiner.hidden.weight)[0]
        torch.testing.assert_close(hidden_gradient, torch.zeros_like(hidden_gradient), atol=0, rtol=0)
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)

    def test_one_weight_uses_median_original_total_in_both_parameter_regions(self):
        records = [
            {"r2_total": {"trunk_gradient_norm": trunk, "rotation_head_gradient_norm": rotation},
             "body_geometry": {"trunk_gradient_norm": geometry_trunk,
                               "rotation_head_gradient_norm": geometry_rotation}}
            for trunk, rotation, geometry_trunk, geometry_rotation in
            ((10.0, 8.0, 2.0, 2.0), (20.0, 16.0, 4.0, 4.0), (30.0, 24.0, 6.0, 6.0))
        ]

        result = body_geometry_calibrated_weight(records)

        self.assertAlmostEqual(result["trunk_10_percent_weight"], 0.5)
        self.assertAlmostEqual(result["rotation_head_25_percent_weight"], 1.0)
        self.assertAlmostEqual(result["body_geometry_loss_weight"], 0.5)
        self.assertIs(result["ranks"], records)


if __name__ == "__main__":
    unittest.main()
