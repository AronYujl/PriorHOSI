import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from priors.contracts import HOI_CONTRACT, HSI_CONTRACT, validate_contract_paths
from priors.data import PriorWindowDataset, hsi_filter, partition_for_scenes
from priors.models import HSIPrior, HOIPrior, assert_parameter_independence, build_expert
from priors.representation import REPRESENTATION, masked_reconstruction_loss


class RepresentationTests(unittest.TestCase):
    def test_schema_is_contiguous_and_232_dimensional(self):
        self.assertEqual(REPRESENTATION.dimension, 232)
        self.assertEqual([(field.start, field.stop) for field in REPRESENTATION.fields], [
            (0, 84), (84, 216), (216, 219), (219, 228), (228, 232),
        ])
        self.assertEqual(REPRESENTATION.window_frames, 16)
        self.assertEqual(REPRESENTATION.history_frames, 2)
        self.assertEqual(REPRESENTATION.diffusion_steps, 500)

    def test_hsi_mask_removes_object_and_contact_loss_and_gradient(self):
        prediction = torch.randn(2, 16, 232, requires_grad=True)
        target = torch.randn_like(prediction)
        loss = masked_reconstruction_loss(prediction, target, "hsi")
        loss.backward()
        self.assertGreater(float(prediction.grad[:, 2:, :216].abs().max()), 0.0)
        self.assertEqual(float(prediction.grad[:, :, 216:].abs().max()), 0.0)
        self.assertEqual(float(prediction.grad[:, :2].abs().max()), 0.0)

    def test_hoi_mask_supervises_all_output_fields_after_history(self):
        prediction = torch.zeros(1, 16, 232, requires_grad=True)
        target = torch.ones_like(prediction)
        masked_reconstruction_loss(prediction, target, "hoi").backward()
        self.assertGreater(float(prediction.grad[:, 2:, 228:].abs().max()), 0.0)


class ContractTests(unittest.TestCase):
    def test_hsi_dynamic_object_filter_is_exact(self):
        actual = hsi_filter(np.array([-1, 3, -1, 2]), np.array([-1, -1, 4, 5]))
        np.testing.assert_array_equal(actual, np.array([True, False, False, False]))
        self.assertIn("seq_length <= 48", HSI_CONTRACT.filter_rule)

    def test_locked_split_has_no_family_or_scene_leakage(self):
        split = json.loads((REPO / HSI_CONTRACT.split).read_text())
        self.assertEqual(split["seed"], 42)
        self.assertFalse(set(split["train"]["scene_families"]) & set(split["validation"]["scene_families"]))
        self.assertFalse(set(split["train"]["scenes"]) & set(split["validation"]["scenes"]))
        scenes = np.asarray(split["train"]["scenes"][:2] + split["validation"]["scenes"][:2])
        sides = partition_for_scenes(split, scenes)
        np.testing.assert_array_equal(sides, np.asarray(["train", "train", "validation", "validation"], dtype=object))

    def test_author_replaced_lingo_normalization_is_omomo_normalization(self):
        validate_contract_paths(REPO)
        np.testing.assert_array_equal(np.load(REPO / "data/dataset/norm.npy"), np.load(REPO / "data/train/norm.npy"))
        self.assertIn("never recompute", HSI_CONTRACT.normalization)

    def test_hoi_contract_forbids_scene_and_hsi_forbids_object_supervision(self):
        self.assertIn("forbidden", HOI_CONTRACT.scene_condition)
        self.assertIn("forbidden", HSI_CONTRACT.object_condition)

    def test_real_domain_items_expose_only_authorized_conditions(self):
        hoi = PriorWindowDataset(str(REPO), "hoi", limit=1)[0]
        self.assertIn("object_bps", hoi)
        self.assertNotIn("scene_condition", hoi)
        hsi = PriorWindowDataset(str(REPO), "hsi", limit=1)[0]
        self.assertIn("scene_condition", hsi)
        self.assertNotIn("object_bps", hsi)
        self.assertEqual(float(hsi["x"][:, 216:].abs().max()), 0.0)


class ExpertTests(unittest.TestCase):
    def test_hoi_forward_api_has_no_scene_input(self):
        parameters = inspect.signature(HOIPrior.forward).parameters
        self.assertNotIn("scene", parameters)
        self.assertNotIn("scene_condition", parameters)

    def test_experts_are_distinct_types_and_share_no_parameters_or_storage(self):
        hoi = build_expert("hoi", dim_model=32, num_heads=4, num_layers=1)
        hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        self.assertIs(type(hoi), HOIPrior)
        self.assertIs(type(hsi), HSIPrior)
        assert_parameter_independence(hoi, hsi)

    def test_two_instances_of_same_expert_also_do_not_share(self):
        first = build_expert("hoi", dim_model=32, num_heads=4, num_layers=1)
        second = build_expert("hoi", dim_model=32, num_heads=4, num_layers=1)
        assert_parameter_independence(first, second)

    def test_released_checkpoint_initialization_is_rejected(self):
        for expert in ("hoi", "hsi"):
            with self.assertRaisesRegex(ValueError, "randomly initialized"):
                build_expert(expert, init_checkpoint="checkpoint/checkpoint.pth")

    def test_cpu_forward_backward_for_both_expert_apis(self):
        batch = 2
        x = torch.randn(batch, 16, 232)
        t = torch.randint(0, 500, (batch,))
        text = torch.randn(batch, 768)
        goals = torch.randn(batch, 9)
        progress = torch.randn(batch, 3)
        hoi = build_expert("hoi", dim_model=32, num_heads=4, num_layers=1)
        hoi_prediction = hoi(x, t, text, torch.randn(batch, 1024, 3), goals, progress)
        masked_reconstruction_loss(hoi_prediction, x, "hoi").backward()
        hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        hsi_prediction = hsi(x, t, text, torch.randn(batch, 8, 8, 8), goals, progress)
        masked_reconstruction_loss(hsi_prediction, x, "hsi").backward()
        self.assertTrue(any(parameter.grad is not None for parameter in hoi.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in hsi.parameters()))

    def test_formal_resource_architectures_are_domain_specific_and_frozen_size(self):
        hoi = build_expert("hoi", dim_model=512, num_heads=16, num_layers=8, scene_grid_size=32)
        hsi = build_expert("hsi", dim_model=512, num_heads=16, num_layers=8, scene_grid_size=32)
        assert_parameter_independence(hoi, hsi)
        self.assertGreater(sum(parameter.numel() for parameter in hoi.parameters()), 25_000_000)
        self.assertGreater(sum(parameter.numel() for parameter in hsi.parameters()), 25_000_000)
        self.assertFalse(any("scene" in name for name, _ in hoi.named_parameters()))
        self.assertTrue(any("scene" in name for name, _ in hsi.named_parameters()))
        config = (REPO / "code/config/config_prior_resource.yaml").read_text()
        for expected in ("dim_model: 512", "num_heads: 16", "num_layers: 8", "scene_grid_size: 32"):
            self.assertIn(expected, config)


if __name__ == "__main__":
    unittest.main()
