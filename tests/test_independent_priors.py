import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from priors.contracts import HOI_CONTRACT, HSI_CONTRACT, validate_contract_paths
from priors.data import PriorWindowDataset, hsi_filter, partition_for_scenes
from priors.diffusion import GaussianDiffusion, normalize_progress
from priors.losses import hoi_training_losses
from priors.models import (
    HSIPrior, HOIPrior, assert_parameter_independence, build_expert,
    load_trained_hoi_prior,
)
from priors.representation import REPRESENTATION, masked_reconstruction_loss
from datasets.utils import get_smpl_parents


WORKER_EXPERT = os.environ.get("INFBAGEL_WORKER_EXPERT")
if WORKER_EXPERT not in {None, "hoi", "hsi"}:
    raise ValueError(f"invalid INFBAGEL_WORKER_EXPERT: {WORKER_EXPERT}")


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

    @unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
    def test_author_replaced_lingo_normalization_is_omomo_normalization(self):
        validate_contract_paths(REPO)
        np.testing.assert_array_equal(np.load(REPO / "data/dataset/norm.npy"), np.load(REPO / "data/train/norm.npy"))
        self.assertIn("never recompute", HSI_CONTRACT.normalization)

    def test_hoi_worker_contract_paths_require_no_lingo_assets(self):
        validate_contract_paths(REPO, expert="hoi")

    def test_hoi_contract_forbids_scene_and_hsi_forbids_object_supervision(self):
        self.assertIn("forbidden", HOI_CONTRACT.scene_condition)
        self.assertIn("forbidden", HSI_CONTRACT.object_condition)

    def test_real_hoi_item_exposes_only_authorized_conditions(self):
        hoi = PriorWindowDataset(str(REPO), "hoi", limit=1)[0]
        self.assertIn("object_bps", hoi)
        self.assertIn("rest_human_offsets", hoi)
        self.assertNotIn("scene_condition", hoi)

    def test_locked_hoi_train_validation_split_has_no_sequence_leakage(self):
        split_path = REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        self.assertEqual(split["algorithm"], "omomo-sequence-sha256-seed42-v1")
        self.assertEqual(split["internal_validation"]["sequence_count"], 216)
        self.assertEqual(split["train"]["sequence_count"], 4088)
        self.assertEqual(split["internal_validation"]["window_count"], 29382)
        self.assertEqual(split["train"]["window_count"], 568486)
        train = set(split["train"]["sequence_indices"])
        validation = set(split["internal_validation"]["sequence_indices"])
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(range(4304)))
        self.assertFalse(split["official_test_used_for_selection"])

    def test_real_hoi_partitions_match_locked_window_counts(self):
        split = "experiments/splits/omomo_hoi_train_validation_seed42.json"
        train = PriorWindowDataset(str(REPO), "hoi", partition="train", split_manifest=split)
        validation = PriorWindowDataset(
            str(REPO), "hoi", partition="internal_validation", split_manifest=split,
        )
        self.assertEqual(len(train), 568486)
        self.assertEqual(len(validation), 29382)
        self.assertFalse(set(train.sequence_ids[train.indices]) & set(validation.sequence_ids[validation.indices]))

    @unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
    def test_real_hsi_item_exposes_only_authorized_conditions(self):
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

    def test_released_checkpoint_schema_is_rejected_by_evaluation_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "released.pth"
            torch.save({"model": {"legacy": torch.zeros(1)}}, path)
            with self.assertRaisesRegex(ValueError, "not a Phase 1B"):
                load_trained_hoi_prior(str(path), torch.device("cpu"))

    def test_phase_1b_checkpoint_loads_strict_ema_weights(self):
        model = build_expert("hoi", dim_model=32, num_heads=4, num_layers=1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hoi.pth"
            torch.save({
                "schema_version": 1,
                "checkpoint_type": "hoi_prior_phase1b",
                "expert": "hoi",
                "initialization": "random",
                "model_config": {"dim_model": 32, "num_heads": 4, "num_layers": 1},
                "model": model.state_dict(),
                "ema_model": model.state_dict(),
            }, path)
            loaded, metadata = load_trained_hoi_prior(str(path), torch.device("cpu"))
            self.assertIsInstance(loaded, HOIPrior)
            self.assertEqual(metadata["weights"], "ema_model")
            for expected, actual in zip(model.parameters(), loaded.parameters()):
                torch.testing.assert_close(expected, actual)

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


class HOITrainingTests(unittest.TestCase):
    def test_diffusion_keeps_two_history_frames_exactly_fixed(self):
        clean = torch.randn(3, 16, 232)
        steps = torch.tensor([0, 249, 499])
        noisy = GaussianDiffusion().q_sample(clean, steps, torch.randn_like(clean))
        torch.testing.assert_close(noisy[:, :2], clean[:, :2], rtol=0, atol=0)
        self.assertGreater(float((noisy[:, 2:] - clean[:, 2:]).abs().max()), 0.0)

    def test_progress_condition_preserves_pi_end_pi_and_length_semantics(self):
        progress = normalize_progress(torch.tensor([[12.0, 60.0, 120.0]]))
        torch.testing.assert_close(progress[0, :2], torch.tensor([0.1, 0.5]))
        self.assertAlmostEqual(float(progress[0, 2]), float(np.log1p(120.0) / 10.0), places=6)

    def test_preregistered_hoi_losses_are_finite_and_backpropagate(self):
        prediction = torch.randn(2, 16, 232, requires_grad=True)
        target = torch.randn_like(prediction)
        goals = torch.randn(2, 9)
        offsets = torch.zeros(2, 24, 3)
        parents = torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long)
        losses = hoi_training_losses(
            prediction, target, goals, offsets, parents,
            torch.full((3,), -2.0), torch.full((3,), 2.0),
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad[:, 2:].abs().max()), 0.0)
        self.assertEqual(float(prediction.grad[:, :2].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
