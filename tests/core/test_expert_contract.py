"""Frozen cross-branch expert contract: the parts neither expert may change.

Split out of the pre-split ``tests/test_independent_priors.py`` at commit
c77b9d8 with every assertion carried over verbatim.  Everything in this file
imports only ``priors.core`` and ``priors.hsi``, so ``phase/01c-hsi`` can run
``tests/core/`` with ``code/priors/hoi/`` deleted.  The HOI-coupled half of the
original file lives in ``tests/hoi/test_hoi_prior.py``.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.core.contracts import HOI_CONTRACT, HSI_CONTRACT, validate_contract_paths
from priors.core.expert_api import (
    DEFAULT_ARCHITECTURE_VARIANT, assert_parameter_independence, build_expert,
)
from priors.core.representation import (
    REPRESENTATION, masked_reconstruction_loss, transform_object_points_for_next_window,
)
from priors.hsi.models import HSIPrior

try:  # ``priors.hoi`` is absent on the HSIPrior branch by design.
    from priors.hoi.models import HOI_ARCHITECTURE_BASE
except ImportError:  # pragma: no cover - exercised only on phase/01c-hsi
    HOI_ARCHITECTURE_BASE = None


WORKER_EXPERT = os.environ.get("INFBAGEL_WORKER_EXPERT")
if WORKER_EXPERT not in {None, "hoi", "hsi"}:
    raise ValueError(f"invalid INFBAGEL_WORKER_EXPERT: {WORKER_EXPERT}")


class RepresentationTests(unittest.TestCase):
    """The 232-D window schema and its per-expert loss masks."""

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

    def test_autoregressive_object_transform_uses_bps_dtype(self):
        points = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]], dtype=torch.float32)
        rotation = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        translation = torch.tensor([[[0.5, -1.0, 2.0]]], dtype=torch.float64)
        transformed = transform_object_points_for_next_window(points, rotation, translation)
        self.assertEqual(transformed.dtype, torch.float32)
        torch.testing.assert_close(
            transformed,
            points + torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32),
        )


class ContractTests(unittest.TestCase):
    """Dataset contracts and normalization assets, without any dataset code."""

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


class ExpertTests(unittest.TestCase):
    """The construction guard that forbids released-checkpoint initialization."""

    def test_released_checkpoint_initialization_is_rejected(self):
        for expert in ("hoi", "hsi"):
            with self.assertRaisesRegex(ValueError, "randomly initialized"):
                build_expert(expert, init_checkpoint="checkpoint/checkpoint.pth")


class SharedExpertApiTests(unittest.TestCase):
    """Core-only cover for the guarantees the HOI half used to prove with HOIPrior.

    The originals still run in ``tests/hoi/test_hoi_prior.py``; these duplicate
    the same properties through HSIPrior alone so the contract layer stays
    self-verifying on a branch without ``priors.hoi``.
    """

    def test_hsi_expert_builds_and_backpropagates_through_the_core_backbone(self):
        batch = 2
        x = torch.randn(batch, 16, 232)
        hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        self.assertIs(type(hsi), HSIPrior)
        prediction = hsi(
            x, torch.randint(0, 500, (batch,)), torch.randn(batch, 768),
            torch.randn(batch, 8, 8, 8), torch.randn(batch, 9), torch.randn(batch, 3),
        )
        masked_reconstruction_loss(prediction, x, "hsi").backward()
        self.assertTrue(any(parameter.grad is not None for parameter in hsi.parameters()))

    def test_two_hsi_experts_share_no_parameters_or_storage(self):
        first = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        second = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        assert_parameter_independence(first, second)

    def test_parameter_independence_rejects_a_module_shared_with_itself(self):
        expert = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        with self.assertRaisesRegex(AssertionError, "share Parameter objects"):
            assert_parameter_independence(expert, expert)

    def test_hsi_rejects_every_hoi_architecture_variant(self):
        with self.assertRaisesRegex(ValueError, "forbidden for HSIPrior"):
            build_expert("hsi", architecture_variant="d2ad_local_frame_interaction_adapter")

    def test_unknown_expert_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown expert"):
            build_expert("mixer")

    @unittest.skipIf(HOI_ARCHITECTURE_BASE is None, "priors.hoi is absent on the HSI branch")
    def test_core_default_variant_still_equals_the_hoi_base_constant(self):
        # core/expert_api.py repeats the literal instead of importing it, so a
        # rename on the HOI side must not silently disable the HSI guard above.
        self.assertEqual(DEFAULT_ARCHITECTURE_VARIANT, HOI_ARCHITECTURE_BASE)


if __name__ == "__main__":
    unittest.main()
