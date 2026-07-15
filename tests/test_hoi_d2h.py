import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.diffusion import GaussianDiffusion, _extract
from priors.exposure import (
    CHECKPOINTS,
    CONDITION_VARIANTS,
    TARGET_TIMESTEPS,
    deterministic_condition_variants,
    fieldwise_mse_per_sample,
    mechanism_gate,
    paired_bootstrap_model_minus_oracle,
)
from priors.remediation import select_teacher_windows, selection_sha256
from priors.representation import REPRESENTATION
from tools.diagnose_hoi_remediation import stable_seed
from tools.summarize_hoi_d2h import RUN_ID, validate_run_identity


class D2HPosteriorTests(unittest.TestCase):
    def setUp(self):
        self.diffusion = GaussianDiffusion()
        self.clean = torch.randn(3, 16, 232)
        self.current = torch.randn_like(self.clean)
        self.fixed = self.clean[:, :2].clone()
        self.noise = torch.randn_like(self.clean)

    def test_registered_posterior_helper_is_formula_equivalent_at_boundaries(self):
        for parent in (1, 499):
            timesteps = torch.full((3,), parent, dtype=torch.long)
            expected_mean = (
                _extract(self.diffusion.posterior_mean_coef1, timesteps, self.current.shape)
                * self.clean
                + _extract(self.diffusion.posterior_mean_coef2, timesteps, self.current.shape)
                * self.current
            )
            actual_mean = self.diffusion.posterior_mean(self.current, self.clean, timesteps)
            torch.testing.assert_close(actual_mean, expected_mean, rtol=0, atol=0)
            expected = expected_mean + (
                0.5 * _extract(
                    self.diffusion.posterior_log_variance, timesteps, self.current.shape,
                )
            ).exp() * self.noise
            expected[:, :2] = self.fixed
            actual = self.diffusion.posterior_sample(
                self.current, self.clean, timesteps, self.noise, self.fixed,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(TARGET_TIMESTEPS[0], 0)
        self.assertEqual(TARGET_TIMESTEPS[-1], 498)

    def test_oracle_posterior_noise_has_registered_marginal_scale(self):
        parent = torch.tensor([250])
        current = self.current[:1]
        clean = self.clean[:1]
        fixed = self.fixed[:1]
        mean = self.diffusion.posterior_mean(current, clean, parent)
        generator = torch.Generator().manual_seed(42)
        noise = torch.randn(4096, 16, 232, generator=generator)
        expanded = self.diffusion.posterior_sample(
            current.expand(4096, -1, -1), clean.expand(4096, -1, -1),
            parent.expand(4096), noise, fixed.expand(4096, -1, -1),
        )
        residual = expanded[:, 2:, 0] - mean[0, 2:, 0]
        observed = float(residual.square().mean())
        expected = float(self.diffusion.posterior_variance[parent])
        self.assertAlmostEqual(observed / expected, 1.0, delta=0.04)

    def test_explicit_posterior_noise_is_identical_across_paired_paths(self):
        timesteps = torch.tensor([1, 10, 499])
        first = self.diffusion.posterior_sample(
            self.current, self.clean, timesteps, self.noise, self.fixed,
        )
        second = self.diffusion.posterior_sample(
            self.current, self.clean, timesteps, self.noise, self.fixed,
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first[:, :2], self.fixed, rtol=0, atol=0)

    def test_invalid_posterior_shapes_and_timesteps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.diffusion.posterior_mean(
                self.current, self.clean, torch.tensor([0, 1, 500]),
            )
        with self.assertRaisesRegex(ValueError, "noise shape"):
            self.diffusion.posterior_sample(
                self.current, self.clean, torch.tensor([0, 1, 2]),
                torch.randn(2, 16, 232), self.fixed,
            )


class D2HReportingTests(unittest.TestCase):
    @staticmethod
    def condition_batch(count=4):
        return {
            "text_embedding": torch.arange(count * 3).reshape(count, 3).float(),
            "object_bps": torch.arange(count * 6).reshape(count, 2, 3).float(),
            "goals": torch.arange(count * 9).reshape(count, 9).float(),
            "progress": torch.arange(count * 3).reshape(count, 3).float(),
        }

    def test_condition_permutations_are_deterministic_and_independent(self):
        batch = self.condition_batch()
        first, permutation = deterministic_condition_variants(batch)
        second, second_permutation = deterministic_condition_variants(batch)
        self.assertEqual(tuple(first), CONDITION_VARIANTS)
        torch.testing.assert_close(permutation, second_permutation, rtol=0, atol=0)
        for name in CONDITION_VARIANTS:
            for field in ("text_embedding", "object_bps", "goals", "progress"):
                torch.testing.assert_close(first[name][field], second[name][field], rtol=0, atol=0)
        torch.testing.assert_close(first["text_permuted"]["object_bps"], batch["object_bps"])
        torch.testing.assert_close(first["bps_permuted"]["text_embedding"], batch["text_embedding"])
        torch.testing.assert_close(first["pelvis_permuted"]["goals"][:, 6:], batch["goals"][:, 6:])
        torch.testing.assert_close(first["object_goal_permuted"]["goals"][:, :6], batch["goals"][:, :6])
        torch.testing.assert_close(first["matched"]["progress"], batch["progress"])

    def test_all_representation_fields_are_reported(self):
        expected = {field.name for field in REPRESENTATION.fields}
        actual = fieldwise_mse_per_sample(
            torch.zeros(2, 16, 232), torch.ones(2, 16, 232),
        )
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(value.shape == (2,) for value in actual.values()))

    def test_paired_bootstrap_is_deterministic_and_preserves_samples(self):
        oracle = np.linspace(0.1, 0.2, 32)
        model = oracle + 0.3
        first = paired_bootstrap_model_minus_oracle(oracle, model, replicates=1000)
        second = paired_bootstrap_model_minus_oracle(oracle, model, replicates=1000)
        self.assertEqual(first, second)
        self.assertTrue(first["positive_lower_bound"])
        self.assertEqual(len(first["per_sample"]["model_minus_oracle"]), 32)
        generator_a = torch.Generator().manual_seed(stable_seed("D2H:test-rng"))
        generator_b = torch.Generator().manual_seed(stable_seed("D2H:test-rng"))
        torch.testing.assert_close(
            torch.randn(8, generator=generator_a),
            torch.randn(8, generator=generator_b),
            rtol=0,
            atol=0,
        )

    def test_teacher_window_selection_is_deterministic(self):
        class Dataset:
            partition = "internal_validation"
            indices = np.asarray([4, 2, 0, 3, 1])
            sequence_ids = np.asarray([0, 1, 0, 2, 1])
            scene_names = np.asarray(["scene-a", "scene-b", "scene-c"], dtype=object)
            language = {"pi": np.asarray([0, 42, 84, 126, 168])}

        first = select_teacher_windows(Dataset(), count=3)
        second = select_teacher_windows(Dataset(), count=3)
        self.assertEqual(first, second)
        self.assertEqual(
            selection_sha256(int(Dataset.indices[position]) for position in first),
            selection_sha256(int(Dataset.indices[position]) for position in second),
        )

    @staticmethod
    def passing_candidates():
        candidates = {}
        for checkpoint in CHECKPOINTS:
            timesteps = {}
            for timestep in TARGET_TIMESTEPS:
                metric = {
                    "positive_lower_bound": timestep in {0, 1, 10, 50},
                    "model_over_oracle_mean_ratio": 3.0,
                }
                timesteps[str(timestep)] = {
                    "matched": {
                        "field_comparison": {
                            "joint_positions": dict(metric),
                            "object_translation": dict(metric),
                        }
                    }
                }
            candidates[checkpoint] = {
                "finite": True,
                "history_max_abs": 0.0,
                "posterior_formula_replay_max_abs": 0.0,
                "timesteps": timesteps,
            }
        return candidates

    def test_mechanism_gate_requires_both_checkpoints_and_finite_checks(self):
        candidates = self.passing_candidates()
        decision = mechanism_gate(candidates)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["classification"], "reverse-state-exposure-positive-stop")
        self.assertFalse(decision["d2h1_started"])
        candidates["R-3072"]["finite"] = False
        decision = mechanism_gate(candidates)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["classification"], "reverse-state-exposure-negative-stop")
        self.assertIn("finite", decision["checkpoint_results"]["R-3072"]["failed_checks"])

    def test_production_sampler_has_no_dataset_or_future_gt_access(self):
        source = inspect.getsource(GaussianDiffusion.sample)
        self.assertNotIn("dataset", source)
        self.assertNotIn("stored", source)
        self.assertNotIn("future", source)
        self.assertNotIn("project_object_rotation_x0", source)
        self.assertIn("posterior_sample", source)

    def test_compact_summary_uses_experiment_manifest_identifier(self):
        validate_run_identity(
            {"run_id": RUN_ID}, {"experiment_id": RUN_ID}, {"run_id": RUN_ID},
        )
        with self.assertRaisesRegex(ValueError, "run-id mismatch"):
            validate_run_identity(
                {"run_id": RUN_ID}, {"run_id": RUN_ID}, {"run_id": RUN_ID},
            )


if __name__ == "__main__":
    unittest.main()
