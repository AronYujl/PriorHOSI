import inspect
import math
import os
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from priors.data import PriorWindowDataset
from priors.gradient_routing import (
    BASE_COMPONENTS,
    CHECKPOINTS,
    EXPECTED_PRIMARY_SHA256,
    EXPECTED_TERMINAL_SHA256,
    GATE_TIMESTEPS,
    LOSS_COMPONENTS,
    PARAMETER_GROUPS,
    gradient_geometry,
    mechanism_gate,
    parameter_group_indices,
    select_fresh_holdouts,
    stable_seed,
    state_dict_sha256,
)
from priors.models import HOIPrior
import tools.diagnose_hoi_d2i as diagnostic
from tools.summarize_hoi_d2i import RUN_ID, validate_run_identity


class D2ISelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT), "hoi", partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )

    def test_fresh_selections_are_deterministic_disjoint_and_locked(self):
        first = select_fresh_holdouts(self.dataset)
        second = select_fresh_holdouts(self.dataset)
        self.assertEqual(first["primary"]["positions"], second["primary"]["positions"])
        self.assertEqual(first["terminal"]["positions"], second["terminal"]["positions"])
        self.assertEqual(first["primary"]["sha256"], EXPECTED_PRIMARY_SHA256)
        self.assertEqual(first["terminal"]["sha256"], EXPECTED_TERMINAL_SHA256)
        self.assertEqual(first["primary"]["terminal_windows"], 0)
        self.assertEqual(first["terminal"]["terminal_windows"], 64)
        self.assertFalse(set(first["primary"]["global_indices"]) & first["d2h_global_indices"])
        self.assertFalse(set(first["terminal"]["global_indices"]) & first["d2h_global_indices"])

    def test_stable_rng_is_label_bound(self):
        self.assertEqual(stable_seed("D2I:primary:499:0"), stable_seed("D2I:primary:499:0"))
        self.assertNotEqual(stable_seed("D2I:primary:499:0"), stable_seed("D2I:primary:499:1"))


class D2IGradientTests(unittest.TestCase):
    def _gradients(self):
        values = {}
        for index, name in enumerate(BASE_COMPONENTS, start=1):
            values[name] = (
                torch.tensor([float(index), float(index + 1)]),
                torch.tensor([float(index + 2)]),
            )
        return values

    def test_weighted_formula_replay_and_all_group_reporting(self):
        base = self._gradients()
        direct = tuple(sum(base[name][i] for name in BASE_COMPONENTS) for i in range(2))
        groups = {name: (0, 1) for name in PARAMETER_GROUPS}
        result = gradient_geometry(base, direct, groups)
        self.assertTrue(result["finite"])
        self.assertLessEqual(result["total_gradient_formula_relative_l2"], 1e-12)
        self.assertEqual(set(result["groups"]), set(PARAMETER_GROUPS))
        for record in result["groups"].values():
            self.assertEqual(set(record["gradient_l2_norm"]), set(LOSS_COMPONENTS))
            self.assertEqual(set(record["cosine_matrix"]), set(LOSS_COMPONENTS))
            self.assertTrue(math.isfinite(record["total_over_reconstruction_norm_ratio"]))

    def test_zero_gradient_cosines_are_explicitly_undefined_but_finite(self):
        base = self._gradients()
        base["terminal_goal"] = (torch.zeros(2), torch.zeros(1))
        direct = tuple(sum(base[name][i] for name in BASE_COMPONENTS) for i in range(2))
        groups = {name: (0, 1) for name in PARAMETER_GROUPS}
        result = gradient_geometry(base, direct, groups)
        entry = result["groups"]["all_parameters"]["cosine_matrix"]["terminal_goal"]["total"]
        self.assertEqual(entry, {"value": 0.0, "defined": False})
        self.assertTrue(result["finite"])

    def test_formula_replay_detects_mismatch(self):
        base = self._gradients()
        direct = tuple(sum(base[name][i] for name in BASE_COMPONENTS) + 1.0 for i in range(2))
        groups = {name: (0, 1) for name in PARAMETER_GROUPS}
        result = gradient_geometry(base, direct, groups)
        self.assertGreater(result["total_gradient_formula_relative_l2"], 1e-5)

    def test_parameter_groups_and_state_hash_cover_frozen_model(self):
        model = HOIPrior(dim_model=32, num_heads=4, num_layers=1)
        parameters, groups = parameter_group_indices(model)
        self.assertEqual(set(groups), set(PARAMETER_GROUPS))
        self.assertTrue(all(groups[name] for name in PARAMETER_GROUPS))
        before = state_dict_sha256(model)
        loss = sum(parameter.square().sum() for parameter in parameters)
        torch.autograd.grad(loss, parameters)
        after = state_dict_sha256(model)
        self.assertEqual(before, after)
        self.assertTrue(all(parameter.grad is None for parameter in parameters))


class D2IGateTests(unittest.TestCase):
    @staticmethod
    def _candidate(*, ratio=30.0, cosine=0.1, formula=0.0, state="same"):
        blocks = [
            {
                "finite": True,
                "total_gradient_formula_relative_l2": formula,
                "groups": {
                    "all_parameters": {
                        "total_over_reconstruction_norm_ratio": ratio,
                        "total_reconstruction_cosine": {"value": cosine, "defined": True},
                    },
                },
            }
            for _ in range(8)
        ]
        return {
            "finite": True,
            "state_dict_sha256_before": state,
            "state_dict_sha256_after": state,
            "cohorts": {
                "primary": {
                    "timesteps": {str(timestep): {"blocks": blocks} for timestep in GATE_TIMESTEPS},
                },
            },
        }

    def test_gate_requires_both_checkpoints_and_all_locked_checks(self):
        candidates = {name: self._candidate() for name in CHECKPOINTS}
        decision = mechanism_gate(candidates)
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "weighted-objective-gradient-dominance-positive-stop",
        )
        self.assertFalse(decision["training_authorized"])
        candidates["R-3072"] = self._candidate(cosine=0.4)
        decision = mechanism_gate(candidates)
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "weighted-objective-gradient-dominance-negative-stop",
        )

    def test_diagnostic_source_has_no_optimizer_step_or_production_mutation(self):
        source = inspect.getsource(diagnostic)
        self.assertIn("torch.autograd.grad", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".step(", source)
        self.assertIn('"training_updates": 0', source)
        self.assertIn('"production_model_change": False', source)

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
