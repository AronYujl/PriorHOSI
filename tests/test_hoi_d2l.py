import inspect
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from priors.adamw_routing import DIRECTIONS
from priors.auxiliary_balancing import (
    BALANCED_WEIGHTS,
    CANDIDATES,
    CURRENT_WEIGHTS,
    DERIVATION_RAW_FK_NORM,
    DERIVATION_RAW_OBJECT_SURFACE_NORM,
    DERIVATION_TARGET_NORM,
    EXPECTED_PRIMARY_SHA256,
    RAW_COMPONENTS,
    WEIGHT_SOURCE_RUN,
    candidate_gradient_components,
    derive_locked_weights,
    mechanism_gate,
    paired_routing_geometry,
    select_fresh_primary,
)
from priors.data import PriorWindowDataset
from priors.gradient_clipping import (
    FIELD_COMPONENTS,
    GATE_TIMESTEPS,
    LOSS_COMPONENTS,
)
from priors.gradient_routing import CHECKPOINTS, PARAMETER_GROUPS, stable_seed
import tools.diagnose_hoi_d2l as diagnostic
from tools.summarize_hoi_d2l import RUN_ID, compact_blocks, validate_run_identity


class D2LSelectionAndProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT), "hoi", partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )

    def test_fresh_selection_is_locked_nonterminal_and_prior_disjoint(self):
        first = select_fresh_primary(self.dataset)
        second = select_fresh_primary(self.dataset)
        self.assertEqual(first["positions"], second["positions"])
        self.assertEqual(first["sha256"], EXPECTED_PRIMARY_SHA256)
        self.assertEqual(first["selected_ranks"], list(range(898, 1026)))
        self.assertEqual(first["terminal_windows"], 0)
        self.assertFalse(set(first["global_indices"]) & first["prior_global_indices"])

    def test_paired_rng_is_candidate_and_checkpoint_independent(self):
        label = "D2L:primary:499:0"
        self.assertEqual(stable_seed(label), stable_seed(label))
        self.assertNotEqual(stable_seed(label), stable_seed("D2L:primary:499:1"))
        self.assertNotIn("balanced", label)
        self.assertNotIn("R-1024", label)

    @staticmethod
    def _source_metrics():
        norm = {
            "human_reconstruction": DERIVATION_TARGET_NORM,
            "object_reconstruction": DERIVATION_TARGET_NORM,
            "weighted_fk": 50.0 * DERIVATION_RAW_FK_NORM,
            "weighted_object_surface": 50.0 * DERIVATION_RAW_OBJECT_SURFACE_NORM,
        }
        checkpoints = {}
        for checkpoint in CHECKPOINTS:
            checkpoints[checkpoint] = {"cohorts": {"primary": {"timesteps": {
                str(timestep): {"blocks": [
                    {"groups": {"all_parameters": {"gradient_l2_norm": dict(norm)}}}
                    for _ in range(8)
                ]}
                for timestep in GATE_TIMESTEPS
            }}}}
        return {"run_id": WEIGHT_SOURCE_RUN, "candidates": checkpoints}

    def test_locked_weight_derivation_is_unique_and_source_only(self):
        result = derive_locked_weights(self._source_metrics())
        self.assertEqual(result["records"], 32)
        self.assertAlmostEqual(result["target_norm"], DERIVATION_TARGET_NORM, places=12)
        for name, value in BALANCED_WEIGHTS.items():
            self.assertAlmostEqual(result["balanced_weights"][name], value, places=12)
        self.assertEqual(CURRENT_WEIGHTS["fk"], 50.0)
        self.assertEqual(CURRENT_WEIGHTS["object_surface"], 50.0)

    def test_weight_derivation_rejects_wrong_source_run(self):
        metrics = self._source_metrics()
        metrics["run_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source run id"):
            derive_locked_weights(metrics)


class D2LGeometryTests(unittest.TestCase):
    @staticmethod
    def _raw_gradients():
        return {
            name: (torch.tensor([float(index), float(index + 1)], dtype=torch.float64),)
            for index, name in enumerate(RAW_COMPONENTS, start=1)
        }

    def test_current_and_balanced_component_formulas_are_complete(self):
        raw = self._raw_gradients()
        current = candidate_gradient_components(raw, CURRENT_WEIGHTS)
        balanced = candidate_gradient_components(raw, BALANCED_WEIGHTS)
        self.assertEqual(set(current), set(LOSS_COMPONENTS))
        self.assertEqual(set(balanced), set(LOSS_COMPONENTS))
        torch.testing.assert_close(current["weighted_fk"][0], 50.0 * raw["fk"][0])
        torch.testing.assert_close(
            balanced["weighted_object_surface"][0],
            BALANCED_WEIGHTS["object_surface"] * raw["object_surface"][0],
        )
        self.assertFalse(torch.equal(current["total"][0], balanced["total"][0]))

    def test_missing_raw_gradient_semantics_are_preserved(self):
        raw = self._raw_gradients()
        raw["terminal_goal"] = (None,)
        balanced = candidate_gradient_components(raw, BALANCED_WEIGHTS)
        self.assertIsNone(balanced["terminal_goal"][0])
        self.assertIsNotNone(balanced["total"][0])

    def _geometry(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
        raw = self._raw_gradients()
        components = {
            name: candidate_gradient_components(raw, weights)
            for name, weights in (
                ("current", CURRENT_WEIGHTS), ("balanced", BALANCED_WEIGHTS),
            )
        }
        direct = {name: value["total"] for name, value in components.items()}
        state = ({
            "step": 5,
            "exp_avg": torch.tensor([0.1, -0.1], dtype=torch.float64),
            "exp_avg_sq": torch.tensor([0.01, 0.02], dtype=torch.float64),
        },)
        optimizer_group = {
            "lr": 0.01, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
        }
        groups = {name: (0,) for name in PARAMETER_GROUPS}
        return paired_routing_geometry(
            raw, direct, (parameter,), state, optimizer_group, groups,
        )

    def test_paired_geometry_reports_every_candidate_direction_loss_and_group(self):
        result = self._geometry()
        self.assertTrue(result["finite"])
        self.assertEqual(set(result["candidates"]), set(CANDIDATES))
        self.assertEqual(set(result["paired_candidate_difference"]), set(PARAMETER_GROUPS))
        for candidate in result["candidates"].values():
            self.assertLessEqual(candidate["total_gradient_formula_relative_l2"], 1e-12)
            self.assertLessEqual(candidate["adamw_decomposition_relative_l2"], 1e-12)
            self.assertEqual(set(candidate["groups"]), set(PARAMETER_GROUPS))
            for record in candidate["groups"].values():
                self.assertEqual(set(record["loss_gradient_l2_norm"]), set(LOSS_COMPONENTS))
                self.assertEqual(set(record["direction_l2_norm"]), set(DIRECTIONS))
                self.assertEqual(set(record["direction_loss_cosine"]), set(DIRECTIONS))
        for group in result["paired_candidate_difference"].values():
            self.assertEqual(set(group), set(DIRECTIONS))
            self.assertTrue(all(
                math.isfinite(value)
                for direction in group.values() for value in direction.values()
            ))


class D2LGateAndLifecycleTests(unittest.TestCase):
    @staticmethod
    def _candidate(*, delta=0.2, human=0.4, objects=0.5, formula=0.0):
        blocks = []
        for _ in range(8):
            candidate_records = {}
            for name in CANDIDATES:
                human_value = human - delta if name == "current" else human
                candidate_records[name] = {
                    "finite": True,
                    "total_gradient_formula_relative_l2": formula,
                    "clipping": {"formula_replay_max_abs": formula},
                    "adamw_decomposition_relative_l2": formula,
                    "groups": {"all_parameters": {
                        "direction_loss_cosine": {
                            "clipped_total": {
                                "human_reconstruction": {
                                    "value": human_value, "defined": True,
                                },
                                "object_reconstruction": {
                                    "value": objects, "defined": True,
                                },
                            },
                            "adamw_full": {
                                "human_reconstruction": {
                                    "value": human_value, "defined": True,
                                },
                                "object_reconstruction": {
                                    "value": objects, "defined": True,
                                },
                            },
                        },
                    }},
                }
            blocks.append({"finite": True, "candidates": candidate_records})
        return {
            "finite": True,
            "model_state_sha256_before": "model",
            "model_state_sha256_after": "model",
            "optimizer_state_sha256_before": "optimizer",
            "optimizer_state_sha256_after": "optimizer",
            "mapped_state_sha256_before": "mapped",
            "mapped_state_sha256_after": "mapped",
            "parameter_grad_buffers_clear": True,
            "optimizer_contract_exact": True,
            "weight_provenance_exact": True,
            "timesteps": {
                str(timestep): {"blocks": blocks} for timestep in GATE_TIMESTEPS
            },
        }

    def test_gate_requires_both_paths_checkpoints_and_every_conjunct(self):
        checkpoints = {name: self._candidate() for name in CHECKPOINTS}
        decision = mechanism_gate(checkpoints)
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "gradient-balanced-auxiliary-routing-positive-stop",
        )
        self.assertFalse(decision["training_authorized"])
        checkpoints["R-3072"] = self._candidate(delta=0.05)
        decision = mechanism_gate(checkpoints)
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["classification"],
            "gradient-balanced-auxiliary-routing-negative-stop",
        )

    def test_compact_summary_keeps_both_candidates_and_paired_differences(self):
        geometry = D2LGeometryTests()._geometry()
        block = {
            "windows": 16,
            "q_noise_sha256": "noise",
            "production_total_value_replay_abs": 0.0,
            "raw_loss_values": {name: 1.0 for name in RAW_COMPONENTS},
            "candidate_loss_values": {
                candidate: {loss: 1.0 for loss in LOSS_COMPONENTS}
                for candidate in CANDIDATES
            },
            **geometry,
        }
        compact = compact_blocks([block])
        self.assertEqual(set(compact["candidates"]), set(CANDIDATES))
        self.assertEqual(
            set(compact["balanced_minus_current_direction_loss_cosine_mean"]),
            set(PARAMETER_GROUPS),
        )

    def test_source_is_zero_update_and_does_not_create_optimizer(self):
        source = inspect.getsource(diagnostic)
        self.assertIn("torch.autograd.grad", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn(".step(", source)
        self.assertIn('"training_updates": 0', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"production_loss_change": False', source)
        self.assertIn('"weight_sweep": False', source)

    def test_diagnostic_is_directly_executable(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/diagnose_hoi_d2l.py"), "--help"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_summary_requires_manifest_experiment_identifier(self):
        validate_run_identity(
            {"run_id": RUN_ID}, {"experiment_id": RUN_ID}, {"run_id": RUN_ID},
        )
        with self.assertRaisesRegex(ValueError, "run-id mismatch"):
            validate_run_identity(
                {"run_id": RUN_ID}, {"run_id": RUN_ID}, {"run_id": RUN_ID},
            )


if __name__ == "__main__":
    unittest.main()
