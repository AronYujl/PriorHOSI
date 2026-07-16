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

from priors.adamw_routing import (
    DIRECTIONS,
    EXPECTED_OPTIMIZER,
    EXPECTED_PRIMARY_SHA256,
    adamw_directions,
    mechanism_gate,
    optimizer_state_sha256,
    routing_geometry,
    select_fresh_primary,
    validate_optimizer_contract,
)
from priors.data import PriorWindowDataset
from priors.gradient_clipping import (
    BASE_COMPONENTS,
    FIELD_COMPONENTS,
    GATE_TIMESTEPS,
    LOSS_COMPONENTS,
)
from priors.gradient_routing import CHECKPOINTS, PARAMETER_GROUPS, stable_seed
import tools.diagnose_hoi_d2k as diagnostic
from tools.summarize_hoi_d2k import RUN_ID, compact_blocks, validate_run_identity


class D2KSelectionTests(unittest.TestCase):
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
        self.assertEqual(len(first["positions"]), 128)
        self.assertEqual(first["terminal_windows"], 0)
        self.assertEqual(first["selected_ranks"][0], 769)
        self.assertEqual(first["selected_ranks"][-1], 897)
        self.assertEqual(first["skipped_terminal_ranks"], [768, 770])
        self.assertFalse(set(first["global_indices"]) & first["prior_global_indices"])

    def test_paired_rng_is_checkpoint_independent_and_block_bound(self):
        label = "D2K:primary:499:0"
        self.assertEqual(stable_seed(label), stable_seed(label))
        self.assertNotEqual(stable_seed(label), stable_seed("D2K:primary:499:1"))
        self.assertNotIn("R-1024", label)


class D2KAdamWTests(unittest.TestCase):
    @staticmethod
    def _state(parameter, *, step=5):
        return {
            "step": step,
            "exp_avg": torch.tensor([0.03, -0.07], dtype=parameter.dtype),
            "exp_avg_sq": torch.tensor([0.004, 0.009], dtype=parameter.dtype),
        }

    def test_counterfactual_matches_real_adamw_next_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.25, -2.5], dtype=torch.float64))
        raw_gradient = torch.tensor([2.0, -4.0], dtype=torch.float64)
        group = {
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
        state = self._state(parameter)
        replay = adamw_directions((raw_gradient,), (parameter,), (state,), group)

        actual = torch.nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW(
            [actual], lr=group["lr"], betas=group["betas"], eps=group["eps"],
            weight_decay=group["weight_decay"],
        )
        optimizer.state[actual]["step"] = torch.tensor(float(state["step"]))
        optimizer.state[actual]["exp_avg"] = state["exp_avg"].clone()
        optimizer.state[actual]["exp_avg_sq"] = state["exp_avg_sq"].clone()
        before = actual.detach().clone()
        actual.grad = raw_gradient.clone()
        torch.nn.utils.clip_grad_norm_([actual], 1.0)
        optimizer.step()
        observed = (before - actual.detach()) / group["lr"]

        torch.testing.assert_close(
            replay["directions"]["adamw_full"][0], observed,
            rtol=1e-10, atol=1e-12,
        )
        self.assertLessEqual(replay["adamw_decomposition_relative_l2"], 1e-12)
        self.assertAlmostEqual(replay["clipping"]["postclip_norm"], 1.0, places=6)

    def test_missing_gradient_skips_moments_and_weight_decay(self):
        parameter = torch.nn.Parameter(torch.tensor([1.25, -2.5], dtype=torch.float64))
        state = self._state(parameter)
        group = {"lr": 0.01, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
        result = adamw_directions((None,), (parameter,), (state,), group)
        self.assertTrue(all(result["directions"][name][0] is None for name in DIRECTIONS))
        self.assertEqual(result["adamw_decomposition_relative_l2"], 0.0)

    @staticmethod
    def _complete_optimizer_state(checkpoint="R-1024"):
        expected = EXPECTED_OPTIMIZER[checkpoint]
        parameters = tuple(torch.nn.Parameter(torch.zeros(1)) for _ in range(119))
        state = {
            index: {
                "step": torch.tensor(float(expected["step"])),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter),
            }
            for index, parameter in enumerate(parameters)
        }
        optimizer = {
            "state": state,
            "param_groups": [{
                "lr": expected["lr"],
                "initial_lr": expected["initial_lr"],
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "params": list(range(119)),
            }],
        }
        return parameters, optimizer

    def test_optimizer_contract_order_and_state_hash_are_exact(self):
        parameters, optimizer = self._complete_optimizer_state()
        contract = validate_optimizer_contract("R-1024", optimizer, parameters)
        self.assertEqual(contract["state_count"], 119)
        self.assertEqual(contract["next_step"], 6001)
        self.assertEqual(optimizer_state_sha256(optimizer), optimizer_state_sha256(optimizer))
        optimizer["param_groups"][0]["params"][0:2] = [1, 0]
        with self.assertRaisesRegex(ValueError, "parameter order"):
            validate_optimizer_contract("R-1024", optimizer, parameters)


class D2KGeometryAndGateTests(unittest.TestCase):
    def _geometry(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
        base = {
            name: (torch.tensor([float(index), float(index + 1)], dtype=torch.float64),)
            for index, name in enumerate(BASE_COMPONENTS, start=1)
        }
        direct = (sum(base[name][0] for name in BASE_COMPONENTS),)
        state = ({
            "step": 5,
            "exp_avg": torch.tensor([0.1, -0.1], dtype=torch.float64),
            "exp_avg_sq": torch.tensor([0.01, 0.02], dtype=torch.float64),
        },)
        group = {"lr": 0.01, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
        groups = {name: (0,) for name in PARAMETER_GROUPS}
        return routing_geometry(base, direct, (parameter,), state, group, groups)

    def test_all_fields_directions_losses_and_groups_are_reported(self):
        result = self._geometry()
        self.assertTrue(result["finite"])
        self.assertLessEqual(result["total_gradient_formula_relative_l2"], 1e-12)
        self.assertLessEqual(result["adamw_decomposition_relative_l2"], 1e-12)
        self.assertEqual(set(result["groups"]), set(PARAMETER_GROUPS))
        self.assertTrue(set(FIELD_COMPONENTS).issubset(LOSS_COMPONENTS))
        for record in result["groups"].values():
            self.assertEqual(set(record["loss_gradient_l2_norm"]), set(LOSS_COMPONENTS))
            self.assertEqual(set(record["direction_l2_norm"]), set(DIRECTIONS))
            self.assertEqual(set(record["direction_loss_cosine"]), set(DIRECTIONS))
            self.assertEqual(set(record["direction_cosine"]), set(DIRECTIONS))
            self.assertTrue(all(math.isfinite(value) for value in record["adamw_minus_clipped_efficiency"].values()))

    @staticmethod
    def _candidate(*, delta=0.1, human=0.3, objects=0.4, formula=0.0):
        blocks = [{
            "finite": True,
            "total_gradient_formula_relative_l2": formula,
            "clipping": {"formula_replay_max_abs": formula},
            "adamw_decomposition_relative_l2": formula,
            "groups": {"all_parameters": {
                "adamw_minus_clipped_efficiency": {"human_reconstruction": delta},
                "direction_loss_cosine": {"adamw_full": {
                    "human_reconstruction": {"value": human, "defined": True},
                    "object_reconstruction": {"value": objects, "defined": True},
                }},
            }},
        } for _ in range(8)]
        return {
            "finite": True,
            "model_state_sha256_before": "model",
            "model_state_sha256_after": "model",
            "optimizer_state_sha256_before": "raw",
            "optimizer_state_sha256_after": "raw",
            "mapped_state_sha256_before": "mapped",
            "mapped_state_sha256_after": "mapped",
            "parameter_grad_buffers_clear": True,
            "optimizer_contract_exact": True,
            "timesteps": {str(timestep): {"blocks": blocks} for timestep in GATE_TIMESTEPS},
        }

    def test_gate_requires_both_checkpoints_and_every_high_noise_conjunct(self):
        candidates = {name: self._candidate() for name in CHECKPOINTS}
        decision = mechanism_gate(candidates)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["classification"], "adamw-human-routing-rescue-positive-stop")
        self.assertFalse(decision["training_authorized"])
        candidates["R-3072"] = self._candidate(delta=0.01)
        decision = mechanism_gate(candidates)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["classification"], "adamw-human-routing-rescue-negative-stop")

    def test_compact_summary_keeps_field_complete_geometry(self):
        geometry = self._geometry()
        block = {"windows": 16, "q_noise_sha256": "noise", "loss_values": {
            loss: 1.0 for loss in LOSS_COMPONENTS
        }, **geometry}
        compact = compact_blocks([block])
        self.assertEqual(compact["windows"], 16)
        self.assertEqual(
            set(compact["groups"]["all_parameters"]["direction_loss_cosine_mean"]),
            set(DIRECTIONS),
        )

    def test_source_is_zero_update_and_does_not_create_optimizer(self):
        source = inspect.getsource(diagnostic)
        self.assertIn("torch.autograd.grad", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn(".step(", source)
        self.assertIn('"training_updates": 0', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"checkpoint_write": False', source)

    def test_diagnostic_is_directly_executable(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/diagnose_hoi_d2k.py"), "--help"],
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
