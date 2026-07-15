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

from priors.data import PriorWindowDataset
from priors.gradient_clipping import (
    BASE_COMPONENTS,
    EXPECTED_PRIMARY_SHA256,
    FIELD_COMPONENTS,
    GATE_TIMESTEPS,
    GRADIENT_CLIP_NORM,
    LOSS_COMPONENTS,
    clip_gradient_geometry,
    clipping_replay,
    mechanism_gate,
    select_fresh_primary,
)
from priors.gradient_routing import CHECKPOINTS, PARAMETER_GROUPS, stable_seed
import tools.diagnose_hoi_d2j as diagnostic
from tools.summarize_hoi_d2j import RUN_ID, validate_run_identity


class D2JSelectionTests(unittest.TestCase):
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
        selected = set(first["global_indices"])
        self.assertFalse(selected & first["d2h_global_indices"])
        self.assertFalse(selected & first["d2i_global_indices"])

    def test_paired_rng_is_checkpoint_independent_and_block_bound(self):
        label = "D2J:primary:499:0"
        self.assertEqual(stable_seed(label), stable_seed(label))
        self.assertNotEqual(stable_seed(label), stable_seed("D2J:primary:499:1"))
        self.assertNotIn("R-1024", label)


class D2JGeometryTests(unittest.TestCase):
    def _base(self):
        return {
            name: (torch.tensor([float(index), float(index + 1)]),)
            for index, name in enumerate(BASE_COMPONENTS, start=1)
        }

    def test_production_clip_formula_replay(self):
        result = clipping_replay(100.0)
        self.assertEqual(result["max_norm"], GRADIENT_CLIP_NORM)
        self.assertAlmostEqual(result["clip_coefficient"], 1.0 / (100.0 + 1e-6))
        self.assertLessEqual(result["formula_replay_max_abs"], 1e-6)
        unclipped = clipping_replay(0.5)
        self.assertEqual(unclipped["clip_coefficient"], 1.0)
        self.assertLessEqual(unclipped["formula_replay_max_abs"], 1e-6)

    def test_field_complete_formula_and_group_reporting(self):
        base = self._base()
        direct = (sum(base[name][0] for name in BASE_COMPONENTS),)
        groups = {name: (0,) for name in PARAMETER_GROUPS}
        result = clip_gradient_geometry(base, direct, groups)
        self.assertTrue(result["finite"])
        self.assertLessEqual(result["total_gradient_formula_relative_l2"], 1e-12)
        self.assertEqual(set(result["groups"]), set(PARAMETER_GROUPS))
        self.assertTrue(set(FIELD_COMPONENTS).issubset(LOSS_COMPONENTS))
        for record in result["groups"].values():
            self.assertEqual(set(record["gradient_l2_norm"]), set(LOSS_COMPONENTS))
            self.assertEqual(set(record["cosine_matrix"]), set(LOSS_COMPONENTS))
            self.assertTrue(math.isfinite(record["human_directional_efficiency"]["value"]))

    def test_formula_mismatch_and_nonfinite_clip_input_are_rejected(self):
        base = self._base()
        direct = (sum(base[name][0] for name in BASE_COMPONENTS) + 1.0,)
        groups = {name: (0,) for name in PARAMETER_GROUPS}
        self.assertGreater(
            clip_gradient_geometry(base, direct, groups)["total_gradient_formula_relative_l2"],
            1e-5,
        )
        with self.assertRaises(ValueError):
            clipping_replay(float("nan"))


class D2JGateTests(unittest.TestCase):
    @staticmethod
    def _candidate(*, preclip=100.0, coefficient=0.01, human=0.05, objects=0.3):
        blocks = [{
            "finite": True,
            "total_gradient_formula_relative_l2": 0.0,
            "clipping": {
                "preclip_norm": preclip,
                "clip_coefficient": coefficient,
                "formula_replay_max_abs": 0.0,
            },
            "groups": {"all_parameters": {
                "human_directional_efficiency": {"value": human, "defined": True},
                "object_directional_efficiency": {"value": objects, "defined": True},
            }},
        } for _ in range(8)]
        return {
            "finite": True,
            "state_dict_sha256_before": "same",
            "state_dict_sha256_after": "same",
            "parameter_grad_buffers_clear": True,
            "timesteps": {str(t): {"blocks": blocks} for t in GATE_TIMESTEPS},
        }

    def test_gate_requires_both_checkpoints_and_every_conjunct(self):
        candidates = {name: self._candidate() for name in CHECKPOINTS}
        decision = mechanism_gate(candidates)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["classification"], "gradient-clip-routing-positive-stop")
        self.assertFalse(decision["training_authorized"])
        candidates["R-3072"] = self._candidate(objects=0.1)
        decision = mechanism_gate(candidates)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["classification"], "gradient-clip-routing-negative-stop")

    def test_source_is_zero_update_and_production_clip_call_is_unchanged(self):
        source = inspect.getsource(diagnostic)
        training = (ROOT / "code/train_hoi_prior.py").read_text(encoding="utf-8")
        self.assertIn("torch.autograd.grad", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn(".step(", source)
        self.assertIn('"training_updates": 0', source)
        self.assertIn(
            "torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip_norm))",
            training,
        )

    def test_diagnostic_is_directly_executable(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/diagnose_hoi_d2j.py"), "--help"],
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
