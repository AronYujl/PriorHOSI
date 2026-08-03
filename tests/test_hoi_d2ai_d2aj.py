"""Targeted tests for the D2-AI full-budget and D2-AJ goal-pathway arms.

Preregistered in docs/EXPERIMENT_PLAN.md,
"2026-08-03 Phase 1B D2-AI 全预算与 D2-AJ 目标条件通路（双臂，用户批准）"
and experiments/registry.jsonl rows
p1-hoi-d2a{i,j}-*-preregister-s42-20260803.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AJ,
    HOI_D2AJ_CONDITION_TOKENS,
    _HOICleanMotionNetwork,
    build_expert,
    load_trained_hoi_prior,
)

SEALED_D2X_PARAMETERS = 29_673_448
D2AJ_PARAMETER_DELTA = 525_312


def _network(variant, seed=0):
    torch.manual_seed(seed)
    return _HOICleanMotionNetwork(512, 16, 8, architecture_variant=variant)


def _batch(batch=3, seed=1):
    torch.manual_seed(seed)
    goals = torch.randn(batch, 9)
    # Match the real convention: goals[3:6] and pelvis y are always zero.
    goals[:, 3:6] = 0.0
    goals[:, 1] = 0.0
    return {
        "noisy": torch.randn(batch, 16, 232),
        "timesteps": torch.randint(0, 500, (batch,)),
        "text_embedding": torch.randn(batch, 768),
        "object_bps": torch.randn(batch, 1024, 3),
        "goals": goals,
        "progress": torch.randn(batch, 3),
    }


class TestD2AJArchitecture(unittest.TestCase):
    def test_base_architecture_unchanged(self):
        """Arm 1 must be provably unaffected by the Arm 2 edit."""
        base = _network(HOI_ARCHITECTURE_BASE)
        self.assertEqual(
            sum(p.numel() for p in base.parameters()), SEALED_D2X_PARAMETERS
        )
        self.assertEqual(base.condition_tokens, 4)
        self.assertEqual(tuple(base.position.shape), (1, 20, 512))
        self.assertIsNotNone(base.goal_progress)
        self.assertIsNone(base.pelvis_goal)

    def test_parameter_delta(self):
        base = sum(p.numel() for p in _network(HOI_ARCHITECTURE_BASE).parameters())
        d2aj = sum(p.numel() for p in _network(HOI_ARCHITECTURE_D2AJ).parameters())
        self.assertEqual(d2aj - base, D2AJ_PARAMETER_DELTA)

    def test_output_contract(self):
        for variant in (HOI_ARCHITECTURE_BASE, HOI_ARCHITECTURE_D2AJ):
            net = _network(variant).eval()
            with torch.no_grad():
                out = net(**_batch())
            self.assertEqual(tuple(out.shape), (3, 16, 232), variant)
            self.assertTrue(torch.isfinite(out).all(), variant)

    def test_condition_and_position_tokens(self):
        net = _network(HOI_ARCHITECTURE_D2AJ)
        self.assertEqual(net.condition_tokens, HOI_D2AJ_CONDITION_TOKENS)
        self.assertEqual(tuple(net.position.shape), (1, 16 + 6, 512))
        self.assertIsNone(net.goal_progress)
        for module in (net.pelvis_goal, net.object_goal, net.progress):
            self.assertIsNotNone(module)

    def test_determinism(self):
        net = _network(HOI_ARCHITECTURE_D2AJ).eval()
        payload = _batch()
        with torch.no_grad():
            first = net(**payload)
            second = net(**payload)
        self.assertTrue(torch.equal(first, second))

    def test_dead_goal_dims_are_not_read(self):
        """D2-AJ must ignore goals[3:6] and the pelvis y channel."""
        net = _network(HOI_ARCHITECTURE_D2AJ).eval()
        payload = _batch()
        perturbed = dict(payload)
        goals = payload["goals"].clone()
        goals[:, 3:6] = torch.randn(goals.shape[0], 3)
        goals[:, 1] = torch.randn(goals.shape[0])
        perturbed["goals"] = goals
        with torch.no_grad():
            self.assertTrue(torch.equal(net(**payload), net(**perturbed)))

    def test_base_does_read_dead_dims(self):
        """The confound the preregistered diagnostic separates must be real."""
        net = _network(HOI_ARCHITECTURE_BASE).eval()
        payload = _batch()
        perturbed = dict(payload)
        goals = payload["goals"].clone()
        goals[:, 3:6] = torch.randn(goals.shape[0], 3)
        perturbed["goals"] = goals
        with torch.no_grad():
            self.assertFalse(torch.equal(net(**payload), net(**perturbed)))

    def test_goal_separability(self):
        """Zeroing the pelvis goal and the object goal must differ.

        This is the mechanism D2-AJ claims; a single fused token cannot do it.
        """
        net = _network(HOI_ARCHITECTURE_D2AJ).eval()
        payload = _batch()
        pelvis_zero = dict(payload)
        goals = payload["goals"].clone()
        goals[:, 0] = 0.0
        goals[:, 2] = 0.0
        pelvis_zero["goals"] = goals
        object_zero = dict(payload)
        goals = payload["goals"].clone()
        goals[:, 6:9] = 0.0
        object_zero["goals"] = goals
        with torch.no_grad():
            reference = net(**payload)
            pelvis_delta = (net(**pelvis_zero) - reference).abs().mean().item()
            object_delta = (net(**object_zero) - reference).abs().mean().item()
        self.assertGreater(pelvis_delta, 0.0)
        self.assertGreater(object_delta, 0.0)
        self.assertGreater(abs(pelvis_delta - object_delta), 1e-6)


class TestD2AJCheckpointRoundTrip(unittest.TestCase):
    def _contract(self, network):
        return {
            "architecture_variant": HOI_ARCHITECTURE_D2AJ,
            "condition_tokens": int(network.condition_tokens),
            "position_tokens": int(network.position.shape[1]),
            "goals_3_to_6_read": False,
            "pelvis_goal_y_read": False,
            "fused_goal_progress_module_present": network.goal_progress is not None,
            "parameter_delta_vs_base": D2AJ_PARAMETER_DELTA,
        }

    def _payload(self, model, contract=None):
        model_config = {"dim_model": 512, "num_heads": 16, "num_layers": 8}
        value = {
            "schema_version": 2,
            "checkpoint_type": "hoi_prior_phase1b",
            "window_state_codec": "state-compositional-v1",
            "expert": "hoi",
            "initialization": "random",
            "run_id": "p1-hoi-d2aj-split-goal-tokens-s42-20260803",
            "seed": 42,
            "model": model.state_dict(),
            "primary_weight_variant": "online",
            "ema_models": {},
        }
        if contract is not None:
            model_config["architecture_variant"] = HOI_ARCHITECTURE_D2AJ
            value["architecture_variant"] = HOI_ARCHITECTURE_D2AJ
            value["split_goal_token_contract"] = contract
        value["model_config"] = model_config
        return value

    def _round_trip(self, payload):
        handle = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        handle.close()
        try:
            torch.save(payload, handle.name)
            return load_trained_hoi_prior(
                handle.name, torch.device("cpu"), weight_variant="online"
            )
        finally:
            os.unlink(handle.name)

    def test_round_trip_preserves_variant_and_weights(self):
        torch.manual_seed(42)
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AJ,
        )
        loaded, _ = self._round_trip(
            self._payload(model, self._contract(model.network))
        )
        self.assertEqual(loaded.network.condition_tokens, HOI_D2AJ_CONDITION_TOKENS)
        self.assertEqual(tuple(loaded.network.position.shape), (1, 22, 512))
        for saved, restored in zip(
            model.state_dict().values(), loaded.state_dict().values()
        ):
            self.assertTrue(torch.equal(saved, restored))

    def test_missing_contract_is_rejected(self):
        """A D2-AJ checkpoint without provenance must not load as base."""
        torch.manual_seed(42)
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AJ,
        )
        payload = self._payload(model, self._contract(model.network))
        del payload["split_goal_token_contract"]
        with self.assertRaises(ValueError):
            self._round_trip(payload)

    def test_tampered_contract_is_rejected(self):
        torch.manual_seed(42)
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AJ,
        )
        contract = self._contract(model.network)
        contract["goals_3_to_6_read"] = True
        with self.assertRaises(ValueError):
            self._round_trip(self._payload(model, contract))


class TestLongBudgetConfigs(unittest.TestCase):
    """Both arms must inherit D2-X's recipe with budget as the only difference."""

    def setUp(self):
        os.environ.setdefault("ROOT_DIR", str(CODE.parent))
        self.config_dir = str((CODE / "config").resolve())
        import train_hoi_prior

        self.train = train_hoi_prior

    def _compose(self, name):
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            return compose(config_name=name)

    def test_both_arms_share_budget_and_recipe(self):
        for name in (
            "config_train_hoi_prior_d2ai", "config_train_hoi_prior_d2aj",
        ):
            cfg = self._compose(name)
            self.assertEqual(int(cfg.max_processed_windows), 299_520_000, name)
            self.assertEqual(int(cfg.effective_batch_size), 2048, name)
            self.assertEqual(int(cfg.batch_size), 512, name)
            self.assertEqual(int(cfg.num_gpus), 4, name)
            self.assertEqual(int(cfg.seed), 42, name)
            self.assertEqual(float(cfg.learning_rate), 1e-4, name)
            self.assertEqual(int(cfg.warmup_windows), 0, name)
            self.assertEqual(float(cfg.minimum_lr_ratio), 1.0, name)
            self.assertEqual(list(cfg.ema_decays), [], name)
            self.assertFalse(bool(cfg.amp), name)
            self.assertEqual(str(cfg.optimizer_name), "Adam", name)
            self.assertEqual(int(cfg.checkpoint_interval_windows), 3_072_000, name)
            # 146,250 updates exactly, and 61.44M is a cadence multiple.
            self.assertEqual(
                int(cfg.max_processed_windows) % int(cfg.effective_batch_size), 0, name
            )
            self.assertEqual(61_440_000 % int(cfg.checkpoint_interval_windows), 0, name)

    def test_arms_stay_outside_the_d2x_contract(self):
        """_is_d2x must be false, or _validate_d2x_contract rejects the budget."""
        for name in (
            "config_train_hoi_prior_d2ai", "config_train_hoi_prior_d2aj",
        ):
            cfg = self._compose(name)
            self.assertFalse(self.train._is_d2x(cfg), name)
            self.assertTrue(bool(cfg.fk_foot_temporal_routing), name)
            self.assertTrue(self.train._uses_author_update_rule(cfg), name)
            weights = self.train._locked_loss_weights(cfg)
            self.assertAlmostEqual(weights["fk"], 0.3569973401779424, places=15, msg=name)
            self.assertAlmostEqual(
                weights["object_surface"], 0.4772322188400037, places=15, msg=name
            )
            self.train._validate_fk_foot_temporal_routing_mode(cfg)

    def test_modes_are_mutually_exclusive(self):
        for name, predicate in (
            ("config_train_hoi_prior_d2ai", "_is_d2ai"),
            ("config_train_hoi_prior_d2aj", "_is_d2aj"),
        ):
            cfg = self._compose(name)
            active = [
                flag for flag in (
                    "_is_d2ac", "_is_d2ad", "_is_d2ae", "_is_d2af", "_is_d2ag",
                    "_is_d2ai", "_is_d2aj",
                )
                if getattr(self.train, flag)(cfg)
            ]
            self.assertEqual(active, [predicate], name)

    def test_architecture_variant_per_arm(self):
        self.assertEqual(
            str(self._compose("config_train_hoi_prior_d2ai").hoi_architecture_variant),
            HOI_ARCHITECTURE_BASE,
        )
        self.assertEqual(
            str(self._compose("config_train_hoi_prior_d2aj").hoi_architecture_variant),
            HOI_ARCHITECTURE_D2AJ,
        )

    def test_model_config_records_variant_for_d2aj_only(self):
        d2ai = self.train._model_config(self._compose("config_train_hoi_prior_d2ai"))
        d2aj = self.train._model_config(self._compose("config_train_hoi_prior_d2aj"))
        self.assertNotIn("architecture_variant", d2ai)
        self.assertEqual(d2aj.get("architecture_variant"), HOI_ARCHITECTURE_D2AJ)


if __name__ == "__main__":
    unittest.main()
