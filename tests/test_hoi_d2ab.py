import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.d2ab import (  # noqa: E402
    D2AB_CLEARANCE_SCALE_M,
    D2AB_FOOT_JOINTS,
    D2AB_METADATA_SCHEMA,
    D2AB_METADATA_RUN_ID,
    D2AB_PAIR_INDEX,
    D2AB_SAMPLE_INTERVAL_S,
    D2AB_VELOCITY_SCALE_S_PER_M,
    _d2ab_physical_terms,
    d2ab_hoi_training_losses,
    d2ab_velocity_loss,
    load_train_floor_map,
    sequence_floor_and_clearance,
)
from priors.losses import D2X_FOOT_XZ_VELOCITY_SLOTS  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    _loss_routing_contract,
    _resume_contract,
    _validate_d2ab_contract,
    _validate_fk_foot_temporal_routing_mode,
)
from tools import run_hoi_d2ab_evaluation as d2ab_evaluation  # noqa: E402
from tools import run_hoi_d2x_evaluation as shared_evaluation  # noqa: E402
from tools.diagnose_hoi_d2ab import _aggregate_comparison, _per_sequence  # noqa: E402
from tools.run_hoi_d2ab_evaluation import (  # noqa: E402
    CONTROL_CHECKPOINT_SHA256,
    EXPECTED_INITIAL_MODEL_STATE_SHA256,
    INTERNAL_RUN_ID,
    INTERNAL_SELECTION_SHA256,
    SUPPORT_METADATA_SHA256,
    TRAINING_RUN_ID,
    validate_training_result,
)
from tools.smoke_hoi_d2ab import _support_record  # noqa: E402


METADATA = ROOT / "experiments/metadata/omomo_hoi_d2ab_train_support_seed42.json"
METADATA_SHA256 = "807978580221910ad00260c2dff4f33ddacbb1bf72bad7443bf21ac48f31f079"
SPLIT = ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"


def merged_config():
    base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
    intervention = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ab.yaml")
    cfg = OmegaConf.merge(base, intervention)
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(SPLIT)
    cfg.d2ab_support_metadata_path = str(METADATA)
    return cfg


class D2ABMetadataTests(unittest.TestCase):
    def test_metadata_hash_schema_and_constants_are_locked(self):
        actual = hashlib.sha256(METADATA.read_bytes()).hexdigest()
        self.assertEqual(actual, METADATA_SHA256)
        value = json.loads(METADATA.read_text(encoding="utf-8"))
        self.assertEqual(value["schema"], D2AB_METADATA_SCHEMA)
        self.assertEqual(value["run_id"], D2AB_METADATA_RUN_ID)
        self.assertEqual(value["seed"], 42)
        self.assertEqual(value["partition"], "train")
        self.assertEqual(value["split"]["sha256"], "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e")
        self.assertEqual(value["split"]["sequence_count"], 4088)
        self.assertEqual(
            value["statistics"]["strictly_positive_clearance_median_m"],
            D2AB_CLEARANCE_SCALE_M,
        )
        self.assertEqual(
            value["constants"]["velocity_scale_s_per_m"],
            D2AB_VELOCITY_SCALE_S_PER_M,
        )
        self.assertFalse(value["official_test_used"])

    def test_metadata_floor_map_covers_train_sequences(self):
        split = json.loads(SPLIT.read_text(encoding="utf-8"))
        floors = load_train_floor_map(
            METADATA,
            METADATA_SHA256,
            split_path=SPLIT,
            expected_train_sequence_indices=split["train"]["sequence_indices"],
        )
        self.assertEqual(len(floors), 4088)
        self.assertAlmostEqual(min(floors.values()), -0.004783304338343441, places=15)
        self.assertAlmostEqual(max(floors.values()), 0.06221588589251041, places=15)

    def test_floor_formula_is_toe_quantile_and_four_foot_clearance(self):
        joints = np.zeros((4, 28, 3), dtype=np.float64)
        joints[:, 10, 1] = [0.0, 1.0, 2.0, 3.0]
        joints[:, 11, 1] = [0.5, 1.5, 2.5, 3.5]
        joints[:, 7, 1] = [0.2, 1.2, 2.2, 3.2]
        joints[:, 8, 1] = [0.3, 1.3, 2.3, 3.3]
        floor, clearance = sequence_floor_and_clearance(joints, 0, 4)
        self.assertAlmostEqual(floor, 0.175, places=12)
        self.assertEqual(clearance.shape, (16,))
        self.assertEqual(int((clearance > 0.0).sum()), 15)


class D2ABLossTests(unittest.TestCase):
    def _terms(self):
        batch, frames = 1, 16
        minimum = torch.full((3,), -10.0)
        maximum = torch.full((3,), 10.0)
        target = torch.zeros(batch, frames, 232)
        # Encode direct positions from known physical coordinates.
        positions = torch.zeros(batch, frames, 28, 3)
        positions[..., 7, 1] = 0.10
        positions[..., 10, 1] = 0.20
        positions[..., 8, 1] = 0.30
        positions[..., 11, 1] = 0.40
        target[..., :84] = ((positions - minimum) / (maximum - minimum) * 2.0 - 1.0).reshape(
            batch, frames, 84,
        )
        prediction = target.clone().requires_grad_(True)
        supervision = target.clone().requires_grad_(True)
        predicted_fk = positions[..., :24, :].clone().requires_grad_(True)
        terms = _d2ab_physical_terms(
            prediction,
            supervision,
            predicted_fk,
            minimum,
            maximum,
            torch.zeros(batch),
        )
        return supervision, predicted_fk, terms

    def test_pair_order_and_first_previous_are_locked(self):
        target, predicted_fk, terms = self._terms()
        support = terms["support_pair"][0, 0]
        # Left pair is [7,10], right pair is [8,11], hence left is lower here.
        self.assertGreater(float(support[0]), float(support[1]))
        original = terms["predicted_velocity"].detach().clone()
        predicted_fk.data[:, 1, D2AB_FOOT_JOINTS, 0] += 100.0
        changed = _d2ab_physical_terms(
            target.clone().requires_grad_(True),
            target,
            predicted_fk,
            torch.full((3,), -10.0),
            torch.full((3,), 10.0),
            torch.zeros(1),
        )["predicted_velocity"]
        # Frame 1 is immutable GT history and cannot affect the first residual.
        self.assertTrue(torch.equal(original[:, 0], changed[:, 0]))

    def test_support_is_differentiable_and_floor_is_detached(self):
        target, predicted_fk, terms = self._terms()
        self.assertTrue(terms["support_pair"].requires_grad)
        self.assertFalse(terms["pair_distance"].grad_fn is None)
        terms["support_pair"].sum().backward()
        self.assertIsNotNone(predicted_fk.grad)
        self.assertGreater(float(predicted_fk.grad.abs().sum()), 0.0)
        self.assertIsNone(target.grad)

    def test_only_eight_routed_slots_are_replaced(self):
        prediction = torch.randn(1, 14, 87, requires_grad=True)
        target = torch.randn_like(prediction)
        predicted_velocity = torch.randn(1, 14, 4, 2, requires_grad=True)
        target_velocity = torch.randn_like(predicted_velocity)
        support = torch.full((1, 14, 4), 0.5, requires_grad=True)
        loss, _ = d2ab_velocity_loss(
            prediction, target, predicted_velocity, target_velocity, support,
        )
        loss.backward()
        routed = set(D2X_FOOT_XZ_VELOCITY_SLOTS)
        for slot in range(87):
            if slot in routed:
                self.assertEqual(float(prediction.grad[..., slot].abs().sum()), 0.0)
            else:
                self.assertGreater(float(prediction.grad[..., slot].abs().sum()), 0.0)

    def test_full_velocity_routes_foot_xz_away_from_direct_channels(self):
        generator = torch.Generator().manual_seed(42)
        prediction = torch.randn(2, 16, 232, generator=generator).requires_grad_(True)
        target = torch.randn(2, 16, 232, generator=generator)
        offsets = torch.randn(2, 24, 3, generator=generator) * 0.05
        losses = d2ab_hoi_training_losses(
            prediction,
            target,
            torch.zeros(2, 9),
            offsets,
            torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long),
            torch.full((3,), -10.0),
            torch.full((3,), 10.0),
            torch.full((3,), -10.0),
            torch.full((3,), 10.0),
            torch.zeros(2, dtype=torch.bool),
            torch.randn(2, 8, 3, generator=generator),
            torch.eye(3).repeat(2, 1, 1),
            torch.eye(3).repeat(2, 1, 1),
            torch.zeros(2),
            fk_weight=0.3569973401779424,
            object_surface_weight=0.4772322188400037,
            velocity_weight=0.1,
            goal_weight=1.0,
        )
        losses["velocity"].backward()
        direct_routed_channels = list(D2X_FOOT_XZ_VELOCITY_SLOTS)
        self.assertEqual(
            float(prediction.grad[..., direct_routed_channels].abs().sum()),
            0.0,
        )
        self.assertGreater(float(prediction.grad[..., :3].abs().sum()), 0.0)
        self.assertGreater(
            float(prediction.grad[..., 84:216].abs().sum()),
            0.0,
        )

    def test_zero_and_one_support_targets(self):
        prediction = torch.zeros(1, 14, 87, requires_grad=True)
        target = torch.zeros_like(prediction)
        velocity = torch.ones(1, 14, 4, 2)
        target_velocity = torch.ones_like(velocity) * 3.0
        zero = torch.zeros(1, 14, 4, requires_grad=True)
        one = torch.ones(1, 14, 4, requires_grad=True)
        _, residual_zero = d2ab_velocity_loss(
            prediction, target, velocity, target_velocity, zero,
        )
        _, residual_one = d2ab_velocity_loss(
            prediction, target, velocity, target_velocity, one,
        )
        self.assertTrue(torch.allclose(residual_zero, velocity - target_velocity))
        self.assertTrue(torch.allclose(residual_one, velocity))

    def test_smoke_support_contract_rejects_only_degenerate_mass(self):
        varied = torch.cat(
            (
                torch.full((20,), 1.0e-4),
                torch.full((20,), 0.2),
                torch.full((20,), 0.8),
            )
        )
        record = _support_record(varied)
        self.assertTrue(record["noncollapsed"])
        self.assertGreaterEqual(record["fraction_gt_0.05"], 0.05)
        self.assertGreaterEqual(record["fraction_lt_0.95"], 0.05)
        self.assertFalse(_support_record(torch.zeros(60))["noncollapsed"])
        self.assertFalse(_support_record(torch.ones(60))["noncollapsed"])

    def test_internal_reduces_96_windows_to_32_sequences_and_uses_schema(self):
        values = torch.arange(96 * 14 * 4, dtype=torch.float32).reshape(96, 14, 4)
        reduced = _per_sequence(values)
        expected = values.reshape(96, -1).mean(dim=1).reshape(32, 3).mean(dim=1)
        self.assertEqual(len(reduced), 32)
        self.assertTrue(torch.allclose(torch.tensor(reduced), expected))
        control = {
            "timesteps": {
                str(timestep): {
                    "supported_velocity_m2_s2_by_sequence": [2.0] * 32,
                    "no_slip_residual_m2_s2_by_sequence": [2.0] * 32,
                    "support_mass_by_sequence": [1.0] * 32,
                }
                for timestep in (249, 499)
            }
        }
        target = {
            "timesteps": {
                str(timestep): {
                    "supported_velocity_m2_s2_by_sequence": [1.0] * 32,
                    "no_slip_residual_m2_s2_by_sequence": [1.0] * 32,
                    "support_mass_by_sequence": [1.0] * 32,
                }
                for timestep in (249, 499)
            }
        }
        comparison = _aggregate_comparison(control, target)
        self.assertTrue(comparison["contract_passed"])
        self.assertTrue(comparison["mechanism_passed"])
        self.assertTrue(comparison["support_sanity_passed"])


class D2ABContractTests(unittest.TestCase):
    def test_config_contract_and_routing_contract(self):
        cfg = merged_config()
        _validate_fk_foot_temporal_routing_mode(cfg)
        _validate_d2ab_contract(cfg, 4)
        routing = _loss_routing_contract(cfg)
        self.assertTrue(routing["d2ab_predicted_support_no_slip"])
        self.assertEqual(routing["weighted_slots"], 8)
        self.assertEqual(routing["total_velocity_slots"], 87)
        self.assertEqual(
            _resume_contract(cfg)["d2ab_support_metadata_sha256"],
            METADATA_SHA256,
        )

    def test_other_modes_reject_d2ab_fields(self):
        cfg = merged_config()
        cfg.d2ab_predicted_support_no_slip = False
        cfg.fk_foot_temporal_routing = False
        with self.assertRaisesRegex(ValueError, "support metadata"):
            _validate_fk_foot_temporal_routing_mode(cfg)

    def test_evaluation_locks_training_and_internal_artifact_contracts(self):
        target_sha256 = "a" * 64
        expected_checkpoint = (
            f"{TRAINING_RUN_ID}_windows061440000.pth"
        )
        routing = _loss_routing_contract(merged_config())
        metrics = {
            "status": "stable",
            "run_id": TRAINING_RUN_ID,
            "seed": 42,
            "initialization": "random",
            "training_start": "random",
            "released_checkpoint_used": False,
            "processed_windows": 61_440_000,
            "optimizer_updates": 30_000,
            "world_size": 4,
            "effective_batch_size": 2048,
            "optimization_contract": {
                "optimizer": "Adam",
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "learning_rate": 0.0001,
                "scheduler": "none",
                "warmup_windows": 0,
                "gradient_clipping": False,
                "gradient_clip_norm": None,
                "amp": False,
                "ema_decays": [],
                "primary_weight_variant": "online",
            },
            "loss_weights": {
                "fk": 0.3569973401779424,
                "object_surface": 0.4772322188400037,
                "velocity": 0.1,
                "terminal_object_goal": 1.0,
            },
            "loss_routing": routing,
            "support_metadata": {
                "path": str(METADATA.resolve()),
                "sha256": SUPPORT_METADATA_SHA256,
            },
            "ema_decays": [],
            "primary_weight_variant": "online",
            "weight_initialization": {
                "mode": "random",
                "restored_components": [],
                "source_checkpoint": None,
                "initial_model_state_sha256": EXPECTED_INITIAL_MODEL_STATE_SHA256,
            },
            "checkpoint_hashes": [{
                "processed_windows": 61_440_000,
                "sha256": target_sha256,
                "path": expected_checkpoint,
            }],
        }
        internal = {
            "schema_version": 1,
            "status": "completed",
            "run_id": INTERNAL_RUN_ID,
            "selection": {
                "sha256": INTERNAL_SELECTION_SHA256,
                "sequences": 32,
                "windows": 96,
            },
            "control_checkpoint": {"sha256": CONTROL_CHECKPOINT_SHA256},
            "target_checkpoint": {"sha256": target_sha256},
            "support_metadata": {"sha256": SUPPORT_METADATA_SHA256},
            "pairing": {
                "same_clean_windows": True,
                "same_timestep": True,
                "same_noise": True,
                "same_condition_dropout": True,
            },
            "comparison": {
                "contract_passed": True,
                "finite": True,
                "mechanism_passed": True,
                "support_sanity_passed": True,
                "mechanism_checks": {"249": True, "499": True},
                "support_sanity": {},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            internal_path = root / "internal.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            internal_path.write_text(json.dumps(internal), encoding="utf-8")
            args = SimpleNamespace(
                target_checkpoint=root / expected_checkpoint,
                target_sha256=target_sha256,
                training_metrics=metrics_path,
                internal_diagnostic=internal_path,
                support_metadata=METADATA,
            )
            result = validate_training_result(args)
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(
            result["internal_diagnostic"]["optimization_gate_passed"]
        )

    def test_evaluation_wrapper_uses_saved_shared_resolver(self):
        shared_names = (
            "RUN_ID",
            "SUBPHASE",
            "CONTROL_CHECKPOINT_SHA256",
            "CONTROL_AGGREGATE_SHA256",
            "CONTROL_PER_SEQUENCE_SHA256",
            "parse_args",
            "resolved_config",
            "additional_runtime_artifact_hashes",
            "validate_training_result",
            "compare_records",
            "classify",
        )
        original = {
            name: getattr(shared_evaluation, name)
            for name in shared_names
        }
        try:
            d2ab_evaluation.configure_shared_module()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                output.mkdir()
                (output / "resolved_target.yaml").write_text(
                    "seed: 42\n", encoding="utf-8",
                )
                args = SimpleNamespace(
                    python=Path(sys.executable),
                    target_checkpoint=root / "target.pth",
                    target_sha256="a" * 64,
                    training_metrics=root / "training.json",
                    training_metrics_sha256="b" * 64,
                    internal_diagnostic=root / "internal.json",
                    internal_diagnostic_sha256="c" * 64,
                    support_metadata=METADATA,
                    support_metadata_sha256=SUPPORT_METADATA_SHA256,
                    control_aggregate=root / "control-aggregate.json",
                    control_per_sequence=root / "control-sequence.json",
                    baseline=root / "baseline.json",
                    output=output,
                    metrics=root / "metrics.json",
                    resolved_config=root / "resolved.json",
                    device="cuda:0",
                )
                resolved = shared_evaluation.resolved_config(args)
            self.assertEqual(resolved["run_id"], d2ab_evaluation.RUN_ID)
            self.assertEqual(
                resolved["internal_diagnostic"]["sha256"],
                "c" * 64,
            )
            self.assertEqual(
                resolved["support_metadata"]["registered_sha256"],
                SUPPORT_METADATA_SHA256,
            )
        finally:
            for name, value in original.items():
                setattr(shared_evaluation, name, value)


class D2ABGovernanceTests(unittest.TestCase):
    def test_plan_and_registry_lock_first_budget_and_stop_rules(self):
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-AB predicted-support no-slip objective", plan)
        self.assertIn("p1-hoi-d2ab-predicted-support-no-slip-s42-20260725", plan)
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        record = next(
            item for item in records
            if item["experiment_id"]
            == "p1-hoi-d2ab-predicted-support-no-slip-preregister-s42-20260725"
        )
        self.assertEqual(record["config"]["campaign_budget"]["this_subphase_uses_training_number"], 1)
        self.assertFalse(record["config"]["conditional_fallback"]["authorized"])
        self.assertFalse(record["config"]["consistency_authorized"])
        self.assertFalse(record["config"]["checkpoint_selection"])


if __name__ == "__main__":
    unittest.main()
