from __future__ import annotations

import inspect
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    ALPHA_BAR_SHA256,
    BETA_SHA256,
    SQRT_ALPHA_BAR_SENTINELS,
    SQRT_ALPHA_BAR_SHA256,
    canonical_diffusion_schedule,
    tensor_sha256,
    validate_canonical_diffusion_schedule,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AE,
    HOI_ARCHITECTURE_D2AF,
    build_expert,
    load_trained_hoi_prior,
)
from priors.sparse_relation import (  # noqa: E402
    D2AF_DIAGNOSTIC_VARIANTS,
    SPARSE_RELATION_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT,
    SparseCurrentStateRelationField,
    diffusion_reliability_contract_metadata,
    validate_diffusion_reliability_contract,
)
from tools.diagnose_hoi_d2ae import synthetic_inputs  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AF_MAXIMUM_ETA_HOURS,
    D2AF_MINIMUM_THROUGHPUT,
    _forward_losses,
    _state_dict_sha256,
    _validate_d2af_contract,
    _validate_fk_foot_temporal_routing_mode,
)


EXPECTED_INITIAL_STATE_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)


def relation_arguments(batch: int):
    values = synthetic_inputs(batch=batch)
    return values, {
        "rest_object_points": values["rest_object_points"],
        "world_to_local_rotation": values["world_to_local_rotation"],
        "object_rotation_reference": values["object_rotation_reference"],
        "position_minimum": values["position_minimum"],
        "position_maximum": values["position_maximum"],
        "object_minimum": values["object_minimum"],
        "object_maximum": values["object_maximum"],
    }


class D2AFScheduleTests(unittest.TestCase):
    def test_canonical_schedule_hashes_sentinels_and_diffusion_parity(self):
        schedule = canonical_diffusion_schedule()
        contract = validate_canonical_diffusion_schedule(schedule)
        self.assertEqual(tensor_sha256(schedule["betas"]), BETA_SHA256)
        self.assertEqual(tensor_sha256(schedule["alpha_bar"]), ALPHA_BAR_SHA256)
        self.assertEqual(
            tensor_sha256(schedule["sqrt_alpha_bar"]),
            SQRT_ALPHA_BAR_SHA256,
        )
        for index, expected in SQRT_ALPHA_BAR_SENTINELS.items():
            self.assertEqual(float(schedule["sqrt_alpha_bar"][index]), expected)
        self.assertTrue(torch.all(
            schedule["sqrt_alpha_bar"][1:] < schedule["sqrt_alpha_bar"][:-1]
        ))
        diffusion = GaussianDiffusion()
        self.assertTrue(torch.equal(
            diffusion.sqrt_alpha_bar.cpu(),
            schedule["sqrt_alpha_bar"],
        ))
        self.assertEqual(
            contract["sqrt_alpha_bar_sha256"],
            SQRT_ALPHA_BAR_SHA256,
        )

    def test_schedule_constructor_is_the_only_project_linear_schedule(self):
        hits = []
        for path in (ROOT / "code").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "torch.linspace(" in source and "BETA_START" in source:
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(hits, ["code/priors/diffusion_schedule.py"])


class D2AFFieldTests(unittest.TestCase):
    def test_parameter_and_seed42_state_are_exactly_d2ae(self):
        torch.manual_seed(42)
        d2ae = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AE,
        )
        torch.manual_seed(42)
        d2af = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AF,
        )
        self.assertEqual(sum(p.numel() for p in d2af.parameters()), TOTAL_PARAMETER_COUNT)
        self.assertEqual(
            sum(p.numel() for p in d2af.network.sparse_relation_field.parameters()),
            SPARSE_RELATION_PARAMETER_COUNT,
        )
        self.assertEqual(d2ae.state_dict().keys(), d2af.state_dict().keys())
        for key, value in d2ae.state_dict().items():
            self.assertTrue(torch.equal(value, d2af.state_dict()[key]), key)
        self.assertEqual(_state_dict_sha256(d2af.state_dict()), EXPECTED_INITIAL_STATE_SHA256)
        self.assertNotIn(
            "network.sparse_relation_field.sqrt_alpha_bar",
            d2af.state_dict(),
        )

    def test_mixed_timestep_scaling_and_unit_rho_counterfactual(self):
        field = SparseCurrentStateRelationField(
            512,
            diffusion_reliability=True,
        ).eval()
        field.set_gate_override(0.1)
        values, arguments = relation_arguments(3)
        motion = torch.randn(
            3, 16, 512, generator=torch.Generator().manual_seed(8),
        )
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        scheduled = field(
            motion,
            values["current"],
            **arguments,
            timesteps=timesteps,
        )
        field.set_rho_override(1.0)
        unit = field(
            motion,
            values["current"],
            **arguments,
            timesteps=timesteps,
        )
        rho = canonical_diffusion_schedule()["sqrt_alpha_bar"][timesteps]
        expected = rho[:, None, None] * (unit - motion)
        self.assertLessEqual(
            float(((scheduled - motion) - expected).abs().max()),
            1.0e-6,
        )
        self.assertGreater(float((unit - scheduled).abs().max()), 0.0)
        self.assertIn("unit_rho", D2AF_DIAGNOSTIC_VARIANTS)

    def test_timestep_contract_fails_closed(self):
        field = SparseCurrentStateRelationField(
            512,
            diffusion_reliability=True,
        ).eval()
        values, arguments = relation_arguments(2)
        motion = torch.zeros(2, 16, 512)
        invalid = (
            None,
            torch.zeros(2, 1, dtype=torch.long),
            torch.zeros(2, dtype=torch.float32),
            torch.tensor([-1, 0], dtype=torch.long),
            torch.tensor([0, 500], dtype=torch.long),
        )
        for timesteps in invalid:
            with self.subTest(timesteps=timesteps):
                with self.assertRaises(ValueError):
                    field(
                        motion,
                        values["current"],
                        **arguments,
                        timesteps=timesteps,
                    )

    def test_zero_alpha_parity_and_three_timestep_gradients(self):
        values, arguments = relation_arguments(3)
        noisy = values["current"].clone().requires_grad_(True)
        text = torch.randn(3, 768)
        bps = torch.randn(3, 1024, 3)
        goals = torch.randn(3, 9)
        progress = torch.randn(3, 3)
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        torch.manual_seed(42)
        d2ae = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AE,
        ).eval()
        torch.manual_seed(42)
        d2af = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AF,
        ).eval()
        with torch.no_grad():
            ae = d2ae(noisy.detach(), timesteps, text, bps, goals, progress, **arguments)
            af = d2af(noisy.detach(), timesteps, text, bps, goals, progress, **arguments)
        self.assertEqual(float((ae - af).abs().max()), 0.0)
        output = d2af(noisy, timesteps, text, bps, goals, progress, **arguments)
        output.square().mean().backward()
        alpha_grad = d2af.network.sparse_relation_field.alpha.grad
        self.assertIsNotNone(alpha_grad)
        self.assertTrue(torch.isfinite(alpha_grad))
        self.assertNotEqual(float(alpha_grad), 0.0)
        d2af.zero_grad(set_to_none=True)
        d2af.network.set_sparse_relation_gate_override(0.1)
        output = d2af(noisy.detach(), timesteps, text, bps, goals, progress, **arguments)
        output.square().mean().backward()
        field = d2af.network.sparse_relation_field
        for parameter in (
            field.point_encoder[0].weight,
            field.projection.weight,
            field.temporal_embeddings,
            d2af.network.transformer.layers[0].self_attn.in_proj_weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(torch.any(parameter.grad != 0))

    def test_capture_separates_raw_rho_and_attenuated_writeback(self):
        field = SparseCurrentStateRelationField(
            512,
            diffusion_reliability=True,
        ).eval()
        field.set_gate_override(0.1)
        field.set_capture(True)
        values, arguments = relation_arguments(3)
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        field(torch.zeros(3, 16, 512), values["current"], **arguments, timesteps=timesteps)
        snapshot = field.snapshot()
        self.assertEqual(snapshot["rho"].tolist(), [
            SQRT_ALPHA_BAR_SENTINELS[0],
            SQRT_ALPHA_BAR_SENTINELS[249],
            SQRT_ALPHA_BAR_SENTINELS[499],
        ])
        self.assertEqual(tuple(snapshot["raw_writeback_norm"].shape), (16,))
        self.assertEqual(tuple(snapshot["attenuated_writeback_norm"].shape), (16,))


class D2AFCheckpointAndConfigTests(unittest.TestCase):
    def _checkpoint(self, variant: str):
        torch.manual_seed(42)
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=variant,
        )
        value = {
            "schema_version": 2,
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "run_id": "test-only",
            "seed": 42,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": variant,
            },
            "architecture_variant": variant,
            "model": model.state_dict(),
        }
        if variant == HOI_ARCHITECTURE_D2AE:
            value["sparse_relation_contract"] = (
                model.network.sparse_relation_field.contract_metadata()
            )
        else:
            value["diffusion_reliability_contract"] = (
                model.network.sparse_relation_field.contract_metadata()
            )
        return value

    def test_independent_contract_and_cross_variant_rejection(self):
        contract = diffusion_reliability_contract_metadata()
        self.assertEqual(contract["architecture_variant"], HOI_ARCHITECTURE_D2AF)
        self.assertEqual(
            validate_diffusion_reliability_contract(contract),
            contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            ae_path = Path(directory) / "ae.pth"
            af_path = Path(directory) / "af.pth"
            torch.save(self._checkpoint(HOI_ARCHITECTURE_D2AE), ae_path)
            torch.save(self._checkpoint(HOI_ARCHITECTURE_D2AF), af_path)
            with self.assertRaisesRegex(ValueError, "architecture variant mismatch"):
                load_trained_hoi_prior(
                    str(ae_path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AF,
                )
            with self.assertRaisesRegex(ValueError, "architecture variant mismatch"):
                load_trained_hoi_prior(
                    str(af_path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
                )

    def test_resolved_config_is_single_factor_and_cpu_contract_accepts_no_bindings(self):
        base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
        ae = OmegaConf.merge(
            base,
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ae.yaml"),
        )
        af = OmegaConf.merge(
            base,
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2af.yaml"),
        )
        self.assertIsNone(af.run_id)
        ignored = {
            "mode", "subphase", "run_id", "d2ae_sparse_relation_field",
            "d2af_sqrt_alpha_bar_reliability", "hoi_architecture_variant",
            "d2ae_performance_benchmark_path",
            "d2ae_performance_benchmark_sha256",
            "d2af_clean_signal_eligibility_path",
            "d2af_clean_signal_eligibility_sha256",
            "d2af_performance_benchmark_path",
            "d2af_performance_benchmark_sha256",
        }
        ae_values = OmegaConf.to_container(ae, resolve=False)
        af_values = OmegaConf.to_container(af, resolve=False)
        for key in set(ae_values) | set(af_values):
            if key not in ignored:
                self.assertEqual(ae_values.get(key), af_values.get(key), key)
        actual_date = __import__("datetime").datetime.now().astimezone().strftime("%Y%m%d")
        af.run_id = f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{actual_date}"
        af.repo_root = str(ROOT)
        af.split_manifest = str(
            ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )
        _validate_fk_foot_temporal_routing_mode(af)
        contract = _validate_d2af_contract(
            af,
            4,
            require_eligibility_gate=False,
            require_performance_gate=False,
        )
        self.assertFalse(contract["eligibility_gate_required"])
        self.assertFalse(contract["performance_gate_required"])
        self.assertAlmostEqual(D2AF_MINIMUM_THROUGHPUT, 3179.689863044761)
        self.assertAlmostEqual(D2AF_MAXIMUM_ETA_HOURS, 5.367399778519349)

    def test_no_loss_weighting_or_alternate_relation_source(self):
        field_source = inspect.getsource(SparseCurrentStateRelationField.forward).lower()
        trainer_source = (ROOT / "code/train_hoi_prior.py").read_text(encoding="utf-8").lower()
        for forbidden in (
            "x_start", "future_gt", "contact_label", "scene",
            "snr_weight", "timestep_weight", "gamma", "per_anchor",
        ):
            self.assertNotIn(forbidden, field_source)
        self.assertNotIn("d2af_timestep_loss_weight", trainer_source)
        self.assertNotIn("d2af_snr_weight", trainer_source)


class D2AFSamplerTests(unittest.TestCase):
    def test_training_qsample_and_model_receive_exact_same_timestep_tensor(self):
        class DiffusionStub:
            def __init__(self):
                self.timesteps = None

            def q_sample(self, clean, timesteps, noise):
                del noise
                self.timesteps = timesteps
                return clean

        class ModelStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.timesteps = None

            def forward(self, noisy, timesteps, *args, **kwargs):
                del args, kwargs
                self.timesteps = timesteps
                return noisy

        cfg = OmegaConf.create({
            "d2ad_local_frame_interaction_adapter": False,
            "d2ae_sparse_relation_field": False,
            "d2af_sqrt_alpha_bar_reliability": True,
            "d2z_immutable_gt_near_ground_gating": False,
            "d2ab_predicted_support_no_slip": False,
            "fk_weight": 0.3569973401779424,
            "object_surface_weight": 0.4772322188400037,
            "velocity_weight": 0.1,
            "goal_weight": 1.0,
            "fk_foot_temporal_routing": True,
            "routed_foot_residual_multiplier": 1.0,
        })
        batch = {
            "x": torch.zeros(2, 16, 232),
            "text_embedding": torch.zeros(2, 768),
            "object_bps": torch.zeros(2, 1024, 3),
            "goals": torch.zeros(2, 9),
            "progress": torch.ones(2, 3),
            "rest_human_offsets": torch.zeros(2, 24, 3),
            "terminal_window": torch.zeros(2),
            "rest_object_points": torch.zeros(2, 100, 3),
            "world_to_local_rotation": torch.eye(3).repeat(2, 1, 1),
            "object_rotation_reference": torch.eye(3).repeat(2, 1, 1),
        }
        diffusion = DiffusionStub()
        model = ModelStub()
        fake_losses = {
            key: torch.tensor(0.0)
            for key in (
                "total", "reconstruction", "joint_position", "joint_rotation",
                "object_translation", "object_rotation", "contact", "fk",
                "object_surface", "velocity", "object_goal", "contact_accuracy",
            )
        }
        with mock.patch(
            "train_hoi_prior.hoi_training_losses",
            return_value=fake_losses,
        ):
            _forward_losses(
                model,
                diffusion,
                batch,
                torch.zeros(24, dtype=torch.long),
                torch.zeros(3),
                torch.ones(3),
                torch.zeros(3),
                torch.ones(3),
                cfg,
                generator=torch.Generator().manual_seed(42),
            )
        self.assertIs(diffusion.timesteps, model.timesteps)

    def test_sampler_forwards_exact_reverse_timestep_trace(self):
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.steps = []

            def forward(self, current, timesteps, *args, **kwargs):
                del args, kwargs
                self.steps.append(int(timesteps[0]))
                return torch.zeros_like(current)

        diffusion = GaussianDiffusion()
        recorder = Recorder()
        fixed_history = torch.zeros(1, 2, 232)
        diffusion.sample(
            recorder,
            fixed_history,
            torch.zeros(1, 768),
            torch.zeros(1, 1024, 3),
            torch.zeros(1, 9),
            torch.zeros(1, 3),
            generator=torch.Generator().manual_seed(42),
        )
        self.assertEqual(recorder.steps, list(reversed(range(500))))


if __name__ == "__main__":
    unittest.main()
