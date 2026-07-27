"""Authority CPU contract tests for the fixed D2-AC0 interaction adapter."""

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from priors.diffusion import HOIPriorSampler  # noqa: E402
from priors.interaction_adapter import (  # noqa: E402
    ADAPTER_PARAMETER_COUNT,
    ASSIGNMENT_SHA256,
    BPS_SHA256,
    CENTER_INDICES,
    CLUSTER_SIZES,
    LocalObjectInteractionAdapter,
    cluster_bps_features,
    load_bps_partition,
)
from priors.interaction_diagnostic import (  # noqa: E402
    PROTECTION_METRICS,
    VARIANTS,
    GT_CONTACT_FINITE_SEQUENCE_COUNT,
    GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256,
    attention_entropy,
    gt_contact_frame_distance,
    internal_mechanism_gate,
    native_gate,
    paired_finite_difference,
    paired_nonnegative_ratio_fixed,
    paired_ratio_fixed,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AC,
    HOIPrior,
    assert_parameter_independence,
    build_expert,
    load_trained_hoi_prior,
)
from train_hoi_prior import (  # noqa: E402
    _d2ac_gradient_audit,
    _locked_loss_weights,
    _optimization_contract,
    _validate_d2ac_contract,
)
from tools.run_hoi_d2ac_internal import (  # noqa: E402
    RUN_ID_RE as INTERNAL_RUN_ID_RE,
    penetration_object_key,
)
from tools.run_hoi_d2ac_native_evaluation import (  # noqa: E402
    INTERNAL_RUN_ID_RE as NATIVE_INTERNAL_RUN_ID_RE,
)


class D2ACPartitionTests(unittest.TestCase):
    def test_immutable_partition_and_canonical_hash(self):
        basis, assignment, means, sizes, metadata = load_bps_partition()
        self.assertEqual(metadata["bps_sha256"], BPS_SHA256)
        self.assertEqual(metadata["assignment_sha256"], ASSIGNMENT_SHA256)
        self.assertEqual(len(ASSIGNMENT_SHA256), 64)
        self.assertEqual(metadata["center_indices"], list(CENTER_INDICES))
        self.assertEqual(metadata["cluster_sizes"], list(CLUSTER_SIZES))
        self.assertEqual(tuple(basis.shape), (1024, 3))
        self.assertEqual(tuple(assignment.shape), (1024,))
        self.assertEqual(tuple(means.shape), (16, 3))
        self.assertEqual(tuple(sizes.shape), (16,))
        self.assertTrue(torch.isfinite(basis).all())
        self.assertTrue(torch.isfinite(means).all())
        payload = metadata["canonical_assignment_payload"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), ASSIGNMENT_SHA256)

    def test_fixed_local_features_are_10d_finite_and_permutation_is_delta_only(self):
        basis, assignment, means, sizes, _ = load_bps_partition()
        bps = torch.randn(3, 1024, 3, generator=torch.Generator().manual_seed(42))
        normal = cluster_bps_features(bps, basis, assignment, means, sizes)
        permuted = cluster_bps_features(
            bps, basis, assignment, means, sizes,
            permute_delta_statistics=True,
        )
        self.assertEqual(tuple(normal.shape), (3, 16, 10))
        self.assertEqual(normal.dtype, bps.dtype)
        self.assertTrue(torch.isfinite(normal).all())
        torch.testing.assert_close(normal[..., :3], permuted[..., :3])
        self.assertGreater(float((normal[..., 3:] - permuted[..., 3:]).abs().max()), 0.0)

    def test_zero_constant_and_extreme_bps_remain_finite(self):
        basis, assignment, means, sizes, _ = load_bps_partition()
        for value in (torch.zeros(2, 1024, 3), torch.full((2, 1024, 3), 3.0),
                      torch.full((2, 1024, 3), 1.0e4)):
            result = cluster_bps_features(value, basis, assignment, means, sizes)
            self.assertTrue(torch.isfinite(result).all())


class D2ACModelTests(unittest.TestCase):
    @staticmethod
    def inputs(batch=2):
        generator = torch.Generator().manual_seed(123)
        return (
            torch.randn(batch, 16, 232, generator=generator),
            torch.arange(batch, dtype=torch.long) * 249 % 500,
            torch.randn(batch, 768, generator=generator),
            torch.randn(batch, 1024, 3, generator=generator),
            torch.randn(batch, 9, generator=generator),
            torch.randn(batch, 3, generator=generator),
        )

    def test_exact_parameter_count_and_api(self):
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AC,
        )
        self.assertIsInstance(model, HOIPrior)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            30_023_145,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.network.interaction_adapter.parameters()),
            ADAPTER_PARAMETER_COUNT,
        )
        self.assertEqual(float(model.network.interaction_adapter.alpha), 0.0)
        self.assertEqual(
            list(inspect.signature(HOIPrior.forward).parameters),
            ["self", "noisy", "timesteps", "text_embedding", "object_bps", "goals", "progress"],
        )
        output = model(*self.inputs(batch=2))
        self.assertEqual(tuple(output.shape), (2, 16, 232))

    def test_alpha_zero_parity_with_shared_base_trunk(self):
        torch.manual_seed(42)
        base = build_expert("hoi", dim_model=512, num_heads=16, num_layers=8)
        torch.manual_seed(99)
        adapter = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AC,
        )
        missing, unexpected = adapter.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertFalse(unexpected)
        base.eval()
        adapter.eval()
        values = self.inputs(batch=1)
        with torch.no_grad():
            expected = base(*values)
            actual = adapter(*values)
        self.assertLessEqual(float((expected - actual).abs().max()), 1.0e-6)

    def test_initial_alpha_and_activated_adapter_gradients(self):
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AC,
        )
        values = self.inputs(batch=2)
        prediction = model(*values)
        (prediction - values[0]).square().mean().backward()
        initial = _d2ac_gradient_audit(model, require_adapter_paths=False)
        self.assertTrue(initial["alpha"]["finite"])
        self.assertTrue(initial["alpha"]["nonzero"])
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.network.interaction_adapter.alpha.copy_(
                torch.atanh(torch.tensor(0.1))
            )
        prediction = model(*values)
        (prediction - values[0]).square().mean().backward()
        activated = _d2ac_gradient_audit(model, require_adapter_paths=True)
        self.assertAlmostEqual(activated["gate_value"], 0.1, places=5)
        self.assertTrue(all(
            item["finite"] and item["nonzero"]
            for item in activated["adapter_groups"].values()
        ))

    def test_local_correspondence_permutation_has_causal_effect(self):
        model = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AC,
        ).eval()
        values = self.inputs(batch=1)
        with torch.no_grad():
            model.network.interaction_adapter.set_diagnostic_variant("full")
            model.network.interaction_adapter.set_gate_override(0.1)
            full = model(*values)
            model.network.interaction_adapter.set_diagnostic_variant(
                "local_correspondence_permuted"
            )
            model.network.interaction_adapter.set_gate_override(0.1)
            permuted = model(*values)
        self.assertGreater(float((full - permuted).abs().max()), 1.0e-8)

    def test_role_query_separation_and_dtype_batch_propagation(self):
        adapter = LocalObjectInteractionAdapter().double()
        values = self.inputs(batch=3)
        motion = torch.randn(
            3, 16, 512, dtype=torch.float64,
            generator=torch.Generator().manual_seed(7),
        )
        bps = values[3].double()
        output = adapter(motion, bps)
        self.assertEqual(tuple(output.shape), (3, 16, 512))
        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(
            float((adapter.part_embedding[0] - adapter.part_embedding[1]).abs().max()),
            0.0,
        )


class D2ACGovernanceTests(unittest.TestCase):
    def merged_config(self):
        base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
        d2ac = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ac.yaml")
        config = OmegaConf.merge(base, d2ac)
        config.repo_root = str(ROOT)
        config.split_manifest = str(
            ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )
        return config

    def test_exact_training_contract_and_optimizer(self):
        config = self.merged_config()
        _validate_d2ac_contract(config, 4)
        self.assertEqual(
            _locked_loss_weights(config),
            {
                "fk": 0.3569973401779424,
                "object_surface": 0.4772322188400037,
                "velocity": 0.1,
                "terminal_goal": 1.0,
            },
        )
        self.assertEqual(_optimization_contract(config)["optimizer"], "Adam")
        self.assertEqual(_optimization_contract(config)["scheduler"], "none")

    def test_contract_rejects_checkpoint_or_pause_mutation(self):
        for field, value in (
            ("init_checkpoint", "/tmp/released.pth"),
            ("weight_init_checkpoint", "/tmp/d2x.pth"),
            ("pause_after_windows", 3_072_000),
            ("fk_foot_temporal_routing", False),
        ):
            config = self.merged_config()
            setattr(config, field, value)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "D2-AC"):
                    _validate_d2ac_contract(config, 4)

    def test_checkpoint_variant_and_provenance_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-provenance.pth"
            torch.save({
                "checkpoint_type": "hoi_prior_phase1b",
                "expert": "hoi",
                "initialization": "random",
                "model_config": {
                    "dim_model": 512, "num_heads": 16, "num_layers": 8,
                    "architecture_variant": HOI_ARCHITECTURE_D2AC,
                },
            }, path)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_trained_hoi_prior(str(path), torch.device("cpu"))

    def test_hsi_independence_and_mixer_clean_output_contract(self):
        hoi = build_expert(
            "hoi", dim_model=32, num_heads=4, num_layers=1,
            architecture_variant=HOI_ARCHITECTURE_BASE,
        )
        hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        assert_parameter_independence(hoi, hsi)
        sampler = HOIPriorSampler(device="cpu", auto_regre_num=2, timesteps=500)
        del sampler
        values = D2ACModelTests.inputs(batch=1)
        output = hoi(*values)
        self.assertEqual(tuple(output.shape), (1, 16, 232))

    def test_static_model_path_has_no_evaluator_or_future_bps_input(self):
        source = (
            (ROOT / "code/priors/interaction_adapter.py").read_text(encoding="utf-8")
            + (ROOT / "code/priors/models.py").read_text(encoding="utf-8")
        )
        for forbidden in (
            "eval_metrics", "near_ground", "contact_guidance",
            "stored_per_frame_bps", "future_gt", "contact_label",
        ):
            self.assertNotIn(forbidden, source)

    def test_fixed_internal_and_native_runners_are_registered_and_static(self):
        internal = (
            ROOT / "tools/run_hoi_d2ac_internal.py"
        ).read_text(encoding="utf-8")
        native = (
            ROOT / "tools/run_hoi_d2ac_native_evaluation.py"
        ).read_text(encoding="utf-8")
        smoke = (
            ROOT / "tools/smoke_hoi_d2ac.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            VARIANTS,
            ("full", "gate_ablated", "local_correspondence_permuted"),
        )
        self.assertIn("set_interaction_diagnostic_variant", internal)
        self.assertIn("paired_noise_identity", internal)
        self.assertIn("paired_nonnegative_ratio_fixed", internal)
        self.assertIn("official_test_used", native)
        self.assertNotIn("paired_nonnegative_ratio_fixed", native)
        self.assertIn(
            "69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a",
            native,
        )
        self.assertNotIn("checkpoint_weight_variant=ema", native)
        self.assertIn(
            "registered_formal_cross_attention_score_elements_estimate",
            smoke,
        )
        self.assertIn(
            'RUN_ID = "p1-hoi-d2ac-gpu-smoke-r1-s42-20260726"',
            smoke,
        )
        self.assertIn("if args.run_id != RUN_ID:", smoke)
        self.assertNotIn("RUN_ID_RE", smoke)
        diagnostic = (
            ROOT / "tools/diagnose_hoi_d2ac.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'RUN_ID = "p1-hoi-d2ac-cpu-contract-r1-s42-20260726"',
            diagnostic,
        )
        self.assertIn("if run_id != RUN_ID:", diagnostic)


class D2ACDiagnosticMetricTests(unittest.TestCase):
    def test_official_sdf_keying_and_internal_retry_identity(self):
        self.assertEqual(
            penetration_object_key(Path("floorlamp.ply.npy")),
            "floorlamp",
        )
        retry = (
            "p1-hoi-d2ac-interaction-adapter-internal-r1-s42-20260727"
        )
        self.assertIsNotNone(INTERNAL_RUN_ID_RE.fullmatch(retry))
        self.assertIsNotNone(NATIVE_INTERNAL_RUN_ID_RE.fullmatch(retry))
        retry2 = (
            "p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727"
        )
        self.assertIsNotNone(INTERNAL_RUN_ID_RE.fullmatch(retry2))
        self.assertIsNotNone(NATIVE_INTERNAL_RUN_ID_RE.fullmatch(retry2))

    def test_positive_penetration_ratio_preserves_locked_ratio_fields(self):
        numerator = [0.1, 0.2, 0.3]
        denominator = [0.2, 0.4, 0.6]
        locked = paired_ratio_fixed(numerator, denominator)
        value = paired_nonnegative_ratio_fixed(numerator, denominator)
        self.assertTrue(value["ratio_defined"])
        for key, expected in locked.items():
            self.assertEqual(value[key], expected)
        self.assertEqual(
            value["paired_difference"]["bootstrap_replicates"],
            10_000,
        )
        self.assertEqual(value["paired_difference"]["bootstrap_seed"], 42)

    def test_zero_penetration_denominator_is_explicitly_undefined(self):
        value = paired_nonnegative_ratio_fixed(
            [0.0, 1.0e-6, 0.0],
            [0.0, 0.0, 0.0],
        )
        self.assertFalse(value["ratio_defined"])
        self.assertEqual(value["undefined_reason"], "zero_denominator_mean")
        self.assertEqual(value["denominator_mean"], 0.0)
        self.assertIsNone(value["mean_ratio"])
        self.assertIsNone(value["bootstrap_95_ci"])
        difference = value["paired_difference"]
        self.assertEqual(difference["first_mean"], value["numerator_mean"])
        self.assertEqual(difference["second_mean"], 0.0)
        self.assertGreater(
            difference["paired_mean_first_minus_second"],
            0.0,
        )
        self.assertEqual(difference["bootstrap_replicates"], 10_000)
        self.assertEqual(difference["bootstrap_seed"], 42)

    def test_nonnegative_penetration_comparison_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            paired_nonnegative_ratio_fixed([0.0, -1.0], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_nonnegative_ratio_fixed([0.0, float("nan")], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "equal non-empty"):
            paired_nonnegative_ratio_fixed([0.0], [0.0, 0.0])

    def test_attention_entropy_is_role_preserving_and_normalized(self):
        weights = torch.full((2, 16, 3, 4, 16), 1.0 / 16.0)
        value = attention_entropy(weights)
        self.assertEqual(tuple(value["nats"].shape), (2, 16, 3, 4))
        torch.testing.assert_close(
            value["normalized"], torch.ones_like(value["normalized"]),
        )
        concentrated = torch.zeros_like(weights)
        concentrated[..., 0] = 1.0
        concentrated_value = attention_entropy(concentrated)
        torch.testing.assert_close(
            concentrated_value["normalized"],
            torch.zeros_like(concentrated_value["normalized"]),
        )

    def test_gt_contact_distance_uses_fixed_target_mask_without_imputation(self):
        target = torch.tensor([
            [0.01, 0.20],
            [0.20, 0.02],
            [0.20, 0.20],
        ]).numpy()
        predicted = torch.tensor([
            [0.03, 0.30],
            [0.40, 0.04],
            [0.10, 0.10],
        ]).numpy()
        value = gt_contact_frame_distance(predicted, target)
        self.assertEqual(value["union"]["frames"], 2)
        self.assertAlmostEqual(value["union"]["mean_cm"], 3.5, places=6)
        no_contact = gt_contact_frame_distance(
            predicted, torch.full_like(torch.from_numpy(target), 0.20).numpy(),
        )
        self.assertIsNone(no_contact["union"]["mean_cm"])
        self.assertFalse(no_contact["union"]["finite"])

    def test_paired_finite_difference_reports_the_unimputed_sequence_mask(self):
        value = paired_finite_difference(
            [2.0, None, 4.0],
            [1.0, None, 1.0],
            ["a", "b", "c"],
        )
        self.assertEqual(value["finite_sequence_count"], 2)
        self.assertEqual(value["finite_sequence_names"], ["a", "c"])
        self.assertGreater(
            value["paired_mean_first_minus_second"], 0.0,
        )
        self.assertEqual(GT_CONTACT_FINITE_SEQUENCE_COUNT, 57)
        self.assertEqual(
            GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256,
            "2fa79d30ab6dd6a915098344c4aa7267cb6c3323c6d2a762b4b704f8757cebaa",
        )

    @staticmethod
    def _positive_statistic():
        return {"bootstrap_95_ci": [0.01, 0.02]}

    def test_internal_gate_distinguishes_unused_and_locality_negative(self):
        positive = self._positive_statistic()
        comparisons = {
            "full_vs_gate_ablated": {
                "full_minus_other_direct_union_5cm_f1": positive,
                "other_minus_full_gt_contact_distance_cm": positive,
            },
            "full_vs_local_correspondence_permuted": {
                "full_minus_other_direct_union_5cm_f1": positive,
                "other_minus_full_gt_contact_distance_cm": positive,
            },
        }
        passed = internal_mechanism_gate({"finite": True}, comparisons)
        self.assertTrue(passed["mechanism_passed"])
        unused = json.loads(json.dumps(comparisons))
        unused["full_vs_gate_ablated"][
            "full_minus_other_direct_union_5cm_f1"
        ]["bootstrap_95_ci"][0] = 0.0
        self.assertEqual(
            internal_mechanism_gate({"finite": True}, unused)["classification"],
            "interaction-adapter-unused-optimization-negative-stop",
        )
        locality = json.loads(json.dumps(comparisons))
        locality["full_vs_local_correspondence_permuted"][
            "other_minus_full_gt_contact_distance_cm"
        ]["bootstrap_95_ci"][0] = -0.01
        self.assertEqual(
            internal_mechanism_gate(
                {"finite": True}, locality,
            )["classification"],
            "interaction-adapter-locality-negative-stop",
        )

    def test_native_gate_requires_transfer_protection_and_released_effectiveness(self):
        comparison = {
            "penetration_mask_contract": {"passed": True},
            "target_minus_control_contact_f1": self._positive_statistic(),
            "target_minus_control_contact_recall": self._positive_statistic(),
            "target_minus_control_contact_precision": {
                "bootstrap_95_ci": [-0.01, 0.01],
            },
            "contact_f1_released_gap_closure": 0.25,
            "target_over_control_protection": {
                metric: {"bootstrap_95_ci": [0.9, 1.0]}
                for metric in PROTECTION_METRICS
            },
        }
        baseline_ratios = {
            "end_obj_trans_err": 1.0,
            "xy_points_err": 1.0,
            "foot_sliding": 1.0,
            "human_pen_loss_infbagel": 1.0,
            "mpjpe": 1.0,
            "trans_dist": 1.0,
            "obj_trans_dist": 1.0,
            "obj_rot_dist": 1.0,
            "contact_precision": 1.0,
            "contact_recall": 1.0,
            "contact_f1": 1.0,
        }
        internal = {
            "contract_passed": True,
            "adapter_used": True,
            "locality_passed": True,
            "mechanism_passed": True,
        }
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics={"contact_f1": 1.0},
            baseline_ratios=baseline_ratios,
        )
        self.assertEqual(
            decision["classification"],
            "interaction-adapter-positive-candidate-stop",
        )
        self.assertTrue(decision["selectable_autonomous_diffusion_candidate"])
        transfer_negative = json.loads(json.dumps(comparison))
        transfer_negative["target_minus_control_contact_recall"][
            "bootstrap_95_ci"
        ][0] = 0.0
        self.assertEqual(
            native_gate(
                contract_passed=True,
                internal=internal,
                comparison=transfer_negative,
                target_metrics={"contact_f1": 1.0},
                baseline_ratios=baseline_ratios,
            )["classification"],
            "interaction-adapter-transfer-negative-stop",
        )


if __name__ == "__main__":
    unittest.main()
