"""HOIPrior objective and registered-gradient-probe contracts."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hoi import diagnostics, losses as loss_module
from priors.hoi.diffusion import GaussianDiffusion
from priors.hoi.losses import hoi_training_losses


def _synthetic_inputs(batch_size=2):
    torch.manual_seed(42)
    prediction = (torch.randn(batch_size, 16, 232) * 0.15).requires_grad_()
    target = torch.randn(batch_size, 16, 232) * 0.15
    # Engage both semantic hand-contact channels on active frames.
    target[:, 2:10, 228] = 1.0
    target[:, 6:, 229] = 1.0
    rest_offsets = torch.randn(batch_size, 24, 3) * 0.08
    rest_offsets[:, 0] = 0.0
    parents = torch.zeros(24, dtype=torch.long)
    parents[0] = -1
    identity = torch.eye(3).expand(batch_size, 3, 3).clone()
    values = {
        "prediction": prediction,
        "target": target,
        "goals": torch.randn(batch_size, 9) * 0.1,
        "rest_human_offsets": rest_offsets,
        "parents_24": parents,
        "position_minimum": torch.tensor([-1.5, -1.0, -1.5]),
        "position_maximum": torch.tensor([1.5, 2.0, 1.5]),
        "object_minimum": torch.tensor([-1.0, -1.0, -1.0]),
        "object_maximum": torch.tensor([1.0, 1.0, 1.0]),
        "terminal_window": torch.tensor([1.0, 0.0])[:batch_size],
        "rest_object_points": torch.randn(batch_size, 12, 3) * 0.12,
        "world_to_local_rotation": identity,
        "object_rotation_reference": identity.clone(),
    }
    return values


def _call(values, detach_root, weight=3.0):
    return hoi_training_losses(
        values["prediction"],
        values["target"],
        values["goals"],
        values["rest_human_offsets"],
        values["parents_24"],
        values["position_minimum"],
        values["position_maximum"],
        values["object_minimum"],
        values["object_maximum"],
        values["terminal_window"],
        values["rest_object_points"],
        values["world_to_local_rotation"],
        values["object_rotation_reference"],
        hand_object_contact_weight=weight,
        hand_object_contact_detach_root=detach_root,
    )


class RootDetachLossTests(unittest.TestCase):
    def test_forward_values_and_rotation_gradients_are_bit_identical(self):
        values = _synthetic_inputs()
        attached = _call(values, False)
        detached = _call(values, True)
        self.assertTrue(torch.equal(attached["hand_object_contact_geometry"],
                                    detached["hand_object_contact_geometry"]))
        self.assertTrue(torch.equal(attached["total"], detached["total"]))
        attached_gradient, = torch.autograd.grad(
            attached["hand_object_contact_geometry"], values["prediction"],
            retain_graph=True,
        )
        detached_gradient, = torch.autograd.grad(
            detached["hand_object_contact_geometry"], values["prediction"],
            retain_graph=True,
        )
        self.assertGreater(float(attached_gradient[..., 0:3].norm()), 0.0)
        self.assertEqual(int(torch.count_nonzero(detached_gradient[..., 0:3])), 0)
        self.assertTrue(torch.equal(attached_gradient[..., 84:216],
                                    detached_gradient[..., 84:216]))
        self.assertGreater(float(attached_gradient[..., 84:216].norm()), 0.0)

    def test_default_path_has_one_fk_pass_and_detached_path_has_two(self):
        values = _synthetic_inputs()
        original = loss_module._fk_positions
        with mock.patch.object(loss_module, "_fk_positions", wraps=original) as wrapped:
            _call(values, False)
            self.assertEqual(wrapped.call_count, 1)
        with mock.patch.object(loss_module, "_fk_positions", wraps=original) as wrapped:
            _call(values, True)
            self.assertEqual(wrapped.call_count, 2)

    def test_fk_value_and_root_gradient_remain_attached(self):
        values = _synthetic_inputs()
        attached = _call(values, False)
        detached = _call(values, True)
        self.assertTrue(torch.equal(attached["fk"], detached["fk"]))
        attached_gradient, = torch.autograd.grad(
            attached["fk"], values["prediction"], retain_graph=True,
        )
        detached_gradient, = torch.autograd.grad(
            detached["fk"], values["prediction"], retain_graph=True,
        )
        self.assertGreater(float(attached_gradient[..., 0:3].norm()), 0.0)
        self.assertGreater(float(detached_gradient[..., 0:3].norm()), 0.0)
        self.assertTrue(torch.equal(attached_gradient, detached_gradient))

    def test_inert_root_detach_is_rejected_but_default_zero_weight_is_valid(self):
        values = _synthetic_inputs()
        with self.assertRaisesRegex(ValueError, "requires non-zero"):
            _call(values, True, weight=0.0)
        result = _call(values, False, weight=0.0)
        self.assertNotIn("hand_object_contact_geometry", result)


class _SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, noisy, timesteps, text_embedding, object_bps, goals, progress):
        del timesteps, text_embedding, object_bps, goals, progress
        return noisy * 0.05 + self.bias


def _probe_fixture():
    values = _synthetic_inputs()
    batch = {
        "x": values["target"],
        "text_embedding": torch.randn(2, 4),
        "object_bps": torch.randn(2, 8),
        "goals": values["goals"],
        "progress": torch.tensor([[0.0, 1.0, 10.0], [1.0, 2.0, 10.0]]),
        "rest_human_offsets": values["rest_human_offsets"],
        "terminal_window": values["terminal_window"],
        "rest_object_points": values["rest_object_points"],
        "world_to_local_rotation": values["world_to_local_rotation"],
        "object_rotation_reference": values["object_rotation_reference"],
    }
    cfg = OmegaConf.create({
        "fk_weight": 0.3569973401779424,
        "object_surface_weight": 0.4772322188400037,
        "velocity_weight": 0.1,
        "goal_weight": 1.0,
        "hand_object_contact_weight": 3.0,
        "hand_object_contact_hinge": 0.0,
        "hand_object_contact_detach_object": False,
        "hand_object_contact_detach_root": False,
        "fk_foot_temporal_routing": True,
        "routed_foot_residual_multiplier": 1.0,
        "d2ai_full_budget": True,
    })
    return values, batch, cfg


class RootGradientShareProbeTests(unittest.TestCase):
    def _run(self, directory):
        values, batch, cfg = _probe_fixture()
        checkpoint = Path(directory) / "sealed_w3.pth"
        checkpoint.write_bytes(b"synthetic-checkpoint")
        output = Path(directory) / "probe.json"
        result = diagnostics.root_gradient_share_probe(
            _SyntheticModel(),
            GaussianDiffusion(500),
            [batch],
            values["parents_24"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
            cfg,
            checkpoint_path=checkpoint,
            output_path=output,
            window_count=2,
        )
        return result, output

    def test_probe_reports_training_gradients_and_passes_exact_self_check(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run(directory)
            self.assertEqual(result["window_count"], 2)
            self.assertGreater(result["geometry_gradient_l2"]["root_translation"], 0.0)
            self.assertGreater(result["geometry_gradient_l2"]["rotations"], 0.0)
            self.assertGreater(result["root_gradient_share"], 0.0)
            self.assertTrue(result["self_check"]["detached_root_gradient_exactly_zero"])
            self.assertTrue(result["self_check"]["detached_rotation_gradient_bitwise_equal"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_probe_self_check_asserts_on_any_rotation_gradient_change(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(diagnostics.torch, "equal", return_value=False):
                with self.assertRaisesRegex(AssertionError, "rotation gradient"):
                    self._run(directory)


class GeometryTermForwardScaleProbeTests(unittest.TestCase):
    def _run(self, directory, name):
        values, batch, cfg = _probe_fixture()
        output = Path(directory) / name
        result = diagnostics.geometry_term_forward_scale_probe(
            [batch],
            values["parents_24"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
            cfg,
            output_path=output,
            window_count=2,
        )
        return result, output

    def test_reports_finite_floor_coverage_and_sensitivity_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run(directory, "forward.json")
            self.assertTrue(math.isfinite(result["floor"]))
            self.assertGreaterEqual(result["coverage"]["engaged_frame_fraction"], 0.0)
            self.assertLessEqual(result["coverage"]["engaged_frame_fraction"], 1.0)
            self.assertGreaterEqual(result["coverage"]["engaged_window_fraction"], 0.0)
            self.assertLessEqual(result["coverage"]["engaged_window_fraction"], 1.0)
            self.assertEqual(set(result["sensitivity"]), {"0.02", "0.05", "0.10"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_is_bit_identical_for_the_same_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            first, first_output = self._run(directory, "first.json")
            second, second_output = self._run(directory, "second.json")
            self.assertEqual(first, second)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())


class GeometryTermPalmDecompositionProbeTests(unittest.TestCase):
    def _run(self, directory, name="palm-decomposition.json", batches=None):
        values, batch, cfg = _probe_fixture()
        output = Path(directory) / name
        result = diagnostics.geometry_term_palm_decomposition_probe(
            batches if batches is not None else [batch],
            values["parents_24"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
            cfg,
            output_path=output,
            window_count=2,
        )
        return result, output

    def _run_label_fixture(self, directory, contacting_distance, free_distance):
        values, batch, cfg = _probe_fixture()
        frames = diagnostics.REPRESENTATION.history_frames + 1
        fk = torch.zeros(2, frames, 24, 3)
        surface = torch.zeros(2, frames, 1, 3)
        contact = torch.zeros(2, frames, 4)
        fk[:, -1, 22, 0] = contacting_distance
        fk[:, -1, 23, 0] = free_distance
        contact[:, -1, 0] = 1.0

        def adversarial_geometry_losses(*_args, **_kwargs):
            scalar = loss_module.masked_hand_object_distance_loss(
                fk, surface, contact
            )
            return {"hand_object_contact_geometry": scalar}

        with mock.patch.object(
            diagnostics, "_geometry_losses", side_effect=adversarial_geometry_losses
        ):
            return diagnostics.geometry_term_palm_decomposition_probe(
                [batch],
                values["parents_24"],
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
                cfg,
                output_path=Path(directory) / "label-fixture.json",
                window_count=2,
            )

    @staticmethod
    def _split_batch(batch):
        batch_size = int(batch["x"].shape[0])
        return [
            {
                key: (
                    value[index:index + 1]
                    if torch.is_tensor(value)
                    and value.ndim
                    and value.shape[0] == batch_size
                    else value
                )
                for key, value in batch.items()
            }
            for index in range(batch_size)
        ]

    def assert_moments_almost_equal(self, first, second):
        self.assertEqual(first["count"], second["count"])
        for name in ("mean_distance_m", "mean_squared_m2", "rms_m"):
            if first[name] is None or second[name] is None:
                self.assertIsNone(first[name])
                self.assertIsNone(second[name])
            else:
                self.assertAlmostEqual(first[name], second[name], delta=1e-6)

    def test_reaggregates_real_loss_and_partitions_active_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run(directory)
            counts = result["frame_counts"]
            self.assertEqual(
                counts["left_only"]
                + counts["right_only"]
                + counts["both"]
                + counts["neither"],
                counts["active"],
            )
            self.assertEqual(
                counts["left_only"] + counts["right_only"] + counts["both"],
                counts["engaged"],
            )
            self.assertLessEqual(
                abs(result["reaggregation"]["relative_error"]),
                result["reaggregation"]["tolerance"],
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_contacting_assignment_follows_label_when_contacting_palm_is_farther(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run_label_fixture(directory, 3.0, 1.0)
        self.assertEqual(result["frame_counts"]["left_only"], 2)
        self.assertEqual(result["contacting"]["mean_distance_m"], 3.0)
        self.assertEqual(result["free"]["mean_distance_m"], 1.0)
        self.assertGreater(
            result["contacting"]["mean_distance_m"],
            result["free"]["mean_distance_m"],
        )
        nearest = result["nearest_palm_vs_label"]
        self.assertEqual(nearest["total"]["disagree"]["count"], 2)
        self.assertEqual(nearest["total"]["agree"]["count"], 0)
        self.assertEqual(nearest["total"]["tie"]["count"], 0)
        self.assertEqual(nearest["left_only"]["disagree"]["count"], 2)
        self.assertEqual(
            nearest["total"]["disagree_subset"]["contacting"][
                "mean_distance_m"
            ],
            3.0,
        )
        self.assertEqual(
            nearest["total"]["disagree_subset"]["free"]["mean_distance_m"],
            1.0,
        )

    def test_tie_is_not_folded_into_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run_label_fixture(directory, 2.0, 2.0)
        total = result["nearest_palm_vs_label"]["total"]
        self.assertEqual(total["tie"]["count"], 2)
        self.assertEqual(total["agree"]["count"], 0)
        self.assertEqual(total["disagree"]["count"], 0)

    def test_labelled_contacting_palm_strictly_nearer_is_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run_label_fixture(directory, 1.0, 3.0)
        total = result["nearest_palm_vs_label"]["total"]
        self.assertEqual(total["agree"]["count"], 2)
        self.assertEqual(total["disagree"]["count"], 0)
        self.assertEqual(total["tie"]["count"], 0)

    def test_streaming_probe_is_bit_identical_for_the_same_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            first, first_output = self._run(directory, "first.json")
            second, second_output = self._run(directory, "second.json")
            self.assertEqual(first, second)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())

    def test_streaming_moments_are_equivalent_across_batching(self):
        values, batch, _cfg = _probe_fixture()
        del values, _cfg
        with tempfile.TemporaryDirectory() as directory:
            whole, _ = self._run(directory, "whole.json", [batch])
            split, _ = self._run(
                directory, "split.json", self._split_batch(batch)
            )
        self.assertEqual(whole["frame_counts"], split["frame_counts"])
        for role in ("contacting", "free"):
            self.assert_moments_almost_equal(whole[role], split[role])
        for category in ("left_only", "right_only", "both"):
            for role in ("contacting", "free"):
                self.assert_moments_almost_equal(
                    whole["by_category"][category][role],
                    split["by_category"][category][role],
                )
        for category in ("left_only", "right_only", "total"):
            whole_nearest = whole["nearest_palm_vs_label"][category]
            split_nearest = split["nearest_palm_vs_label"][category]
            for name in ("single_contact_frames", "proportion_denominator"):
                self.assertEqual(whole_nearest[name], split_nearest[name])
            for name in ("agree", "disagree", "tie"):
                self.assertEqual(
                    whole_nearest[name]["count"], split_nearest[name]["count"]
                )
            for role in ("contacting", "free"):
                self.assert_moments_almost_equal(
                    whole_nearest["disagree_subset"][role],
                    split_nearest["disagree_subset"][role],
                )

    def test_nearest_palm_vs_label_categories_partition_single_contact_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._run(directory)
        for category in ("left_only", "right_only", "total"):
            nearest = result["nearest_palm_vs_label"][category]
            self.assertEqual(
                nearest["agree"]["count"]
                + nearest["disagree"]["count"]
                + nearest["tie"]["count"],
                nearest["single_contact_frames"],
            )
        self.assertTrue(
            result["self_check"][
                "nearest_palm_vs_label_partitions_single_contact_frames"
            ]
        )

    def test_capture_wrapper_hard_raises_on_zero_or_two_invocations(self):
        with self.assertRaisesRegex(AssertionError, "exactly one.*observed 0"):
            with diagnostics._captured_geometry_loss_call():
                pass

        frames = diagnostics.REPRESENTATION.history_frames + 1
        fk = torch.zeros(1, frames, 24, 3)
        surface = torch.zeros(1, frames, 1, 3)
        contact = torch.zeros(1, frames, 4)
        with self.assertRaisesRegex(AssertionError, "exactly one.*observed 2"):
            with diagnostics._captured_geometry_loss_call():
                loss_module.masked_hand_object_distance_loss(fk, surface, contact)
                loss_module.masked_hand_object_distance_loss(fk, surface, contact)


class GeometryWeightDerivationProbeTests(unittest.TestCase):
    def _run(self, directory, weight, output_name="weight.json"):
        values, batch, cfg = _probe_fixture()
        cfg.hand_object_contact_weight = weight
        checkpoint = Path(directory) / "checkpoint.pth"
        if not checkpoint.exists():
            checkpoint.write_bytes(b"synthetic-checkpoint")
        output = Path(directory) / output_name
        result = diagnostics.geometry_weight_derivation_probe(
            _SyntheticModel(),
            GaussianDiffusion(500),
            [batch],
            values["parents_24"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
            cfg,
            checkpoint_path=checkpoint,
            output_path=output,
            window_count=2,
            timesteps=(250, 499),
        )
        return result

    def test_zero_and_weight_three_report_all_documented_keys(self):
        expected = {
            "probe", "seed", "window_count", "batch_count", "checkpoint_path",
            "checkpoint_sha256", "git_commit",
            "configured_hand_object_contact_weight", "geometry_from_separate_call",
            "timesteps", "timestep_seam", "gradient_l2", "human_side_l2",
            "object_side_l2", "target_channel", "root_channel_geometry_l2",
            "p0_calibration", "self_check",
        }
        with tempfile.TemporaryDirectory() as directory:
            zero = self._run(directory, 0.0, "zero.json")
            weighted = self._run(directory, 3.0, "three.json")
            self.assertEqual(set(zero), expected)
            self.assertEqual(set(weighted), expected)
            self.assertTrue(zero["geometry_from_separate_call"])
            self.assertFalse(weighted["geometry_from_separate_call"])
            self.assertEqual(zero["configured_hand_object_contact_weight"], 0.0)
            self.assertEqual(weighted["configured_hand_object_contact_weight"], 3.0)
            for result in (zero, weighted):
                self.assertEqual(result["timestep_seam"], "pinned_real_forward_losses")
                self.assertTrue(result["self_check"]["all_values_finite"])
                self.assertTrue(
                    result["self_check"][
                        "geometry_gradient_on_joint_positions_exactly_zero"
                    ]
                )
                self.assertEqual(set(result["gradient_l2"]), {"250", "499"})

    def test_pinned_timestep_bitwise_reproduces_plain_forward_losses_batch_two(self):
        import train_hoi_prior

        values, batch, cfg = _probe_fixture()
        model = _SyntheticModel()
        diffusion = GaussianDiffusion(500)
        device = batch["x"].device
        # This seed makes both independently drawn batch timesteps equal, so a
        # scalar pin can reproduce the unpatched two-window trainer call.
        seed = 1597

        reference_generator = torch.Generator(device=device)
        reference_generator.manual_seed(seed)
        real_randint = train_hoi_prior.torch.randint
        recorded_timesteps = []

        def recording_randint(*args, **kwargs):
            drawn = real_randint(*args, **kwargs)
            recorded_timesteps.append(drawn.clone())
            return drawn

        with mock.patch.object(
            train_hoi_prior.torch, "randint", side_effect=recording_randint
        ):
            reference = train_hoi_prior._forward_losses(
                model,
                diffusion,
                batch,
                values["parents_24"],
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
                cfg,
                generator=reference_generator,
            )
        self.assertEqual(len(recorded_timesteps), 1)
        self.assertEqual(recorded_timesteps[0].shape, (2,))
        self.assertTrue(
            torch.equal(recorded_timesteps[0], recorded_timesteps[0][0].expand(2))
        )

        reproduced_generator = torch.Generator(device=device)
        reproduced_generator.manual_seed(seed)
        with diagnostics._pinned_timestep(int(recorded_timesteps[0][0])) as pin:
            reproduced = train_hoi_prior._forward_losses(
                model,
                diffusion,
                batch,
                values["parents_24"],
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
                cfg,
                generator=reproduced_generator,
            )
        self.assertEqual(pin.substitutions, 1)

        self.assertEqual(set(reproduced), set(reference))
        discrepancies = {}
        for key in reproduced:
            if not torch.equal(reproduced[key], reference[key]):
                delta = (reproduced[key] - reference[key]).abs()
                discrepancies[key] = {
                    "mismatched_elements": int(
                        torch.count_nonzero(reproduced[key] != reference[key])
                    ),
                    "max_abs_difference": float(delta.max()),
                }
        self.assertFalse(discrepancies, discrepancies)

    def test_probe_hard_raises_unless_one_timestep_is_substituted(self):
        for substitutions in (0, 2):
            with self.subTest(substitutions=substitutions):
                with tempfile.TemporaryDirectory() as directory:
                    context = mock.MagicMock()
                    context.__enter__.return_value = mock.Mock(
                        substitutions=substitutions
                    )
                    with mock.patch.object(
                        diagnostics, "_pinned_timestep", return_value=context
                    ):
                        with self.assertRaisesRegex(
                            AssertionError,
                            f"exactly one.*observed {substitutions}",
                        ):
                            self._run(directory, 3.0)

    def test_probe_component_keys_track_trainer_loss_keys(self):
        from train_hoi_prior import _loss_keys

        _values, _batch, cfg = _probe_fixture()
        deliberately_excluded = {
            "reconstruction",
            "contact",
            "velocity",
            "object_goal",
            "contact_accuracy",
        }
        self.assertEqual(
            set(diagnostics._GEOMETRY_GRADIENT_COMPONENTS),
            set(_loss_keys(cfg)) - deliberately_excluded,
        )

    def test_probe_rejects_specialized_forward_paths(self):
        values, batch, cfg = _probe_fixture()
        cases = (
            ("d2ag_selfcond_relation_source", "_is_d2ag"),
            ("d2ad_local_frame_interaction_adapter", "_is_d2ad"),
            ("d2ae_sparse_relation_field", "_is_sparse_relation"),
            ("d2z_immutable_gt_near_ground_gating", "_is_d2z"),
            ("d2ab_predicted_support_no_slip", "_is_d2ab"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"synthetic-checkpoint")
            for config_key, predicate_name in cases:
                with self.subTest(predicate=predicate_name):
                    branch_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
                    branch_cfg[config_key] = True
                    with self.assertRaisesRegex(ValueError, predicate_name):
                        diagnostics.geometry_weight_derivation_probe(
                            _SyntheticModel(),
                            GaussianDiffusion(500),
                            [batch],
                            values["parents_24"],
                            values["position_minimum"],
                            values["position_maximum"],
                            values["object_minimum"],
                            values["object_maximum"],
                            branch_cfg,
                            checkpoint_path=checkpoint,
                            output_path=Path(directory) / "unused.json",
                            window_count=2,
                            timesteps=(250,),
                        )

    def test_corrupted_joint_position_geometry_gradient_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                diagnostics.torch, "count_nonzero", return_value=torch.tensor(1)
            ):
                with self.assertRaisesRegex(AssertionError, "channels 3:84"):
                    self._run(directory, 0.0)


if __name__ == "__main__":
    unittest.main()
