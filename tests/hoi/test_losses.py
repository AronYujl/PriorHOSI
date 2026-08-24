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
from tools import measure_hoi_geometry_gradient as measure_geometry_tool


def _mask_loss_fixture(left, right, contacts):
    active_frames = len(left)
    frames = loss_module.REPRESENTATION.history_frames + active_frames
    fk = torch.zeros(1, frames, 24, 3)
    surface = torch.zeros(1, frames, 1, 3)
    labels = torch.zeros(1, frames, 4)
    fk[0, -active_frames:, 22, 0] = torch.tensor(left)
    fk[0, -active_frames:, 23, 0] = torch.tensor(right)
    labels[0, -active_frames:, :2] = torch.tensor(contacts)
    return fk, surface, labels


class HandObjectMaskModeTests(unittest.TestCase):
    def _loss(self, left, right, contacts, mode):
        return loss_module.masked_hand_object_distance_loss(
            *_mask_loss_fixture(left, right, contacts),
            hand_object_contact_mask_mode=mode,
        )

    def test_left_only_frame_charges_only_left_palm(self):
        for mode in ("per_hand_global", "per_hand_per_frame"):
            self.assertEqual(float(self._loss([2.0], [9.0], [[1, 0]], mode)), 4.0)

    def test_right_only_frame_charges_only_right_palm(self):
        for mode in ("per_hand_global", "per_hand_per_frame"):
            self.assertEqual(float(self._loss([9.0], [3.0], [[0, 1]], mode)), 9.0)

    def test_both_frame_averages_two_contacting_palms(self):
        for mode in ("per_hand_global", "per_hand_per_frame"):
            self.assertEqual(float(self._loss([2.0], [4.0], [[1, 1]], mode)), 10.0)

    def test_no_contact_frame_is_excluded(self):
        for mode in ("per_hand_global", "per_hand_per_frame"):
            value = self._loss([2.0, 100.0], [9.0, 100.0], [[1, 0], [0, 0]], mode)
            self.assertEqual(float(value), 4.0)

    def test_candidate_a_exact_global_formula(self):
        value = self._loss([1.0, 3.0], [2.0, 4.0], [[1, 0], [1, 1]], "per_hand_global")
        self.assertAlmostEqual(
            float(value), (1.0 + 9.0 + 16.0) / 3.0, delta=1e-6
        )

    def test_candidate_b_exact_per_frame_formula(self):
        value = self._loss([1.0, 3.0], [2.0, 4.0], [[1, 0], [1, 1]], "per_hand_per_frame")
        self.assertEqual(float(value), (1.0 + (9.0 + 16.0) / 2.0) / 2.0)

    def test_default_sealed_path_is_bitwise_unchanged(self):
        fk, surface, labels = _mask_loss_fixture(
            [1.0, 3.0], [2.0, 4.0], [[1, 0], [0, 0]]
        )
        default = loss_module.masked_hand_object_distance_loss(fk, surface, labels)
        explicit = loss_module.masked_hand_object_distance_loss(
            fk, surface, labels, hand_object_contact_mask_mode="sealed"
        )
        nearest = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        engaged = torch.tensor([[True, False]])
        per_frame = nearest.square().mean(dim=-1)
        weight = engaged.to(per_frame)
        pre_change = (per_frame * weight).sum() / weight.sum().clamp_min(1.0)
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.equal(default, pre_change))

    def test_empty_denominator_returns_finite_zero(self):
        for mode in ("per_hand_global", "per_hand_per_frame"):
            value = self._loss([1.0], [2.0], [[0, 0]], mode)
            self.assertTrue(torch.isfinite(value))
            self.assertEqual(float(value), 0.0)

    def test_invalid_modes_fail_closed(self):
        inputs = _mask_loss_fixture([1.0], [2.0], [[1, 0]])
        with self.assertRaisesRegex(ValueError, "unknown.*mask mode"):
            loss_module.masked_hand_object_distance_loss(
                *inputs, hand_object_contact_mask_mode="mystery"
            )
        with self.assertRaisesRegex(ValueError, "require zero hinge"):
            loss_module.masked_hand_object_distance_loss(
                *inputs, hinge=0.02,
                hand_object_contact_mask_mode="per_hand_global",
            )


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


class TrainerMaskModeThreadingTests(unittest.TestCase):
    @staticmethod
    def _forward(cfg, values, batch):
        import train_hoi_prior

        generator = torch.Generator(device=batch["x"].device)
        generator.manual_seed(42)
        return train_hoi_prior._forward_losses(
            _SyntheticModel(), GaussianDiffusion(500), batch,
            values["parents_24"], values["position_minimum"],
            values["position_maximum"], values["object_minimum"],
            values["object_maximum"], cfg, generator=generator,
        )

    def test_per_hand_per_frame_reaches_and_invokes_the_real_reducer(self):
        values, batch, cfg = _probe_fixture()
        cfg.hand_object_contact_mask_mode = "per_hand_per_frame"
        real_reducer = loss_module._reduce_per_hand_per_frame
        with mock.patch.object(
            loss_module, "_reduce_per_hand_per_frame", wraps=real_reducer
        ) as recorder:
            result = self._forward(cfg, values, batch)
        self.assertIn("hand_object_contact_geometry", result)
        recorder.assert_called_once()

    def test_absent_mode_reaches_sealed_branch_without_per_hand_reducers(self):
        values, batch, cfg = _probe_fixture()
        real_masked_loss = loss_module.masked_hand_object_distance_loss
        with mock.patch.object(
            loss_module, "masked_hand_object_distance_loss", wraps=real_masked_loss
        ) as masked, mock.patch.object(
            loss_module, "_reduce_per_hand_global", wraps=loss_module._reduce_per_hand_global
        ) as global_reducer, mock.patch.object(
            loss_module, "_reduce_per_hand_per_frame",
            wraps=loss_module._reduce_per_hand_per_frame,
        ) as frame_reducer:
            self._forward(cfg, values, batch)
        self.assertEqual(
            masked.call_args.kwargs["hand_object_contact_mask_mode"], "sealed"
        )
        global_reducer.assert_not_called()
        frame_reducer.assert_not_called()


class GeometryLossModeContractTests(unittest.TestCase):
    def test_geometry_losses_requires_an_explicit_mask_mode(self):
        values, batch, cfg = _probe_fixture()
        with self.assertRaisesRegex(TypeError, "mask_mode"):
            diagnostics._geometry_losses(
                values["prediction"], batch, values["parents_24"],
                values["position_minimum"], values["position_maximum"],
                values["object_minimum"], values["object_maximum"], cfg,
                weight=1.0, detach_object=False, detach_root=False,
            )


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
    def _manifest(self):
        return {
            "sampling": {"windows_per_shard": 256},
            "dataset_config_fingerprint": {},
            "provenance": {
                "tool_sha256": diagnostics.DERIVATION_MANIFEST_TOOL_SHA256
            },
            "coverage": {"accepted": True,
                         "both_frame_fraction_of_engaged": 0.5,
                         "corpus_reference": 0.5},
            "allocation_quantization_check": {"accepted": True,
                                              "engaged_window_fraction": 0.8},
            "shards": [
                {"shard_id": shard, "window_indices": list(
                    range(shard * 256, (shard + 1) * 256)), "coverage": {
                    "accepted": True, "both_frame_fraction_of_engaged": 0.5},
                 "allocation_quantization_check": {"accepted": True}}
                for shard in range(4)
            ],
        }

    @staticmethod
    def _batches(values, shard, batch_size=16):
        base = _probe_fixture()[1]
        batches = []
        for offset in range(0, 256, batch_size):
            batch = {}
            for key, value in base.items():
                if torch.is_tensor(value) and value.ndim and value.shape[0] == 2:
                    repeats = batch_size // 2
                    batch[key] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
                else:
                    batch[key] = value
            batch["window_index"] = torch.arange(
                shard * 256 + offset, shard * 256 + offset + batch_size, dtype=torch.long
            )
            batches.append(batch)
        return batches

    def _run(self, directory, mask_mode=None, *, measurement_mode="paired_joint",
             shards=(0, 1, 2, 3), timesteps=None, global_rng_trunk=False,
             variant_scale=1.2):
        import train_hoi_prior

        values, _batch, cfg = _probe_fixture()
        shards = tuple(shards)
        timesteps = diagnostics.DERIVATION_TIMESTEPS if timesteps is None else tuple(timesteps)
        cfg.hand_object_contact_weight = 0.0
        checkpoint = Path(directory) / "checkpoint.pth"
        checkpoint.write_bytes(b"synthetic-checkpoint")
        output = Path(directory) / "weight.json"
        loaders = {shard: self._batches(values, shard) for shard in shards}
        rc1 = []
        for offset in range(0, 256, 32):
            left = loaders[0][offset // 16]
            right = loaders[0][offset // 16 + 1]
            rc1.append({
                key: torch.cat((left[key], right[key])) if torch.is_tensor(left[key]) and left[key].ndim
                and left[key].shape[0] == 16 else left[key]
                for key in left
            })

        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.01))
                self.forward_calls = 0

            def forward(self, noisy, timesteps, text_embedding, object_bps, goals, progress):
                del timesteps, text_embedding, object_bps, goals, progress
                self.forward_calls += 1
                prediction = noisy * 0.0 + self.bias
                if global_rng_trunk:
                    prediction = prediction * (1.0 + 0.5 * torch.rand(noisy.shape[0], 1, 1))
                return prediction

        model = ConstantModel()
        geometry_ids = []

        def fake_forward(model, diffusion, batch, parents, minimum, maximum, object_minimum,
                         object_maximum, cfg, *, generator=None, **kwargs):
            del diffusion, parents, minimum, maximum, object_minimum, object_maximum, cfg, kwargs
            timestep = torch.randint(0, 500, (batch["x"].shape[0],), generator=generator)
            noisy = torch.randn(batch["x"].shape, generator=generator)
            prediction = model(noisy, timestep, batch["text_embedding"], batch["object_bps"],
                               batch["goals"], batch["progress"])
            def term(start, end):
                return prediction[..., start:end].square().mean()
            return {
                "joint_position": term(0, 84), "joint_rotation": term(84, 216),
                "fk": term(0, 3), "object_translation": term(216, 219),
                "object_rotation": term(219, 228), "object_surface": term(216, 228),
                "contact": term(228, 232), "velocity": term(0, 3),
                "object_goal": term(216, 219),
                "total": sum(term(start, end) for start, end in (
                    (0, 84), (84, 216), (216, 219), (219, 228), (228, 232))),
            }

        def fake_geometry(prediction, *args, **kwargs):
            mode = kwargs["mask_mode"]
            geometry_ids.append((mode, id(prediction)))
            active = prediction[:, 2:]
            scale = 1.0 if mode == "sealed" else variant_scale
            return {"hand_object_contact_geometry": scale * (
                active[..., 0:3].square().mean()
                + active[..., 84:216].square().mean()
                + active[..., 216:228].square().mean()
            )}

        pin = mock.MagicMock()
        pin.__enter__.return_value = mock.Mock(substitutions=1)
        with mock.patch.object(diagnostics, "_validate_derivation_manifest"), \
             mock.patch.object(diagnostics, "_geometry_losses", side_effect=fake_geometry), \
             mock.patch.object(diagnostics, "_pinned_timestep", return_value=pin), \
             mock.patch.object(train_hoi_prior, "_forward_losses", side_effect=fake_forward), \
             mock.patch.object(train_hoi_prior, "_move_batch", side_effect=lambda batch, device: batch):
            result = diagnostics.geometry_weight_derivation_probe(
                model, GaussianDiffusion(500), loaders, values["parents_24"],
                values["position_minimum"], values["position_maximum"],
                values["object_minimum"], values["object_maximum"], cfg,
                checkpoint_path=checkpoint, output_path=output, manifest=self._manifest(),
                rc1_loader=rc1, mask_mode=mask_mode, measurement_mode=measurement_mode,
                shard_ids=shards, timesteps=timesteps,
            )
        self._forward_calls = model.forward_calls
        self._geometry_ids = geometry_ids
        return result

    def test_zero_report_has_exact_new_top_level_schema_and_shared_modes(self):
        expected = {
            "probe", "seed", "window_count", "batch_count", "timesteps", "timestep_seam",
            "measurement_mode", "measured_mask_modes", "hand_object_contact_mask_mode",
            "cfg_mask_mode_selects_measurement", "configured_hand_object_contact_weight",
            "geometry_from_separate_call", "spec_sha256", "probe_sha256", "checkpoint_path",
            "checkpoint_sha256", "sampling", "pairing", "shared", "modes", "gates_shared",
            "candidate", "report_only", "timing", "self_check", "provenance",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(directory)
        self.assertEqual(set(result), expected)
        self.assertEqual(result["measurement_mode"], "paired_joint")
        self.assertEqual(result["measured_mask_modes"], ["sealed", "per_hand_per_frame"])
        self.assertEqual(set(result["shared"]), {"per_shard"})
        self.assertEqual(set(result["modes"]), {"sealed", "per_hand_per_frame"})
        self.assertEqual(result["candidate"]["source"], diagnostics.DERIVATION_CANDIDATE_SOURCE)
        self.assertFalse(result["candidate"]["produced"])

    def test_paired_joint_uses_one_forward_and_one_prediction_for_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(directory)
        self.assertEqual(self._forward_calls, 4 * 5 * 16 + 8)
        self.assertTrue(result["pairing"]["L2_prediction_is_identical_object"])
        for offset in range(0, len(self._geometry_ids), 2):
            self.assertEqual(self._geometry_ids[offset][1], self._geometry_ids[offset + 1][1])

    def test_nonzero_paired_weight_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "E_PAIRED_REQUIRES_ZERO_WEIGHT"):
            diagnostics._require_paired_zero_weight("paired_joint", 3.0)

    def test_candidate_has_only_variant_source_and_l3_cannot_produce(self):
        self.assertEqual(diagnostics.DERIVATION_CANDIDATE_SOURCE, "modes.per_hand_per_frame.aggregate.w_geom_star")
        self.assertFalse(diagnostics._candidate_is_allowed("single_mode_l3", {"G1": True}, True, False, []))

    def test_per_cell_seed_is_order_independent(self):
        cells = [(shard, timestep) for shard in range(4) for timestep in diagnostics.DERIVATION_TIMESTEPS]
        first = {(s, t): diagnostics._per_cell_seed(s, t) for s, t in cells}
        second = {(s, t): diagnostics._per_cell_seed(s, t) for s, t in reversed(cells)}
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 20)

    def test_cell_numerics_are_independent_of_execution_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paired = self._run(directory, global_rng_trunk=True)
        with tempfile.TemporaryDirectory() as directory:
            alone = self._run(
                directory, mask_mode="sealed", measurement_mode="single_mode_l3",
                shards=(0,), timesteps=(250,), global_rng_trunk=True,
            )
        with tempfile.TemporaryDirectory() as directory:
            reordered = self._run(
                directory, mask_mode="sealed", measurement_mode="single_mode_l3",
                shards=(0,), timesteps=(499, 250, 0), global_rng_trunk=True,
            )
        paired_nongeometry = paired["shared"]["per_shard"]["0"]["gradient_l2_nongeometry"]["250"]
        alone_nongeometry = alone["shared"]["per_shard"]["0"]["gradient_l2_nongeometry"]["250"]
        reordered_nongeometry = reordered["shared"]["per_shard"]["0"]["gradient_l2_nongeometry"]["250"]
        self.assertEqual(paired_nongeometry, alone_nongeometry)
        self.assertEqual(paired_nongeometry, reordered_nongeometry)
        paired_geometry = paired["modes"]["sealed"]["per_shard"]["0"]["geometry_by_channel"]["250"]
        alone_geometry = alone["modes"]["sealed"]["per_shard"]["0"]["geometry_by_channel"]["250"]
        reordered_geometry = reordered["modes"]["sealed"]["per_shard"]["0"]["geometry_by_channel"]["250"]
        self.assertEqual(paired_geometry, alone_geometry)
        self.assertEqual(paired_geometry, reordered_geometry)
        full_nongeometry = paired["shared"]["per_shard"]["0"]["gradient_l2_nongeometry"]
        self.assertNotEqual(full_nongeometry["250"]["total"], full_nongeometry["0"]["total"])
        self.assertNotEqual(full_nongeometry["250"]["total"], full_nongeometry["499"]["total"])
        self.assertEqual(len(set(paired["pairing"]["cell_global_seed"]["0"].values())), 5)

    def test_per_cell_global_seed_is_distinct_and_order_independent(self):
        cells = [(shard, timestep) for shard in range(4) for timestep in diagnostics.DERIVATION_TIMESTEPS]
        first = {(s, t): diagnostics._per_cell_global_seed(s, t) for s, t in cells}
        second = {(s, t): diagnostics._per_cell_global_seed(s, t) for s, t in reversed(cells)}
        noise = {diagnostics._per_cell_seed(s, t) for s, t in cells}
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 20)
        self.assertTrue(set(first.values()).isdisjoint(noise))

    def test_g8_compares_one_mask_mode_against_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(directory, variant_scale=6.0)
        for mode in ("sealed", "per_hand_per_frame"):
            with self.subTest(mode=mode):
                self.assertIs(result["modes"][mode]["gates"]["G8_parameter_space_crosscheck"], True)
                self.assertLess(
                    result["report_only"]["RC2_parameter_space"]["ratio_to_output_space"][mode], 3.0
                )

    def test_support_assertions_and_reference_fields_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(directory)
        self.assertTrue(all(result["self_check"][key] for key in (
            "geometry_gradient_on_joint_positions_3_84_exactly_zero",
            "geometry_gradient_on_contact_228_232_exactly_zero",
            "geometry_gradient_on_history_frames_exactly_zero",
            "geometry_pythagoras_holds", "reference_norm_tensor_sum_equals_quadrature")))
        cell = result["modes"]["per_hand_per_frame"]["per_shard"]["0"]["geometry_by_channel"]["250"]
        self.assertIn("G_human", cell)
        self.assertLessEqual(cell["pythagoras_relative_error"], 1e-12)

    def test_side_gate_cannot_be_masked_by_combined(self):
        values = {shard: {timestep: {"human": 1.0 + shard * 100.0,
            "object": 1.0, "combined": 1.0}
            for timestep in diagnostics.DERIVATION_TIMESTEPS}
            for shard in range(4)}
        with self.assertRaisesRegex(ValueError, "E_DERIVATION_SIDE_DISPERSION_MASKED"):
            diagnostics._side_dispersion_gates(values, range(4), diagnostics.DERIVATION_TIMESTEPS, 1.0)

    def test_rc2_three_x_stop_and_unverified_scope_are_locked(self):
        self.assertEqual(diagnostics._rc2_ratio(3.0, 1.0), (3.0, False))
        self.assertEqual(diagnostics._rc2_ratio(2.99, 1.0), (2.99, True))
        self.assertFalse(diagnostics._candidate_is_allowed(
            "paired_joint", {"G1_manifest": True, "G3_support_assertions": True,
            "G9_pairing": True, "G10_provenance": True}, True, False, [3.0]))
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(directory)
        self.assertEqual(result["candidate"]["parameter_space_unverified_cells"], 19)

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
                            window_count=256,
                            timesteps=diagnostics.DERIVATION_TIMESTEPS,
                        )

    def test_corrupted_joint_position_geometry_gradient_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(diagnostics.torch, "count_nonzero", return_value=torch.tensor(1)):
                with self.assertRaisesRegex(ValueError, "E_DERIVATION_SUPPORT_ASSERTION"):
                    self._run(directory)


class GeometryGradientToolContractTests(unittest.TestCase):
    def test_l3_verdict_accepts_single_timestep_arms_and_rejects_a_wrong_cell(self):
        def artifact(timesteps=(250,), global_seed=17):
            return {
                "timesteps": list(timesteps),
                "pairing": {
                    "input_sha256": {"0": {"250": "input"}},
                    "cell_seed": {"0": {"250": 13}},
                    "cell_global_seed": {"0": {"250": global_seed}},
                },
                "shared": {"per_shard": {"0": {
                    "gradient_l2_nongeometry": {"250": {"total": 1.0}},
                }}},
                "modes": {
                    mode: {"per_shard": {"0": {"geometry_by_channel": {
                        "250": {"combined": 2.0},
                    }}}}
                    for mode in ("sealed", "per_hand_per_frame")
                },
            }

        paired = artifact((0, 125, 250, 375, 499))
        passed, crosscheck = measure_geometry_tool._l3_verdict(
            [artifact(), artifact()], paired
        )
        self.assertTrue(passed)
        self.assertTrue(crosscheck["cell_timestep_equal"])

        passed, crosscheck = measure_geometry_tool._l3_verdict(
            [artifact((0,)), artifact()], paired
        )
        self.assertFalse(passed)
        self.assertFalse(crosscheck["cell_timestep_equal"])

        passed, crosscheck = measure_geometry_tool._l3_verdict(
            [artifact(), artifact(global_seed=18)], paired
        )
        self.assertFalse(passed)
        self.assertFalse(crosscheck["cell_global_seed_equal"])

    def test_manifest_global_indices_map_to_subset_positions_in_manifest_order(self):
        positions = measure_geometry_tool._manifest_subset_positions([10, 20, 30, 40], [10, 30], 0)
        self.assertEqual(positions, [0, 2])
        with self.assertRaisesRegex(RuntimeError, "E_MANIFEST_INDEX_NOT_IN_DATASET"):
            measure_geometry_tool._manifest_subset_positions([10, 20], [10, 30], 0)

    def test_cli_preflight_locks_checkpoint_path_and_all_four_hashes(self):
        root = measure_geometry_tool.ROOT
        checkpoint = (root / measure_geometry_tool.CHECKPOINT_RELATIVE).resolve()
        args = measure_geometry_tool.build_parser().parse_args([
            "weight-derivation", "--checkpoint", str(checkpoint),
            "--output", str(Path("/tmp/b1b-ii-output.json")),
        ])
        expected = {
            checkpoint: measure_geometry_tool.EXPECTED_CHECKPOINT_SHA256,
            (root / measure_geometry_tool.MANIFEST_DEFAULT).resolve(): measure_geometry_tool.EXPECTED_MANIFEST_SHA256,
            (root / measure_geometry_tool.SPLIT_RELATIVE).resolve(): measure_geometry_tool.EXPECTED_SPLIT_SHA256,
            (root / measure_geometry_tool.NORM_RELATIVE).resolve(): measure_geometry_tool.EXPECTED_NORM_SHA256,
        }
        with mock.patch.object(
            measure_geometry_tool, "_sha256", side_effect=lambda path: expected[path.resolve()]
        ):
            measure_geometry_tool._validate_fixed_inputs(args)
        self.assertIn("p1-hoi-p12-frame-repair-baseline-s42-20260819_windows299520000.pth", str(checkpoint))
        with self.assertRaisesRegex(ValueError, "E_MANIFEST_TOOL_SHA_MISMATCH"):
            diagnostics._validate_derivation_manifest({"provenance": {"tool_sha256": "bad"}}, None)

    def test_finalize_writes_output_hash_as_a_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            temporary = directory / "temporary.json"
            output = directory / "output.json"
            temporary.write_text("temporary\n", encoding="utf-8")
            with mock.patch.object(measure_geometry_tool, "provenance",
                                   return_value={"git_dirty": False}):
                measure_geometry_tool._finalize({"value": 1}, temporary, output)
            sidecar = output.with_name("output.json.sha256")
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar.read_text(encoding="utf-8").split()[0], measure_geometry_tool._sha256(output))


if __name__ == "__main__":
    unittest.main()
