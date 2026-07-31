from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.diffusion import GaussianDiffusion, prepare_clean_x0  # noqa: E402
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AE,
    HOI_ARCHITECTURE_D2AF,
    HOI_ARCHITECTURE_D2AG,
    build_expert,
    load_trained_hoi_prior,
)
from priors.representation import REPRESENTATION  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_DIAGNOSTIC_VARIANTS,
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    SPARSE_RELATION_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT,
    SparseCurrentStateRelationField,
    build_d2ag_relation_source,
    selfcond_relation_source_contract_metadata,
    validate_selfcond_relation_source_contract,
)
from tools.diagnose_hoi_d2ae import synthetic_inputs  # noqa: E402
import train_hoi_prior as hoi_trainer  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AF_MAXIMUM_ETA_HOURS,
    D2AF_MINIMUM_THROUGHPUT,
    D2AG_MAXIMUM_ETA_HOURS,
    D2AG_MINIMUM_THROUGHPUT,
    _d2ag_mask_seed,
    _forward_losses,
    _state_dict_sha256,
    _validate_d2ag_contract,
    _validate_fk_foot_temporal_routing_mode,
)


EXPECTED_INITIAL_STATE_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
PARITY_TOLERANCE = 1e-6
LOSS_KEYS = (
    "total", "reconstruction", "joint_position", "joint_rotation",
    "object_translation", "object_rotation", "contact", "fk",
    "object_surface", "velocity", "object_goal", "contact_accuracy",
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


def build_variant(variant: str):
    torch.manual_seed(42)
    return build_expert(
        "hoi", dim_model=512, num_heads=16, num_layers=8,
        architecture_variant=variant,
    ).eval()


def training_batch(batch: int):
    values, arguments = relation_arguments(batch)
    return values, arguments, {
        "x": values["current"].clone(),
        "text_embedding": values["text"],
        "object_bps": values["global_bps"],
        "goals": values["goals"],
        "progress": values["progress"].abs() + 1.0,
        "rest_human_offsets": torch.zeros(batch, 24, 3),
        "terminal_window": torch.zeros(batch),
        "rest_object_points": values["rest_object_points"],
        "world_to_local_rotation": values["world_to_local_rotation"],
        "object_rotation_reference": values["object_rotation_reference"],
    }


def d2ag_cfg():
    return OmegaConf.create({
        "seed": 42,
        "d2ad_local_frame_interaction_adapter": False,
        "d2ae_sparse_relation_field": False,
        "d2af_sqrt_alpha_bar_reliability": False,
        "d2ag_selfcond_relation_source": True,
        "d2z_immutable_gt_near_ground_gating": False,
        "d2ab_predicted_support_no_slip": False,
        "fk_weight": 0.3569973401779424,
        "object_surface_weight": 0.4772322188400037,
        "velocity_weight": 0.1,
        "goal_weight": 1.0,
        "fk_foot_temporal_routing": True,
        "routed_foot_residual_multiplier": 1.0,
    })


class D2AGFieldTests(unittest.TestCase):
    def test_parameter_and_seed42_state_are_exactly_d2ae(self):
        d2ae = build_variant(HOI_ARCHITECTURE_D2AE)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        self.assertEqual(
            sum(p.numel() for p in d2ag.parameters()), TOTAL_PARAMETER_COUNT,
        )
        self.assertEqual(
            sum(p.numel() for p in d2ag.network.sparse_relation_field.parameters()),
            SPARSE_RELATION_PARAMETER_COUNT,
        )
        self.assertEqual(d2ae.state_dict().keys(), d2ag.state_dict().keys())
        for key, value in d2ae.state_dict().items():
            self.assertTrue(torch.equal(value, d2ag.state_dict()[key]), key)
        self.assertEqual(
            _state_dict_sha256(d2ag.state_dict()), EXPECTED_INITIAL_STATE_SHA256,
        )
        field = d2ag.network.sparse_relation_field
        self.assertIsNone(field.sqrt_alpha_bar)
        self.assertFalse(field.diffusion_reliability)
        self.assertTrue(field.selfcond_relation_source)
        self.assertNotIn(
            "network.sparse_relation_field.sqrt_alpha_bar", d2ag.state_dict(),
        )

    def test_per_sample_source_selection_and_history_pin(self):
        values, _ = relation_arguments(4)
        current = values["current"]
        estimate = torch.randn(2, 16, 232)
        index = torch.tensor([0, 3], dtype=torch.long)
        source = build_d2ag_relation_source(current, estimate, index=index)
        # Selected rows take the estimate on the variable frames only.
        self.assertTrue(torch.equal(source[0, 2:], estimate[0, 2:]))
        self.assertTrue(torch.equal(source[3, 2:], estimate[1, 2:]))
        # Unselected rows are bitwise x_t, i.e. exact D2-AE behavior.
        for row in (1, 2):
            self.assertTrue(torch.equal(source[row], current[row]))
        # s[:, :2] == x_t[:, :2] on both sides, exactly.
        self.assertTrue(torch.equal(
            source[:, :REPRESENTATION.history_frames],
            current[:, :REPRESENTATION.history_frames],
        ))
        self.assertIs(build_d2ag_relation_source(current, None), current)

    def test_unselected_and_estimate_forwards_equal_d2ae_reference(self):
        values, arguments = relation_arguments(3)
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        d2ae = build_variant(HOI_ARCHITECTURE_D2AE)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        common = (
            values["current"], timesteps, values["text"],
            values["global_bps"], values["goals"], values["progress"],
        )
        with torch.no_grad():
            reference = d2ae(*common, **arguments)
            unselected = d2ag(*common, **arguments)
            estimate = d2ag(*common, **arguments, relation_source_estimate=True)
            # An explicit all-x_t source must also reproduce the reference.
            explicit = d2ag(
                *common,
                **arguments,
                relation_source=build_d2ag_relation_source(
                    values["current"], None,
                ),
            )
        for name, value in (
            ("unselected", unselected),
            ("estimate", estimate),
            ("explicit_xt_source", explicit),
        ):
            with self.subTest(forward=name):
                self.assertTrue(torch.equal(reference, value))
                self.assertLessEqual(
                    float((reference - value).abs().max()), PARITY_TOLERANCE,
                )

    def test_self_conditioned_source_changes_the_output(self):
        values, arguments = relation_arguments(3)
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        d2ag.network.sparse_relation_field.alpha.data.fill_(0.5)
        source = build_d2ag_relation_source(
            values["current"], torch.randn(3, 16, 232),
        )
        common = (
            values["current"], timesteps, values["text"],
            values["global_bps"], values["goals"], values["progress"],
        )
        with torch.no_grad():
            base = d2ag(*common, **arguments)
            shifted = d2ag(*common, **arguments, relation_source=source)
        self.assertGreater(float((base - shifted).abs().max()), 1e-4)

    def test_timesteps_are_diagnostic_only_for_the_full_variant(self):
        values, arguments = relation_arguments(3)
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        field = build_variant(HOI_ARCHITECTURE_D2AG).network.sparse_relation_field
        field.alpha.data.fill_(0.3)
        self.assertEqual(field._diagnostic_variant, "full")
        motion = torch.randn(3, 16, 512)
        source = build_d2ag_relation_source(
            values["current"], torch.randn(3, 16, 232),
        )
        with torch.no_grad():
            for label, extra in (
                ("xt_source", {}),
                ("selfcond_source", {"relation_source": source}),
            ):
                with_timesteps = field(
                    motion, values["current"], **arguments,
                    timesteps=timesteps, **extra,
                )
                without_timesteps = field(
                    motion, values["current"], **arguments,
                    timesteps=None, **extra,
                )
                with self.subTest(source=label):
                    self.assertTrue(
                        torch.equal(with_timesteps, without_timesteps)
                    )

    def test_high_t_restriction_uses_timesteps_and_falls_back_to_xt(self):
        values, arguments = relation_arguments(3)
        timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
        field = build_variant(HOI_ARCHITECTURE_D2AG).network.sparse_relation_field
        field.alpha.data.fill_(0.3)
        motion = torch.randn(3, 16, 512)
        source = build_d2ag_relation_source(
            values["current"], torch.randn(3, 16, 232),
        )
        with torch.no_grad():
            full = field(
                motion, values["current"], **arguments,
                timesteps=timesteps, relation_source=source,
            )
            substituted = field(
                motion, values["current"], **arguments, timesteps=timesteps,
            )
            field.set_diagnostic_variant("high_t_restricted")
            restricted = field(
                motion, values["current"], **arguments,
                timesteps=timesteps, relation_source=source,
            )
            field.set_diagnostic_variant("source_substituted_xt")
            reverted = field(
                motion, values["current"], **arguments,
                timesteps=timesteps, relation_source=source,
            )
        # t<250 keeps the self-conditioned source, t>=250 falls back to x_t.
        self.assertTrue(torch.equal(restricted[:2], full[:2]))
        self.assertTrue(torch.equal(restricted[2], substituted[2]))
        self.assertTrue(torch.equal(reverted, substituted))
        field.set_diagnostic_variant("high_t_restricted")
        with self.assertRaises(ValueError):
            field(
                motion, values["current"], **arguments,
                timesteps=None, relation_source=source,
            )

    def test_object_displacement_only_moves_variable_anchor_object_channels(self):
        values, arguments = relation_arguments(2)
        timesteps = torch.zeros(2, dtype=torch.long)
        field = build_variant(HOI_ARCHITECTURE_D2AG).network.sparse_relation_field
        field.alpha.data.fill_(0.3)
        field.set_capture(True)
        motion = torch.randn(2, 16, 512)
        source = build_d2ag_relation_source(
            values["current"], torch.randn(2, 16, 232),
        )
        with torch.no_grad():
            plain = field(
                motion, values["current"], **arguments,
                timesteps=timesteps, relation_source=source,
            )
            field.set_diagnostic_variant("object_displaced_counterfactual")
            displaced = field(
                motion, values["current"], **arguments,
                timesteps=timesteps, relation_source=source,
            )
        self.assertGreater(float((plain - displaced).abs().max()), 1e-5)
        self.assertEqual(tuple(D2AG_VARIABLE_ANCHORS), (5, 10, 15))

    def test_relation_source_restricted_to_d2ag_and_shape_checked(self):
        values, arguments = relation_arguments(2)
        timesteps = torch.zeros(2, dtype=torch.long)
        source = values["current"].clone()
        common = (
            values["current"], timesteps, values["text"],
            values["global_bps"], values["goals"], values["progress"],
        )
        for variant in (HOI_ARCHITECTURE_D2AE, HOI_ARCHITECTURE_D2AF):
            model = build_variant(variant)
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "restricted to D2-AG"):
                    model(*common, **arguments, relation_source=source)
                with self.assertRaisesRegex(ValueError, "restricted to D2-AG"):
                    model(*common, **arguments, relation_source_estimate=True)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        with self.assertRaises(ValueError):
            d2ag(*common, **arguments, relation_source=source[:, :8])
        with self.assertRaises(ValueError):
            d2ag(
                *common, **arguments,
                relation_source=source, relation_source_estimate=True,
            )
        field = SparseCurrentStateRelationField(512)
        with self.assertRaisesRegex(ValueError, "restricted to D2-AG"):
            field(
                torch.zeros(2, 16, 512), values["current"], **arguments,
                relation_source=source,
            )
        with self.assertRaises(ValueError):
            SparseCurrentStateRelationField(
                512, diffusion_reliability=True, selfcond_relation_source=True,
            )
        self.assertIn("high_t_restricted", D2AG_DIAGNOSTIC_VARIANTS)
        with self.assertRaises(ValueError):
            build_variant(HOI_ARCHITECTURE_D2AE).network.sparse_relation_field \
                .set_diagnostic_variant("high_t_restricted")

    def test_alpha_gradient_is_nonzero_under_a_partial_mask(self):
        values, arguments = relation_arguments(4)
        timesteps = torch.tensor([0, 249, 499, 120], dtype=torch.long)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        source = build_d2ag_relation_source(
            values["current"],
            torch.randn(2, 16, 232),
            index=torch.tensor([0, 2], dtype=torch.long),
        )
        common = (
            values["current"], timesteps, values["text"], values["global_bps"],
            values["goals"], values["progress"],
        )
        # At the locked alpha=0 initialization the gate is exactly zero, so only
        # alpha itself carries gradient; the field paths are checked below with
        # the gate activated, exactly as the D2-AE/D2-AF audits do.
        output = d2ag(*common, **arguments, relation_source=source)
        output.square().mean().backward()
        field = d2ag.network.sparse_relation_field
        self.assertEqual(float(field.alpha.detach()), 0.0)
        self.assertIsNotNone(field.alpha.grad)
        self.assertTrue(torch.isfinite(field.alpha.grad))
        self.assertNotEqual(float(field.alpha.grad), 0.0)
        d2ag.zero_grad(set_to_none=True)
        d2ag.network.set_sparse_relation_gate_override(0.1)
        output = d2ag(*common, **arguments, relation_source=source)
        output.square().mean().backward()
        for name, parameter in (
            ("point_encoder", field.point_encoder[0].weight),
            ("projection", field.projection.weight),
            ("temporal_embeddings", field.temporal_embeddings),
            (
                "trunk_layer_0",
                d2ag.network.transformer.layers[0].self_attn.in_proj_weight,
            ),
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.any(parameter.grad != 0))

    def test_subset_estimate_forward_matches_the_full_d2ae_forward(self):
        # The estimate runs on the selected subset only, so subset-vs-full-batch
        # numerical agreement is a real contract, not a tautology.
        values, arguments = relation_arguments(4)
        timesteps = torch.tensor([0, 249, 499, 120], dtype=torch.long)
        d2ae = build_variant(HOI_ARCHITECTURE_D2AE)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        index = torch.tensor([0, 2], dtype=torch.long)
        subset_arguments = dict(arguments)
        for key in (
            "rest_object_points",
            "world_to_local_rotation",
            "object_rotation_reference",
        ):
            subset_arguments[key] = arguments[key].index_select(0, index)
        with torch.no_grad():
            reference = d2ae(
                values["current"], timesteps, values["text"],
                values["global_bps"], values["goals"], values["progress"],
                **arguments,
            ).index_select(0, index)
            subset = d2ag(
                values["current"].index_select(0, index),
                timesteps.index_select(0, index),
                values["text"].index_select(0, index),
                values["global_bps"].index_select(0, index),
                values["goals"].index_select(0, index),
                values["progress"].index_select(0, index),
                **subset_arguments,
                relation_source_estimate=True,
            )
        self.assertLessEqual(
            float((reference - subset).abs().max()), PARITY_TOLERANCE,
        )

    def test_source_construction_never_mutates_the_noisy_state(self):
        values, _ = relation_arguments(4)
        current = values["current"]
        before = current.clone()
        source = build_d2ag_relation_source(
            current,
            torch.randn(2, 16, 232),
            index=torch.tensor([1, 3], dtype=torch.long),
        )
        self.assertTrue(torch.equal(current, before))
        self.assertNotEqual(source.data_ptr(), current.data_ptr())
        for bad in (
            (torch.randn(2, 16, 232), torch.tensor([0, 9], dtype=torch.long)),
            (torch.randn(2, 16, 232), torch.tensor([0], dtype=torch.long)),
            (torch.randn(2, 16, 232), torch.tensor([0.0, 1.0])),
            (torch.randn(2, 8, 232), torch.tensor([0, 1], dtype=torch.long)),
            (torch.randn(3, 16, 232), None),
        ):
            with self.subTest(case=tuple(getattr(bad[0], "shape", ()))):
                with self.assertRaises(ValueError):
                    build_d2ag_relation_source(current, bad[0], index=bad[1])
        with self.assertRaises(ValueError):
            build_d2ag_relation_source(
                current, None, index=torch.tensor([0], dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            build_d2ag_relation_source(current[:, :8], None)

    def test_estimate_source_is_detached_from_the_first_forward(self):
        values, arguments = relation_arguments(2)
        timesteps = torch.zeros(2, dtype=torch.long)
        d2ag = build_variant(HOI_ARCHITECTURE_D2AG)
        d2ag.network.sparse_relation_field.alpha.data.fill_(0.5)
        estimate = d2ag(
            values["current"], timesteps, values["text"], values["global_bps"],
            values["goals"], values["progress"], **arguments,
        )
        self.assertTrue(estimate.requires_grad)
        source = build_d2ag_relation_source(values["current"], estimate)
        self.assertFalse(source.requires_grad)
        self.assertIsNone(source.grad_fn)


class D2AGTrainSampleSymmetryTests(unittest.TestCase):
    def _recorder(self, variant: str):
        class Recorder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.architecture_variant = variant
                self.steps = []
                self.sources = []
                self.outputs = []
                self.currents = []

            def forward(self, current, timesteps, *args, **kwargs):
                del args
                self.steps.append(int(timesteps[0]))
                self.currents.append(current.clone())
                self.sources.append(
                    None if kwargs.get("relation_source") is None
                    else kwargs["relation_source"].clone()
                )
                value = current * 0.5 + 0.125
                self.outputs.append(value.clone())
                return value

        return Recorder()

    def _sample(
        self,
        recorder,
        *,
        batch: int = 2,
        generator_seed: int = 42,
        object_so3_x0: bool = False,
    ):
        values, arguments = relation_arguments(batch)
        diffusion = GaussianDiffusion()
        fixed_history = values["current"][:, :REPRESENTATION.history_frames].clone()
        diffusion.sample(
            recorder,
            fixed_history,
            values["text"],
            values["global_bps"],
            values["goals"],
            values["progress"],
            **arguments,
            generator=torch.Generator().manual_seed(generator_seed),
            object_so3_x0=object_so3_x0,
        )
        return fixed_history

    def test_first_reverse_step_uses_xt_and_trace_is_499_to_0(self):
        recorder = self._recorder(HOI_ARCHITECTURE_D2AG)
        self._sample(recorder)
        self.assertEqual(recorder.steps, list(reversed(range(500))))
        # t=499 has no prev_x0, so no explicit source is passed at all: the
        # field reads x_t exactly as D2-AE/D2-AF do.
        self.assertIsNone(recorder.sources[0])
        self.assertTrue(all(
            source is not None for source in recorder.sources[1:]
        ))

    def test_d2ae_sampling_never_receives_a_relation_source(self):
        recorder = self._recorder(HOI_ARCHITECTURE_D2AE)
        self._sample(recorder)
        self.assertEqual(recorder.steps, list(reversed(range(500))))
        self.assertTrue(all(source is None for source in recorder.sources))

    def test_prev_x0_is_the_raw_x0_hat_before_prepare_clean_x0(self):
        # object_so3_x0=True makes prepare_clean_x0 rewrite frames 2..15 of the
        # rotation block, which is the only observable difference between the raw
        # and the prepared x0_hat; the registered contract takes the raw one.
        recorder = self._recorder(HOI_ARCHITECTURE_D2AG)
        fixed_history = self._sample(recorder, object_so3_x0=True)
        history = REPRESENTATION.history_frames
        divergences = 0
        for step in range(1, 6):
            raw = recorder.outputs[step - 1]
            prepared = prepare_clean_x0(raw, fixed_history, object_so3_x0=True)
            source = recorder.sources[step]
            self.assertIsNotNone(source)
            with self.subTest(step=step):
                # The variable frames are the raw x0_hat, byte for byte.
                self.assertTrue(torch.equal(source[:, history:], raw[:, history:]))
                # The history block is pinned to x_t at this step.
                self.assertTrue(torch.equal(
                    source[:, :history],
                    recorder.currents[step][:, :history],
                ))
                if not torch.equal(source[:, history:], prepared[:, history:]):
                    divergences += 1
        # At least one step must actually differ from the prepared clean, so the
        # assertion above is not vacuous.
        self.assertGreater(divergences, 0)

    def test_window_chains_do_not_leak_prev_x0(self):
        first = self._recorder(HOI_ARCHITECTURE_D2AG)
        second = self._recorder(HOI_ARCHITECTURE_D2AG)
        self._sample(first)
        self._sample(second)
        # A second sample() call restarts the chain: its first step again has no
        # previous estimate and its whole trace matches the first call.
        self.assertIsNone(second.sources[0])
        self.assertEqual(first.steps, second.steps)
        for index, (left, right) in enumerate(zip(first.sources, second.sources)):
            with self.subTest(step=index):
                if left is None:
                    self.assertIsNone(right)
                else:
                    self.assertTrue(torch.equal(left, right))

    def test_train_and_sample_share_one_relation_source_builder(self):
        trainer_source = inspect.getsource(
            hoi_trainer._d2ag_relation_source_arguments
        )
        sampler_source = inspect.getsource(GaussianDiffusion.sample)
        for source in (trainer_source, sampler_source):
            self.assertIn("build_d2ag_relation_source(", source)
        self.assertIn("model.module", trainer_source)
        self.assertIn("DistributedDataParallel", trainer_source)
        self.assertIn("inner.eval()", trainer_source)
        self.assertIn("finally", trainer_source)
        # no_sync() belongs to gradient accumulation; the estimate pass must not
        # execute it.  Comment lines are excluded so the prose that explains the
        # rule does not satisfy or break the check.
        executable = "\n".join(
            line for line in trainer_source.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn("no_sync", executable)
        self.assertIn("no_sync", trainer_source)
        # The history pin lives in exactly one place.
        field_source = (ROOT / "code/priors/sparse_relation.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            field_source.count("return torch.cat((current[:, :history]"), 1,
        )

    def test_train_and_sample_source_tensors_agree_elementwise(self):
        values, arguments, batch = training_batch(4)
        cfg = d2ag_cfg()
        model = build_variant(HOI_ARCHITECTURE_D2AG)
        estimate = torch.randn(4, 16, 232)
        # Sampling side: prev_x0 is the full-batch raw x0_hat.
        sample_side = build_d2ag_relation_source(values["current"], estimate)
        # Training side with an all-True mask must produce the same tensor.
        with mock.patch.object(
            hoi_trainer, "_d2ag_selection_mask",
            return_value=torch.ones(4, dtype=torch.bool),
        ), mock.patch.object(
            hoi_trainer.torch.nn.Module, "__call__", autospec=True,
        ) as call:
            call.return_value = estimate
            train_side = hoi_trainer._d2ag_relation_source_arguments(
                model, batch, values["current"],
                torch.zeros(4, dtype=torch.long), arguments, cfg,
                processed_windows=0, rank=0,
            )["relation_source"]
        self.assertLessEqual(
            float((train_side - sample_side).abs().max()), PARITY_TOLERANCE,
        )
        self.assertTrue(torch.equal(train_side, sample_side))

    def test_estimate_forward_consumes_no_global_rng_and_restores_train(self):
        values, arguments, batch = training_batch(4)
        cfg = d2ag_cfg()
        model = build_variant(HOI_ARCHITECTURE_D2AG)
        model.train()
        observed = []
        real_forward = type(model).forward

        def spy(self, *args, **kwargs):
            observed.append(
                (bool(self.training), bool(kwargs.get("relation_source_estimate")))
            )
            return real_forward(self, *args, **kwargs)

        torch.manual_seed(7)
        before = torch.get_rng_state()
        with mock.patch.object(type(model), "forward", spy):
            for mask in (
                torch.tensor([True, True, True, True]),
                torch.tensor([True, False, True, False]),
                torch.zeros(4, dtype=torch.bool),
            ):
                torch.set_rng_state(before)
                with mock.patch.object(
                    hoi_trainer, "_d2ag_selection_mask", return_value=mask,
                ):
                    hoi_trainer._d2ag_relation_source_arguments(
                        model, batch, values["current"],
                        torch.zeros(4, dtype=torch.long), arguments, cfg,
                        processed_windows=0, rank=0,
                    )
                with self.subTest(selected=int(mask.sum())):
                    # Global RNG is untouched regardless of mask.sum().
                    self.assertTrue(torch.equal(torch.get_rng_state(), before))
                    self.assertTrue(model.training)
        self.assertTrue(observed)
        for training_flag, estimate_flag in observed:
            self.assertFalse(training_flag)
            self.assertTrue(estimate_flag)
        self.assertFalse(model.network.sparse_relation_field.relation_source_estimate)

    def test_global_rng_draws_are_independent_of_the_mask(self):
        values, arguments, batch = training_batch(4)
        cfg = d2ag_cfg()
        model = build_variant(HOI_ARCHITECTURE_D2AG)
        model.train()
        diffusion = GaussianDiffusion()
        fake_losses = {key: torch.tensor(0.0) for key in LOSS_KEYS}
        captured = []

        real_randint = torch.randint
        real_randn = torch.randn

        def run(mask):
            torch.manual_seed(1234)
            draws = {}

            def randint(*args, **kwargs):
                value = real_randint(*args, **kwargs)
                draws.setdefault("timesteps", value.clone())
                return value

            def randn(*args, **kwargs):
                value = real_randn(*args, **kwargs)
                draws.setdefault("noise", value.clone())
                return value

            with mock.patch.object(
                hoi_trainer, "_d2ag_selection_mask", return_value=mask,
            ), mock.patch.object(
                hoi_trainer, "hoi_training_losses", return_value=fake_losses,
            ), mock.patch.object(hoi_trainer.torch, "randint", randint), \
                    mock.patch.object(hoi_trainer.torch, "randn", randn):
                _forward_losses(
                    model, diffusion, batch, torch.zeros(24, dtype=torch.long),
                    values["position_minimum"], values["position_maximum"],
                    values["object_minimum"], values["object_maximum"], cfg,
                    processed_windows=0, rank=0,
                )
            captured.append((draws, torch.get_rng_state().clone()))

        run(torch.ones(4, dtype=torch.bool))
        run(torch.tensor([True, False, False, True]))
        run(torch.zeros(4, dtype=torch.bool))
        reference = captured[0]
        for index, (draws, state) in enumerate(captured[1:], start=1):
            with self.subTest(case=index):
                self.assertTrue(torch.equal(
                    draws["timesteps"], reference[0]["timesteps"],
                ))
                self.assertTrue(torch.equal(draws["noise"], reference[0]["noise"]))
                self.assertTrue(torch.equal(state, reference[1]))

    def test_mask_generator_is_independent_and_seed_formula_is_registered(self):
        cfg = d2ag_cfg()
        self.assertEqual(_d2ag_mask_seed(cfg, 0, 0), 42 * 1_000_003)
        self.assertEqual(_d2ag_mask_seed(cfg, 2048, 3), 42 * 1_000_003 + 2048 + 3)
        source = inspect.getsource(hoi_trainer._d2ag_selection_mask)
        self.assertIn("torch.Generator(device=device)", source)
        self.assertIn("generator=generator", source)
        first = hoi_trainer._d2ag_selection_mask(
            cfg, 512, torch.device("cpu"), 0, 0,
        )
        repeat = hoi_trainer._d2ag_selection_mask(
            cfg, 512, torch.device("cpu"), 0, 0,
        )
        other_rank = hoi_trainer._d2ag_selection_mask(
            cfg, 512, torch.device("cpu"), 0, 1,
        )
        self.assertTrue(torch.equal(first, repeat))
        self.assertFalse(torch.equal(first, other_rank))
        self.assertEqual(first.dtype, torch.bool)
        self.assertEqual(D2AG_SELF_CONDITION_PROBABILITY, 0.5)
        self.assertAlmostEqual(float(first.float().mean()), 0.5, delta=0.1)


class D2AGCheckpointAndConfigTests(unittest.TestCase):
    def _checkpoint(self, variant: str):
        model = build_variant(variant)
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
        contract = model.network.sparse_relation_field.contract_metadata()
        if variant == HOI_ARCHITECTURE_D2AE:
            value["sparse_relation_contract"] = contract
        elif variant == HOI_ARCHITECTURE_D2AF:
            value["diffusion_reliability_contract"] = contract
        else:
            value["selfcond_relation_source_contract"] = contract
        return value

    def test_independent_contract_and_cross_variant_rejection(self):
        contract = selfcond_relation_source_contract_metadata()
        self.assertEqual(contract["architecture_variant"], HOI_ARCHITECTURE_D2AG)
        self.assertIs(contract["sqrt_alpha_bar_attenuation"], False)
        self.assertIs(contract["relation_zero_branch"], False)
        self.assertIs(contract["schedule_buffer_registered"], False)
        self.assertNotIn("schedule", contract)
        self.assertEqual(contract["selection_probability"], 0.5)
        self.assertEqual(
            validate_selfcond_relation_source_contract(contract), contract,
        )
        for mutation in ({"selection_probability": 0.25}, {"relation_zero_branch": True}):
            broken = dict(contract)
            broken.update(mutation)
            with self.subTest(mutation=tuple(mutation)):
                with self.assertRaises(ValueError):
                    validate_selfcond_relation_source_contract(broken)
        missing = dict(contract)
        missing.pop("variable_anchor_source")
        with self.assertRaises(ValueError):
            validate_selfcond_relation_source_contract(missing)
        extra = dict(contract)
        extra["rho"] = 1.0
        with self.assertRaises(ValueError):
            validate_selfcond_relation_source_contract(extra)
        with self.assertRaises(ValueError):
            validate_selfcond_relation_source_contract(None)
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for variant in (
                HOI_ARCHITECTURE_D2AE, HOI_ARCHITECTURE_D2AF, HOI_ARCHITECTURE_D2AG,
            ):
                path = Path(directory) / f"{variant}.pth"
                torch.save(self._checkpoint(variant), path)
                paths[variant] = path
            for stored, expected in (
                (HOI_ARCHITECTURE_D2AG, HOI_ARCHITECTURE_D2AE),
                (HOI_ARCHITECTURE_D2AG, HOI_ARCHITECTURE_D2AF),
                (HOI_ARCHITECTURE_D2AE, HOI_ARCHITECTURE_D2AG),
                (HOI_ARCHITECTURE_D2AF, HOI_ARCHITECTURE_D2AG),
            ):
                with self.subTest(stored=stored, expected=expected):
                    with self.assertRaisesRegex(
                        ValueError, "architecture variant mismatch",
                    ):
                        load_trained_hoi_prior(
                            str(paths[stored]),
                            torch.device("cpu"),
                            use_ema=False,
                            expected_architecture_variant=expected,
                        )
            # Loading D2-AG succeeds only with its own complete contract.
            model, metadata = load_trained_hoi_prior(
                str(paths[HOI_ARCHITECTURE_D2AG]),
                torch.device("cpu"),
                use_ema=False,
                weight_variant="online",
                expected_architecture_variant=HOI_ARCHITECTURE_D2AG,
            )
            self.assertEqual(
                metadata["architecture_variant"], HOI_ARCHITECTURE_D2AG,
            )
            self.assertTrue(model.network.sparse_relation_field.selfcond_relation_source)

    def test_checkpoint_schema_fails_closed_on_each_locked_key(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in (
                ("missing_contract", lambda v: v.pop("selfcond_relation_source_contract")),
                (
                    "predecessor_contract",
                    lambda v: v.__setitem__(
                        "sparse_relation_contract",
                        build_variant(HOI_ARCHITECTURE_D2AE)
                        .network.sparse_relation_field.contract_metadata(),
                    ),
                ),
                (
                    "checkpoint_variant",
                    lambda v: v.__setitem__(
                        "architecture_variant", HOI_ARCHITECTURE_D2AE,
                    ),
                ),
                (
                    "model_config_trunk",
                    lambda v: v["model_config"].__setitem__("num_heads", 8),
                ),
                (
                    "initialization",
                    lambda v: v.__setitem__("initialization", "released"),
                ),
            ):
                value = self._checkpoint(HOI_ARCHITECTURE_D2AG)
                mutate(value)
                path = Path(directory) / f"{name}.pth"
                torch.save(value, path)
                with self.subTest(mutation=name):
                    with self.assertRaises(ValueError):
                        load_trained_hoi_prior(
                            str(path),
                            torch.device("cpu"),
                            use_ema=False,
                            expected_architecture_variant=HOI_ARCHITECTURE_D2AG,
                        )

    def test_resolved_config_is_single_factor(self):
        base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
        ae = OmegaConf.merge(
            base,
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ae.yaml"),
        )
        af = OmegaConf.merge(
            base,
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2af.yaml"),
        )
        ag = OmegaConf.merge(
            base,
            OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ag.yaml"),
        )
        self.assertIsNone(ag.run_id)
        self.assertFalse(bool(ag.d2af_sqrt_alpha_bar_reliability))
        self.assertFalse(bool(ag.d2ae_sparse_relation_field))
        self.assertTrue(bool(ag.d2ag_selfcond_relation_source))
        self.assertEqual(ag.hoi_architecture_variant, HOI_ARCHITECTURE_D2AG)
        self.assertEqual(ag.mode, "d2ag-selfcond-relation-source")
        self.assertEqual(ag.subphase, "1B-D2-AG0")
        for absent in (
            "d2af_clean_signal_eligibility_path",
            "d2af_clean_signal_eligibility_sha256",
            "d2af_performance_benchmark_path",
            "d2af_performance_benchmark_sha256",
            "d2af_performance_waiver_path",
            "d2af_performance_waiver_sha256",
            "d2af_checkpoint_race_continuation_path",
            "d2af_checkpoint_race_continuation_sha256",
            "d2ae_performance_benchmark_path",
            "d2ae_performance_benchmark_sha256",
        ):
            self.assertIsNone(ag.get(absent), absent)
        ignored = {
            "mode", "subphase", "run_id",
            "d2ae_sparse_relation_field", "d2af_sqrt_alpha_bar_reliability",
            "d2ag_selfcond_relation_source", "hoi_architecture_variant",
            "d2ae_performance_benchmark_path",
            "d2ae_performance_benchmark_sha256",
            "d2af_clean_signal_eligibility_path",
            "d2af_clean_signal_eligibility_sha256",
            "d2af_performance_benchmark_path",
            "d2af_performance_benchmark_sha256",
            "d2af_performance_waiver_path",
            "d2af_performance_waiver_sha256",
            "d2af_checkpoint_race_continuation_path",
            "d2af_checkpoint_race_continuation_sha256",
            "d2ag_performance_benchmark_path",
            "d2ag_performance_benchmark_sha256",
        }
        ae_values = OmegaConf.to_container(ae, resolve=False)
        af_values = OmegaConf.to_container(af, resolve=False)
        ag_values = OmegaConf.to_container(ag, resolve=False)
        for key in set(ae_values) | set(af_values) | set(ag_values):
            if key not in ignored:
                self.assertEqual(ae_values.get(key), ag_values.get(key), key)
                self.assertEqual(af_values.get(key), ag_values.get(key), key)
        actual_date = __import__("datetime").datetime.now().astimezone().strftime(
            "%Y%m%d"
        )
        ag.run_id = f"p1-hoi-d2ag-selfcond-relation-source-s42-{actual_date}"
        ag.repo_root = str(ROOT)
        ag.split_manifest = str(
            ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )
        _validate_fk_foot_temporal_routing_mode(ag)
        contract = _validate_d2ag_contract(ag, 4, require_performance_gate=False)
        self.assertFalse(contract["performance_gate_required"])
        self.assertIsNone(contract["performance_gate"])
        self.assertEqual(contract["selection_probability"], 0.5)

    def test_independent_throughput_constants_and_run_id_stem(self):
        self.assertAlmostEqual(D2AG_MINIMUM_THROUGHPUT, 2756.580356467847)
        self.assertAlmostEqual(D2AG_MAXIMUM_ETA_HOURS, 6.20)
        self.assertNotAlmostEqual(D2AG_MINIMUM_THROUGHPUT, D2AF_MINIMUM_THROUGHPUT)
        self.assertNotAlmostEqual(D2AG_MAXIMUM_ETA_HOURS, D2AF_MAXIMUM_ETA_HOURS)
        source = (ROOT / "code/train_hoi_prior.py").read_text(encoding="utf-8")
        self.assertIn("D2AG_MINIMUM_THROUGHPUT = 2756.580356467847", source)
        self.assertIn("D2AG_MAXIMUM_ETA_HOURS = 6.20", source)
        self.assertNotIn("D2AG_MINIMUM_THROUGHPUT = D2A", source)
        self.assertNotIn("D2AG_MAXIMUM_ETA_HOURS = D2A", source)
        actual_date = __import__("datetime").datetime.now().astimezone().strftime(
            "%Y%m%d"
        )
        accepted = hoi_trainer._validate_d2ag_formal_run_id(
            f"p1-hoi-d2ag-selfcond-relation-source-s42-{actual_date}"
        )
        self.assertTrue(accepted["date_is_actual"])
        self.assertFalse(accepted["retry"])
        for rejected in (
            "p1-hoi-d2ag-selfcond-relation-source-s42-20260101",
            "p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-" + actual_date,
            "p1-hoi-d2ag-selfcond-relation-source-s43-" + actual_date,
        ):
            with self.subTest(run_id=rejected):
                with self.assertRaises(ValueError):
                    hoi_trainer._validate_d2ag_formal_run_id(rejected)

    def test_no_loss_weighting_schedule_or_forbidden_source(self):
        field_source = inspect.getsource(
            SparseCurrentStateRelationField.forward
        ).lower()
        builder_source = inspect.getsource(build_d2ag_relation_source).lower()
        trainer_source = (ROOT / "code/train_hoi_prior.py").read_text(
            encoding="utf-8"
        ).lower()
        for forbidden in (
            "x_start", "future_gt", "contact_label", "scene",
            "snr_weight", "timestep_weight", "per_anchor",
        ):
            self.assertNotIn(forbidden, field_source)
            self.assertNotIn(forbidden, builder_source)
        for forbidden in (
            "d2ag_timestep_loss_weight", "d2ag_snr_weight",
            "d2ag_selection_probability_sweep", "d2ag_sqrt_alpha_bar",
        ):
            self.assertNotIn(forbidden, trainer_source)
        # The D2-AF waiver must not be reachable from the D2-AG contract path:
        # its config keys may appear only in the forbidden-inputs-absent list.
        # D2-AG's own one-time waiver is permitted.
        d2ag_contract_source = inspect.getsource(_validate_d2ag_contract).lower()
        for forbidden in (
            "_validate_d2af_performance_waiver",
            "d2af_waived",
            "d2af_performance_waiver_classification",
        ):
            self.assertNotIn(forbidden, d2ag_contract_source)
        residual = d2ag_contract_source
        for allowed in (
            "d2af_performance_waiver_path",
            "d2af_performance_waiver_sha256",
            "d2af_waiver_inherited",
            "d2ag_performance_waiver_path",
            "d2ag_performance_waiver_sha256",
            "performance_waiver_path",
            "performance_waiver_sha256",
            "performance_waiver_run_id",
            'performance_gate.get("performance_waiver")',
            "waiver is none",
            "waiver = (",
            'str(waiver["path"])',
            'str(waiver["sha256"])',
            'str(waiver["run_id"])',
        ):
            residual = residual.replace(allowed, "")
        self.assertNotIn("waiver", residual)
        self.assertIn("d2af_waiver_inherited", d2ag_contract_source)


class D2AGPerformanceWaiverTests(unittest.TestCase):
    """The one-time waiver may excuse only the measured throughput/ETA gap.

    Every fixture is built in a temporary directory: these tests must not
    depend on the real ``experiments/contracts/`` artifact existing yet.
    """

    FAILED_CHECKS = [
        "classification", "eta", "formal_authorized", "status", "throughput",
    ]

    def _benchmark(self, **overrides) -> dict:
        record = {
            "run_id": hoi_trainer.D2AG_WAIVED_BENCHMARK_RUN_ID,
            "status": "failed",
            "classification": (
                hoi_trainer.D2AG_PERFORMANCE_FAILURE_CLASSIFICATION
            ),
            "throughput_windows_per_second": (
                hoi_trainer.D2AG_WAIVED_THROUGHPUT
            ),
            "full_budget_eta_hours": hoi_trainer.D2AG_WAIVED_ETA_HOURS,
            "minimum_throughput_windows_per_second": D2AG_MINIMUM_THROUGHPUT,
            "maximum_full_budget_eta_hours": D2AG_MAXIMUM_ETA_HOURS,
            "formal_training_authorized": False,
            "failed_checks": list(self.FAILED_CHECKS),
            "non_speed_contracts_passed": True,
        }
        record.update(overrides)
        return record

    def _waiver(self, **overrides) -> dict:
        record = {
            "schema_version": 1,
            "status": "authorized",
            "classification": (
                hoi_trainer.D2AG_PERFORMANCE_WAIVER_CLASSIFICATION
            ),
            "run_id": "p1-hoi-d2ag-performance-waiver-s42-20260731",
            "formal_run_id": hoi_trainer.D2AG_WAIVED_FORMAL_RUN_ID,
            "seed": 42,
            "benchmark": {
                "run_id": hoi_trainer.D2AG_WAIVED_BENCHMARK_RUN_ID,
                "sha256": hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256,
                "status": "failed",
                "classification": (
                    hoi_trainer.D2AG_PERFORMANCE_FAILURE_CLASSIFICATION
                ),
                "throughput_windows_per_second": (
                    hoi_trainer.D2AG_WAIVED_THROUGHPUT
                ),
                "full_budget_eta_hours": hoi_trainer.D2AG_WAIVED_ETA_HOURS,
                "minimum_throughput_windows_per_second": (
                    D2AG_MINIMUM_THROUGHPUT
                ),
                "maximum_full_budget_eta_hours": D2AG_MAXIMUM_ETA_HOURS,
                "formal_training_authorized": False,
                "failed_checks": list(self.FAILED_CHECKS),
                "non_speed_contracts_passed": True,
            },
        }
        record.update(overrides)
        return record

    def _write(self, directory: Path, name: str, payload: dict) -> tuple:
        path = directory / name
        text = json.dumps(payload, indent=2, sort_keys=True)
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return path, digest

    def _cfg(self, **overrides):
        values = {
            "run_id": hoi_trainer.D2AG_WAIVED_FORMAL_RUN_ID,
            "seed": 42,
            "profile_every_update": True,
            "resume_checkpoint": None,
            "d2ag_performance_waiver_path": None,
            "d2ag_performance_waiver_sha256": None,
        }
        values.update(overrides)
        return OmegaConf.create(values)

    def test_waiver_constants_never_claim_a_passing_gate(self):
        self.assertEqual(
            hoi_trainer.D2AG_PERFORMANCE_WAIVER_STATUS, "failed-waived"
        )
        self.assertEqual(
            hoi_trainer.D2AG_PERFORMANCE_WAIVER_CLASSIFICATION,
            "user-authorized-performance-waiver",
        )
        self.assertEqual(
            hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS, "performance-gate-passed"
        )
        # The waived thresholds must stay D2-AG's own, never D2-AF's.
        self.assertNotEqual(D2AG_MINIMUM_THROUGHPUT, D2AF_MINIMUM_THROUGHPUT)
        self.assertNotEqual(D2AG_MAXIMUM_ETA_HOURS, D2AF_MAXIMUM_ETA_HOURS)
        self.assertLess(
            hoi_trainer.D2AG_WAIVED_THROUGHPUT, D2AG_MINIMUM_THROUGHPUT
        )
        self.assertGreater(
            hoi_trainer.D2AG_WAIVED_ETA_HOURS, D2AG_MAXIMUM_ETA_HOURS
        )

    def test_absent_waiver_keys_are_declared_in_both_configs(self):
        for name in (
            "code/config/config_train_hoi_prior.yaml",
            "code/config/config_train_hoi_prior_d2ag.yaml",
        ):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("d2ag_performance_waiver_path", text)
            self.assertIn("d2ag_performance_waiver_sha256", text)

    def test_missing_waiver_is_rejected_for_a_failed_benchmark(self):
        cfg = self._cfg()
        with self.assertRaises(ValueError) as caught:
            hoi_trainer._validate_d2ag_performance_waiver(
                cfg,
                benchmark=self._benchmark(),
                benchmark_sha256=(
                    hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                ),
                repo=ROOT,
            )
        self.assertIn("explicit sealed waiver", str(caught.exception))

    def test_relative_or_absent_waiver_path_is_rejected(self):
        cfg = self._cfg(
            d2ag_performance_waiver_path="experiments/contracts/x.json",
            d2ag_performance_waiver_sha256="0" * 64,
        )
        with self.assertRaises(ValueError) as caught:
            hoi_trainer._validate_d2ag_performance_waiver(
                cfg,
                benchmark=self._benchmark(),
                benchmark_sha256=(
                    hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                ),
                repo=ROOT,
            )
        self.assertIn("absolute file", str(caught.exception))

    def test_waiver_sha256_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path, _ = self._write(directory, "w.json", self._waiver())
            cfg = self._cfg(
                d2ag_performance_waiver_path=str(path),
                d2ag_performance_waiver_sha256="1" * 64,
            )
            with self.assertRaises(ValueError) as caught:
                hoi_trainer._validate_d2ag_performance_waiver(
                    cfg,
                    benchmark=self._benchmark(),
                    benchmark_sha256=(
                        hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                    ),
                    repo=ROOT,
                )
            self.assertIn("SHA-256 mismatch", str(caught.exception))

    def test_untracked_waiver_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path, digest = self._write(directory, "w.json", self._waiver())
            cfg = self._cfg(
                d2ag_performance_waiver_path=str(path),
                d2ag_performance_waiver_sha256=digest,
            )
            with self.assertRaises(ValueError) as caught:
                hoi_trainer._validate_d2ag_performance_waiver(
                    cfg,
                    benchmark=self._benchmark(),
                    benchmark_sha256=(
                        hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                    ),
                    repo=ROOT,
                )
            self.assertIn("inside the repository", str(caught.exception))

    def test_in_repo_but_untracked_waiver_file_is_rejected(self):
        # A file that exists under experiments/contracts/ but is not committed
        # must not be accepted: the waiver has to be auditable from history.
        with tempfile.TemporaryDirectory(
            dir=str(ROOT / "experiments/contracts")
        ) as raw:
            directory = Path(raw)
            path, digest = self._write(directory, "w.json", self._waiver())
            cfg = self._cfg(
                d2ag_performance_waiver_path=str(path),
                d2ag_performance_waiver_sha256=digest,
            )
            with self.assertRaises(ValueError) as caught:
                hoi_trainer._validate_d2ag_performance_waiver(
                    cfg,
                    benchmark=self._benchmark(),
                    benchmark_sha256=(
                        hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                    ),
                    repo=ROOT,
                )
            self.assertIn("tracked by Git", str(caught.exception))

    def _authorization(self, **overrides) -> dict:
        record = {
            "user_authorized_after_full_failure_disclosure": True,
            "user_accepted_full_budget_eta_hours": (
                hoi_trainer.D2AG_WAIVED_ETA_HOURS
            ),
            "formal_runs_maximum": 1,
            "random_initialization": True,
            "benchmark_retry_authorized": False,
            "execution_sweep_authorized": False,
            "benchmark_reclassification_authorized": False,
            "benchmark_classification_unchanged": True,
            "history_rewritten": False,
            "training_conditions_unchanged": True,
            "runtime_status_label": (
                hoi_trainer.D2AG_PERFORMANCE_WAIVER_STATUS
            ),
            "forbidden_status_label": (
                hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS
            ),
            "d2af_waiver_inherited": False,
            "bound_benchmark_run_ids": [
                hoi_trainer.D2AG_WAIVED_BENCHMARK_RUN_ID
            ],
            "bound_formal_run_ids": [
                hoi_trainer.D2AG_WAIVED_FORMAL_RUN_ID
            ],
        }
        record.update(overrides)
        return record

    def _non_speed(self, **overrides) -> dict:
        record = {
            "all_rank_contract_pass": True,
            "memory_headroom_pass": True,
            "contention_pass": True,
            "losses_finite": True,
            "gradients_finite": True,
            "selfcond_estimate_forward_measured": True,
            "invalid_if_any_non_speed_contract_fails": True,
        }
        record.update(overrides)
        return record

    def _accepting_stubs(self, stack, waiver: dict, directory: Path):
        """Stub only the Git-history lookups, never the contract checks."""
        source_contract = {
            "algorithm": "git-ls-files-path-content-sha256-v1",
            "scopes": ["code"],
            "tracked_file_count": 92,
            "sha256": hoi_trainer.D2AG_WAIVED_SOURCE_CONTRACT_SHA256,
        }
        target_contract = dict(source_contract, sha256="b" * 64)
        transition = {
            "source_commit": hoi_trainer.D2AG_WAIVED_SOURCE_COMMIT,
            "target_commit": "c" * 40,
            "changed_paths": ["code/train_hoi_prior.py"],
            "diff_sha256": "d" * 64,
        }
        stack.enter_context(mock.patch.object(
            hoi_trainer, "_validate_tracked_d2ag_waiver_path",
            return_value=hoi_trainer.D2AG_PERFORMANCE_WAIVER_RELATIVE_PATH,
        ))
        stack.enter_context(mock.patch.object(
            hoi_trainer, "_d2ag_source_transition", return_value=transition,
        ))
        stack.enter_context(mock.patch.object(
            hoi_trainer, "_d2ag_formal_source_contract_at_commit",
            side_effect=lambda repo, commit: (
                dict(source_contract)
                if commit == hoi_trainer.D2AG_WAIVED_SOURCE_COMMIT
                else dict(target_contract)
            ),
        ))
        stack.enter_context(mock.patch.object(
            hoi_trainer, "_d2ag_formal_source_contract",
            return_value=dict(target_contract),
        ))
        waiver.setdefault("source_transition", {
            "source_commit": hoi_trainer.D2AG_WAIVED_SOURCE_COMMIT,
            "target_commit": transition["target_commit"],
            "changed_paths": list(transition["changed_paths"]),
            "diff_sha256": transition["diff_sha256"],
            "source_formal_contract": dict(source_contract),
            "target_formal_contract": dict(target_contract),
        })
        waiver.setdefault("authorization", self._authorization())
        waiver.setdefault("non_speed_contracts", self._non_speed())
        waiver.setdefault("preexisting_formal_artifacts", {
            "formal_output_directory_existed": False,
            "training_state_existed": False,
            "training_metrics_existed": False,
            "checkpoint_count": 0,
        })
        return directory

    def _validate(self, waiver: dict, *, benchmark=None, **cfg_overrides):
        with ExitStack() as stack:
            raw = stack.enter_context(tempfile.TemporaryDirectory())
            directory = Path(raw)
            self._accepting_stubs(stack, waiver, directory)
            path, digest = self._write(directory, "w.json", waiver)
            cfg = self._cfg(
                d2ag_performance_waiver_path=str(path),
                d2ag_performance_waiver_sha256=digest,
                checkpoint_dir=str(directory / "ckpt"),
                metrics_path=str(directory / "metrics.jsonl"),
                state_path=str(directory / "state.json"),
                **cfg_overrides,
            )
            record = self._benchmark() if benchmark is None else benchmark
            record.setdefault("identity", {
                "git_commit": hoi_trainer.D2AG_WAIVED_SOURCE_COMMIT
            })
            record.setdefault(
                "formal_source_contract",
                waiver["source_transition"]["source_formal_contract"],
            )
            return hoi_trainer._validate_d2ag_performance_waiver(
                cfg,
                benchmark=record,
                benchmark_sha256=(
                    hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                ),
                repo=ROOT,
            )

    def test_valid_waiver_is_accepted_and_reports_failed_waived(self):
        result = self._validate(self._waiver())
        self.assertEqual(result["status"], "failed-waived")
        self.assertEqual(
            result["classification"], "user-authorized-performance-waiver"
        )
        self.assertEqual(
            result["benchmark_run_id"],
            hoi_trainer.D2AG_WAIVED_BENCHMARK_RUN_ID,
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertNotIn(
            hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS,
            json.dumps(result, default=str),
        )

    def test_waiver_bound_to_a_different_benchmark_run_id_is_rejected(self):
        waiver = self._waiver()
        waiver["benchmark"]["run_id"] = (
            "p1-hoi-d2ag-performance-benchmark-r3-s42-20260731"
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver)
        self.assertIn("benchmark_binding", str(caught.exception))

    def test_waiver_bound_to_a_different_formal_run_id_is_rejected(self):
        waiver = self._waiver()
        waiver["formal_run_id"] = (
            "p1-hoi-d2ag-selfcond-relation-source-r2-s42-20260731"
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver)
        self.assertIn("formal_run_id", str(caught.exception))

    def test_waiver_claiming_a_passed_gate_is_rejected(self):
        for field, value in (
            ("status", hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS),
            ("classification", hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS),
        ):
            waiver = self._waiver()
            waiver[field] = value
            with self.assertRaises(ValueError) as caught:
                self._validate(waiver)
            self.assertIn("no_passed_claim", str(caught.exception))

    def test_waiver_reclassifying_the_benchmark_as_passed_is_rejected(self):
        waiver = self._waiver()
        waiver["benchmark"]["classification"] = (
            hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver)
        self.assertIn("no_passed_claim", str(caught.exception))

    def test_waiver_may_not_relax_the_registered_thresholds(self):
        waiver = self._waiver()
        waiver["benchmark"]["minimum_throughput_windows_per_second"] = (
            hoi_trainer.D2AG_WAIVED_THROUGHPUT
        )
        benchmark = self._benchmark(
            minimum_throughput_windows_per_second=(
                hoi_trainer.D2AG_WAIVED_THROUGHPUT
            ),
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver, benchmark=benchmark)
        self.assertIn("benchmark_binding", str(caught.exception))

    def test_waiver_may_not_authorize_formal_training_in_the_benchmark(self):
        waiver = self._waiver()
        waiver["benchmark"]["formal_training_authorized"] = True
        benchmark = self._benchmark(formal_training_authorized=True)
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver, benchmark=benchmark)
        self.assertIn("benchmark_binding", str(caught.exception))

    def test_waiver_may_not_excuse_a_non_speed_failure(self):
        waiver = self._waiver()
        waiver["benchmark"]["failed_checks"] = sorted(
            self.FAILED_CHECKS + ["memory_headroom"]
        )
        benchmark = self._benchmark(
            failed_checks=sorted(self.FAILED_CHECKS + ["memory_headroom"]),
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver, benchmark=benchmark)
        self.assertIn("benchmark_binding", str(caught.exception))

    def test_waiver_requires_every_non_speed_contract_to_pass(self):
        for field in (
            "all_rank_contract_pass", "memory_headroom_pass",
            "contention_pass", "losses_finite", "gradients_finite",
            "selfcond_estimate_forward_measured",
            "invalid_if_any_non_speed_contract_fails",
        ):
            with self.subTest(field=field):
                waiver = self._waiver(
                    non_speed_contracts=self._non_speed(**{field: False}),
                )
                with self.assertRaises(ValueError) as caught:
                    self._validate(waiver)
                self.assertIn("non_speed_contracts", str(caught.exception))

    def test_waiver_must_not_authorize_retries_sweeps_or_inheritance(self):
        for field in (
            "benchmark_retry_authorized", "execution_sweep_authorized",
            "benchmark_reclassification_authorized", "d2af_waiver_inherited",
            "history_rewritten",
        ):
            with self.subTest(field=field):
                waiver = self._waiver(
                    authorization=self._authorization(**{field: True}),
                )
                with self.assertRaises(ValueError) as caught:
                    self._validate(waiver)
                self.assertIn("authorization", str(caught.exception))

    def test_waiver_must_not_allow_more_than_one_formal_run(self):
        waiver = self._waiver(
            authorization=self._authorization(formal_runs_maximum=2),
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver)
        self.assertIn("authorization", str(caught.exception))

    def test_waiver_must_declare_the_failed_waived_runtime_label(self):
        waiver = self._waiver(
            authorization=self._authorization(
                runtime_status_label=(
                    hoi_trainer.D2AG_FORBIDDEN_WAIVED_STATUS
                ),
            ),
        )
        with self.assertRaises(ValueError) as caught:
            self._validate(waiver)
        message = str(caught.exception)
        self.assertTrue(
            "authorization" in message or "no_passed_claim" in message,
            message,
        )

    def test_waiver_rejects_out_of_scope_source_changes(self):
        waiver = self._waiver()
        with ExitStack() as stack:
            raw = stack.enter_context(tempfile.TemporaryDirectory())
            directory = Path(raw)
            self._accepting_stubs(stack, waiver, directory)
            # Re-stub the transition to report an out-of-scope changed path.
            transition = {
                "source_commit": hoi_trainer.D2AG_WAIVED_SOURCE_COMMIT,
                "target_commit": "c" * 40,
                "changed_paths": ["code/priors/diffusion.py"],
                "diff_sha256": "d" * 64,
            }
            stack.enter_context(mock.patch.object(
                hoi_trainer, "_d2ag_source_transition",
                return_value=transition,
            ))
            waiver["source_transition"]["changed_paths"] = list(
                transition["changed_paths"]
            )
            path, digest = self._write(directory, "w.json", waiver)
            cfg = self._cfg(
                d2ag_performance_waiver_path=str(path),
                d2ag_performance_waiver_sha256=digest,
                checkpoint_dir=str(directory / "ckpt"),
                metrics_path=str(directory / "metrics.jsonl"),
                state_path=str(directory / "state.json"),
            )
            benchmark = self._benchmark(
                identity={
                    "git_commit": hoi_trainer.D2AG_WAIVED_SOURCE_COMMIT
                },
                formal_source_contract=(
                    waiver["source_transition"]["source_formal_contract"]
                ),
            )
            with self.assertRaises(ValueError) as caught:
                hoi_trainer._validate_d2ag_performance_waiver(
                    cfg,
                    benchmark=benchmark,
                    benchmark_sha256=(
                        hoi_trainer.D2AG_WAIVED_BENCHMARK_SHA256
                    ),
                    repo=ROOT,
                )
            self.assertIn("transition", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
