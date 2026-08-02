"""Phase 1B P2 inference-time contact guidance: CPU contract tests.

Preregistration: ``docs/EXPERIMENT_PLAN.md`` section "2026-08-01 Phase 1B
推理期接触引导 P2（协议对齐，用户批准）" and registry row
``p1-hoi-p2-inference-contact-guidance-preregister-s42-20260801``.

The load-bearing test is :meth:`GuidanceOffIsBitIdenticalTests
.test_disabled_sampling_is_bitwise_identical_to_the_preregistration_source`,
which replays the reverse loop from the pre-implementation commit and requires a
bitwise identical latent and an identical generator state.
"""

import ast
import importlib.util
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

import guidance_loss  # noqa: E402
from datasets.utils import get_smpl_parents  # noqa: E402
from guidance_loss import (  # noqa: E402
    apply_feet_floor_contact_guidance,
    apply_hand_object_interaction_guidance_loss,
)
from priors.contact_guidance import (  # noqa: E402
    author_hand_object_components,
    decoded_fk_positions,
    deterministic_vertex_subset,
)
from priors.diffusion import GaussianDiffusion, HOIPriorSampler  # noqa: E402
from priors.inference_guidance import (  # noqa: E402
    ARM_A,
    ARM_B,
    CHOIS_CLASSIFIER_SCALE,
    DEFAULT_CLAMP,
    DEFAULT_LAST_STEPS,
    GUIDANCE_KEYS,
    GuidanceAudit,
    GuidanceSettings,
    HOIContactGuidance,
    author_full_hoi_loss,
    guidance_gradient,
)
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import WindowFrame, WindowStateCodec  # noqa: E402


# The last commit before the P2 implementation.  The bit-identity test replays
# its reverse loop, so it is pinned rather than resolved from HEAD.
PREREGISTRATION_COMMIT = "9a3a3510b53c9c21fe7b766264e98181230f1695"
SAMPLER_CONFIG = ROOT / "code/config/sampler/hoi_prior.yaml"


def _reference_diffusion_module():
    """Import ``code/priors/diffusion.py`` exactly as of the pinned commit."""
    source = subprocess.check_output(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:code/priors/diffusion.py"],
        cwd=ROOT,
        text=True,
    )
    if "guidance" in source:
        raise AssertionError(
            "the pinned reference already contains guidance; the bit-identity "
            "test would be vacuous",
        )
    name = "priors._p2_preregistration_reference_diffusion"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "priors"
    module.__file__ = str(ROOT / "code/priors/diffusion.py")
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class _DeterministicModel(torch.nn.Module):
    """A small non-trivial x0 predictor with fixed weights."""

    def __init__(self, seed: int = 7) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(
            REPRESENTATION.dimension, REPRESENTATION.dimension, generator=generator,
        ) / REPRESENTATION.dimension
        self.register_buffer("weight", weight)

    def forward(self, noisy, timesteps, text, bps, goals, progress):
        del text, bps, goals, progress
        scale = (timesteps.to(noisy.dtype) / 500.0).reshape(-1, 1, 1)
        return torch.tanh(noisy @ self.weight) * (1.0 - scale) + 0.1 * noisy


def _conditions(batch: int = 2):
    generator = torch.Generator().manual_seed(11)
    fixed_history = torch.randn(
        batch, REPRESENTATION.history_frames, REPRESENTATION.dimension,
        generator=generator,
    ) * 0.3
    return {
        "fixed_history": fixed_history,
        "text_embedding": torch.zeros(batch, 768),
        "object_bps": torch.zeros(batch, 1024, 3),
        "goals": torch.zeros(batch, 9),
        "progress": torch.zeros(batch, 3),
    }


def _codec():
    return WindowStateCodec(
        torch.tensor([-2.0, -2.0, -2.0]),
        torch.tensor([2.0, 2.0, 2.0]),
        torch.tensor([-2.0, -2.0, -2.0]),
        torch.tensor([2.0, 2.0, 2.0]),
        verify_bps=False,
    )


def _frame(batch: int, *, yaw: float = 0.7, origin_xz: float = 1.3):
    cosine, sine = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    rotation = torch.tensor([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])
    origin = torch.tensor([origin_xz, 0.0, -origin_xz]).expand(batch, 3).clone()
    return WindowFrame(
        origin,
        rotation.expand(batch, 3, 3).clone(),
        torch.eye(3).expand(batch, 3, 3).clone(),
    )


def _guidance_inputs(batch: int = 2, *, pelvis_normalized_height: float = 0.5):
    """A controlled state whose support foot sits well above the 0.02 m floor."""
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True).copy()).long()
    # Zero rest offsets collapse the skeleton onto the root, so the toe height
    # is exactly the decoded pelvis height and the scenario is analytic.
    rest_human_offsets = torch.zeros(batch, 24, 3)
    generator = torch.Generator().manual_seed(23)
    clean = torch.randn(
        batch, REPRESENTATION.window_frames, REPRESENTATION.dimension,
        generator=generator,
    ) * 0.05
    clean[..., 1] = pelvis_normalized_height
    clean[..., 216:219] = torch.tensor([0.3, 0.2, -0.4])
    clean[..., 219:228] = torch.eye(3).reshape(9)
    clean[..., 228:230] = 1.0
    clean[..., 230:232] = 0.0
    rest_vertices = torch.randn(batch, 48, 3, generator=generator) * 0.1
    return {
        "clean": clean,
        "codec": _codec(),
        "frame": _frame(batch),
        "rest_human_offsets": rest_human_offsets,
        "parents_24": parents,
        "rest_vertices": rest_vertices,
    }


class SignatureCompatibilityTests(unittest.TestCase):
    def test_new_parameters_are_defaulted_and_leave_positional_callers_intact(self):
        reference = _reference_diffusion_module()
        variadic = (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        )

        def named(signature):
            return [
                name for name, parameter in signature.parameters.items()
                if parameter.kind not in variadic
            ]

        for owner, name in (
            (GaussianDiffusion, "sample"),
            (HOIPriorSampler, "__init__"),
            (HOIPriorSampler, "p_sample_loop"),
        ):
            old = inspect.signature(getattr(getattr(reference, owner.__name__), name))
            new = inspect.signature(getattr(owner, name))
            old_names = named(old)
            new_names = named(new)
            self.assertEqual(
                new_names[:len(old_names)], old_names,
                f"{owner.__name__}.{name} reordered or dropped a parameter",
            )
            self.assertEqual(
                [p.kind for p in old.parameters.values() if p.kind in variadic],
                [p.kind for p in new.parameters.values() if p.kind in variadic],
                f"{owner.__name__}.{name} changed its variadic parameters",
            )
            for parameter in old_names:
                self.assertEqual(
                    new.parameters[parameter].default,
                    old.parameters[parameter].default,
                    f"{owner.__name__}.{name}:{parameter} default changed",
                )
                self.assertEqual(
                    new.parameters[parameter].kind,
                    old.parameters[parameter].kind,
                    f"{owner.__name__}.{name}:{parameter} kind changed",
                )
            for parameter in new_names[len(old_names):]:
                self.assertIsNot(
                    new.parameters[parameter].default,
                    inspect.Parameter.empty,
                    f"{owner.__name__}.{name}:{parameter} must be defaulted",
                )

    def test_p_sample_loop_still_accepts_the_d2ae_positional_call(self):
        source = (ROOT / "tools/diagnose_hoi_d2ae.py").read_text(encoding="utf-8")
        self.assertIn("sampler.p_sample_loop(", source)
        parameters = list(inspect.signature(HOIPriorSampler.p_sample_loop).parameters)
        # 20 positional arguments plus ``self``, exactly what the D2-AE tool passes.
        self.assertEqual(parameters[20], "seq_name_dict")
        self.assertEqual(
            inspect.signature(HOIPriorSampler.p_sample_loop).parameters[
                "object_only"
            ].default,
            False,
        )


class GuidanceOffIsBitIdenticalTests(unittest.TestCase):
    def test_disabled_sampling_is_bitwise_identical_to_the_preregistration_source(self):
        reference = _reference_diffusion_module()
        model = _DeterministicModel().eval()
        conditions = _conditions()

        reference_generator = torch.Generator().manual_seed(4242)
        with torch.no_grad():
            expected = reference.GaussianDiffusion(500).sample(
                model, generator=reference_generator, **conditions,
            )
        current_generator = torch.Generator().manual_seed(4242)
        with torch.no_grad():
            actual = GaussianDiffusion(500).sample(
                model, generator=current_generator, **conditions,
            )
        self.assertTrue(torch.equal(expected, actual), "guidance-off latent changed")
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(
            torch.equal(
                reference_generator.get_state(), current_generator.get_state(),
            ),
            "guidance-off sampling consumed a different number of RNG draws",
        )

        # ``guidance=None`` explicitly must also be bitwise identical.
        explicit_generator = torch.Generator().manual_seed(4242)
        with torch.no_grad():
            explicit = GaussianDiffusion(500).sample(
                model, generator=explicit_generator, guidance=None, **conditions,
            )
        self.assertTrue(torch.equal(expected, explicit))
        self.assertTrue(
            torch.equal(reference_generator.get_state(), explicit_generator.get_state())
        )

    def test_identity_test_would_fail_if_the_hook_perturbed_the_state(self):
        """The bit-identity check is discriminating, not vacuous."""
        model = _DeterministicModel().eval()
        conditions = _conditions()

        class _Perturbing:
            def apply(self, posterior, clean, fixed_history, reverse_step):
                del clean, fixed_history, reverse_step
                return posterior + 1e-7

        first = torch.Generator().manual_seed(4242)
        second = torch.Generator().manual_seed(4242)
        with torch.no_grad():
            baseline = GaussianDiffusion(500).sample(
                model, generator=first, **conditions,
            )
            perturbed = GaussianDiffusion(500).sample(
                model, generator=second, guidance=_Perturbing(), **conditions,
            )
        self.assertFalse(torch.equal(baseline, perturbed))

    def test_guidance_is_never_applied_on_the_final_reverse_step(self):
        model = _DeterministicModel().eval()
        conditions = _conditions()

        class _Spy:
            def __init__(self):
                self.steps = []

            def apply(self, posterior, clean, fixed_history, reverse_step):
                del clean, fixed_history
                self.steps.append(reverse_step)
                return posterior

        spy = _Spy()
        first = torch.Generator().manual_seed(99)
        second = torch.Generator().manual_seed(99)
        with torch.no_grad():
            guided = GaussianDiffusion(500).sample(
                model, generator=first, guidance=spy, **conditions,
            )
            plain = GaussianDiffusion(500).sample(
                model, generator=second, **conditions,
            )
        self.assertEqual(spy.steps, list(range(499, 0, -1)))
        self.assertNotIn(0, spy.steps)
        # A pass-through guidance object must not change the trajectory at all.
        self.assertTrue(torch.equal(guided, plain))


class AuthorFullLossTests(unittest.TestCase):
    def _tensors(self):
        inputs = _guidance_inputs()
        decoded = inputs["codec"].decode(inputs["clean"], inputs["frame"])
        fk = decoded_fk_positions(
            decoded, inputs["rest_human_offsets"], inputs["parents_24"],
        )
        vertices = (
            torch.einsum(
                "bvc,btdc->btvd", inputs["rest_vertices"], decoded["object_rotation"],
            )
            + decoded["object_translation"][:, :, None]
        )
        return inputs, decoded, fk, vertices

    def test_fk_joint_indices_match_the_author_toe_and_palm_convention(self):
        parents = get_smpl_parents(use_joints24=True).tolist()
        # apply_feet_floor_contact_guidance indexes 10/11 as the toes.
        self.assertEqual([parents[1], parents[4], parents[7], parents[10]], [0, 1, 4, 7])
        self.assertEqual([parents[2], parents[5], parents[8], parents[11]], [0, 2, 5, 8])
        # apply_hand_object_interaction_guidance_loss indexes 22/23 as the palms.
        self.assertEqual([parents[22], parents[23]], [20, 21])
        self.assertEqual(len(parents), 24)

    def test_decoded_state_preserves_absolute_floor_height(self):
        """The codec frame is yaw plus an XZ origin, so 0.02 m stays absolute."""
        inputs = _guidance_inputs()
        decoded = inputs["codec"].decode(inputs["clean"], inputs["frame"])
        local = inputs["codec"]._denormalize(
            inputs["clean"][..., :84].reshape(2, REPRESENTATION.window_frames, 28, 3),
            inputs["codec"].position_minimum,
            inputs["codec"].position_maximum,
        )
        self.assertTrue(
            torch.allclose(decoded["joints"][..., 1], local[..., 1], atol=1e-6)
        )
        self.assertEqual(float(inputs["frame"].origin[0, 1]), 0.0)

    def test_full_loss_is_the_author_weighted_sum_of_both_terms(self):
        _, decoded, fk, vertices = self._tensors()
        total = author_full_hoi_loss(
            fk, vertices, decoded["object_translation"],
            decoded["object_rotation"], decoded["contact"],
        )
        hand = apply_hand_object_interaction_guidance_loss(
            fk, vertices, decoded["object_translation"],
            decoded["object_rotation"], decoded["contact"],
        )
        feet = apply_feet_floor_contact_guidance(fk)
        self.assertGreater(float(feet), 0.0)
        self.assertGreater(float(hand), 0.0)
        self.assertTrue(torch.equal(total, hand * 10 + feet * 500))
        # The hand-object half stays byte-identical to the validated D2-Q0 path.
        components = author_hand_object_components(
            fk, vertices, decoded["object_translation"],
            decoded["object_rotation"], decoded["contact"],
        )
        self.assertTrue(torch.equal(hand, components["total"]))

    def test_feet_floor_term_is_live_under_mutation(self):
        inputs = _guidance_inputs()
        arguments = {
            key: inputs[key]
            for key in (
                "codec", "frame", "rest_human_offsets", "parents_24", "rest_vertices",
            )
        }
        gradient, loss, feet = guidance_gradient(inputs["clean"], **arguments)
        self.assertGreater(float(feet), 0.0)
        with mock.patch.object(
            guidance_loss,
            "apply_feet_floor_contact_guidance",
            lambda joints: joints.new_zeros(()),
        ):
            mutated_gradient, mutated_loss, _ = guidance_gradient(
                inputs["clean"], **arguments,
            )
        self.assertNotEqual(float(loss), float(mutated_loss))
        self.assertAlmostEqual(
            float(loss - mutated_loss), 500.0 * float(feet), places=2,
        )
        self.assertFalse(torch.equal(gradient, mutated_gradient))
        self.assertGreater(
            float((gradient - mutated_gradient).abs().max()), 0.0,
        )

    def test_loss_rejects_a_non_24_joint_tensor(self):
        _, decoded, fk, vertices = self._tensors()
        with self.assertRaises(ValueError):
            author_full_hoi_loss(
                fk[:, :, :22], vertices, decoded["object_translation"],
                decoded["object_rotation"], decoded["contact"],
            )

    def test_gradient_is_finite_and_nonzero(self):
        inputs = _guidance_inputs()
        gradient, loss, feet = guidance_gradient(
            inputs["clean"],
            codec=inputs["codec"],
            frame=inputs["frame"],
            rest_human_offsets=inputs["rest_human_offsets"],
            parents_24=inputs["parents_24"],
            rest_vertices=inputs["rest_vertices"],
        )
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(feet)))
        self.assertGreater(float(gradient.abs().max()), 0.0)
        # The pelvis height channel carries the feet-floor pull.
        self.assertGreater(float(gradient[..., 1].abs().max()), 0.0)

    def test_author_temporal_term_is_undefined_when_a_palm_meets_the_object_origin(self):
        """Documented hazard inherited verbatim from the author's loss.

        ``code/guidance_loss.py:58-64`` normalises the palm-to-object-centre
        vector without an epsilon, so an exact coincidence yields 0/0.  The
        preregistration ports that loss unchanged, so the behaviour is recorded
        here instead of being papered over; the registered
        ``nonfinite_values == 0`` gate is what protects the run.
        """
        inputs = _guidance_inputs()
        clean = inputs["clean"].clone()
        clean[..., 216:219] = clean[..., 0:3]
        gradient, loss, _ = guidance_gradient(
            clean,
            codec=inputs["codec"],
            frame=inputs["frame"],
            rest_human_offsets=inputs["rest_human_offsets"],
            parents_24=inputs["parents_24"],
            rest_vertices=inputs["rest_vertices"],
        )
        self.assertFalse(bool(torch.isfinite(loss)))
        self.assertFalse(bool(torch.isfinite(gradient).all()))

    def test_gradient_does_not_consume_global_rng(self):
        inputs = _guidance_inputs()
        before = torch.random.get_rng_state()
        guidance_gradient(
            inputs["clean"],
            codec=inputs["codec"],
            frame=inputs["frame"],
            rest_human_offsets=inputs["rest_human_offsets"],
            parents_24=inputs["parents_24"],
            rest_vertices=inputs["rest_vertices"],
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_module_never_samples_a_random_vertex_subset(self):
        module = ast.parse(
            (ROOT / "code/priors/inference_guidance.py").read_text(encoding="utf-8")
        )
        forbidden = {
            "randperm", "rand", "randn", "rand_like", "randn_like",
            "randint", "multinomial", "manual_seed",
        }
        called = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Attribute):
                    called.add(function.attr)
                elif isinstance(function, ast.Name):
                    called.add(function.id)
        self.assertEqual(
            sorted(called & forbidden), [],
            "the guidance module must not consume RNG",
        )
        subset = deterministic_vertex_subset(torch.arange(9000 * 3).reshape(9000, 3).float())
        again = deterministic_vertex_subset(torch.arange(9000 * 3).reshape(9000, 3).float())
        self.assertEqual(subset.shape[0], 2048)
        self.assertTrue(torch.equal(subset, again))


class ArmBehaviourTests(unittest.TestCase):
    def _guidance(self, settings, batch: int = 2):
        inputs = _guidance_inputs(batch)
        diffusion = GaussianDiffusion(500)
        guidance = HOIContactGuidance(
            settings,
            posterior_variance=diffusion.posterior_variance,
            codec=inputs["codec"],
            frame=inputs["frame"],
            rest_human_offsets=inputs["rest_human_offsets"],
            parents_24=inputs["parents_24"],
            rest_vertices=inputs["rest_vertices"],
            audit=GuidanceAudit(),
        )
        posterior = torch.zeros_like(inputs["clean"])
        return guidance, posterior, inputs

    def test_arm_windows_follow_the_preregistration(self):
        arm_a = GuidanceSettings(enabled=True, arm=ARM_A)
        self.assertFalse(arm_a.applies_at(0))
        self.assertTrue(all(arm_a.applies_at(step) for step in (1, 10, 250, 499)))
        arm_b = GuidanceSettings(enabled=True, arm=ARM_B, last_steps=10)
        self.assertEqual(
            [step for step in range(500) if arm_b.applies_at(step)],
            list(range(1, 10)),
        )
        self.assertFalse(GuidanceSettings(enabled=False).applies_at(300))

    def test_arm_a_adds_the_raw_scaled_gradient(self):
        settings = GuidanceSettings(enabled=True, arm=ARM_A, guidance_scale=1.0)
        guidance, posterior, inputs = self._guidance(settings)
        result = guidance.apply(posterior, inputs["clean"], inputs["clean"][:, :2], 499)
        gradient, _, _ = guidance_gradient(
            inputs["clean"],
            codec=inputs["codec"],
            frame=inputs["frame"],
            rest_human_offsets=inputs["rest_human_offsets"],
            parents_24=inputs["parents_24"],
            rest_vertices=inputs["rest_vertices"],
        )
        expected = posterior + gradient
        expected[:, :REPRESENTATION.history_frames] = inputs["clean"][:, :2]
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.equal(
            result[:, :REPRESENTATION.history_frames], inputs["clean"][:, :2],
        ))

    def test_arm_b_scales_by_posterior_variance(self):
        settings = GuidanceSettings(
            enabled=True, arm=ARM_B, guidance_scale=1.0, last_steps=500, clamp=None,
        )
        guidance, posterior, inputs = self._guidance(settings)
        history = inputs["clean"][:, :2]
        first = guidance.apply(posterior, inputs["clean"], history, 9)
        second = guidance.apply(posterior, inputs["clean"], history, 5)
        variance = guidance.posterior_variance
        frames = slice(REPRESENTATION.history_frames, None)
        ratio = float(variance[9] / variance[5])
        self.assertGreater(ratio, 1.0)
        self.assertTrue(torch.allclose(
            first[:, frames], second[:, frames] * ratio, atol=1e-6, rtol=1e-4,
        ))
        # Variance scaling must genuinely shrink the raw Arm A update.
        raw = GuidanceSettings(enabled=True, arm=ARM_A)
        raw_guidance, _, _ = self._guidance(raw)
        raw_guidance.frame = guidance.frame
        unscaled = raw_guidance.apply(posterior, inputs["clean"], history, 9)
        self.assertLess(
            float(first[:, frames].abs().max()),
            float(unscaled[:, frames].abs().max()),
        )

    def test_arm_b_clamp_bounds_the_update(self):
        history_frames = REPRESENTATION.history_frames
        unclamped_settings = GuidanceSettings(
            enabled=True, arm=ARM_B, guidance_scale=1e6, last_steps=500, clamp=None,
        )
        guidance, posterior, inputs = self._guidance(unclamped_settings)
        history = inputs["clean"][:, :2]
        unclamped = guidance.apply(posterior, inputs["clean"], history, 9)
        self.assertGreater(float(unclamped[:, history_frames:].abs().max()), 1.0)

        clamped_settings = GuidanceSettings(
            enabled=True, arm=ARM_B, guidance_scale=1e6, last_steps=500, clamp=0.25,
        )
        clamped_guidance, _, _ = self._guidance(clamped_settings)
        clamped_guidance.frame = guidance.frame
        clamped = clamped_guidance.apply(posterior, inputs["clean"], history, 9)
        self.assertLessEqual(
            float(clamped[:, history_frames:].abs().max()), 0.25 + 1e-6,
        )
        self.assertGreater(float(clamped[:, history_frames:].abs().max()), 0.0)

    def test_arm_b_state_clamp_target_bounds_the_result(self):
        settings = GuidanceSettings(
            enabled=True,
            arm=ARM_B,
            guidance_scale=1e6,
            last_steps=500,
            clamp=0.5,
            clamp_target="state",
        )
        guidance, posterior, inputs = self._guidance(settings)
        posterior = posterior + 3.0
        history = inputs["clean"][:, :2]
        result = guidance.apply(posterior, inputs["clean"], history, 9)
        self.assertLessEqual(
            float(result[:, REPRESENTATION.history_frames:].abs().max()), 0.5 + 1e-6,
        )
        self.assertTrue(torch.equal(result[:, :REPRESENTATION.history_frames], history))

    def test_guidance_changes_the_sampled_trajectory_and_stays_finite(self):
        settings = GuidanceSettings(enabled=True, arm=ARM_B, last_steps=6, clamp=0.05)
        inputs = _guidance_inputs()
        model = _DeterministicModel().eval()
        conditions = _conditions()
        diffusion = GaussianDiffusion(500)
        guidance = HOIContactGuidance(
            settings,
            posterior_variance=diffusion.posterior_variance,
            codec=inputs["codec"],
            frame=inputs["frame"],
            rest_human_offsets=inputs["rest_human_offsets"],
            parents_24=inputs["parents_24"],
            rest_vertices=inputs["rest_vertices"],
            audit=GuidanceAudit(),
        )
        with torch.no_grad():
            guided = diffusion.sample(
                model, generator=torch.Generator().manual_seed(5), guidance=guidance,
                **conditions,
            )
            plain = diffusion.sample(
                model, generator=torch.Generator().manual_seed(5), **conditions,
            )
        self.assertTrue(bool(torch.isfinite(guided).all()))
        self.assertFalse(torch.equal(guided, plain))
        self.assertEqual(guidance.audit.applied_steps, 5)
        audit = guidance.audit.as_dict()
        self.assertEqual(audit["guidance_nonfinite_steps"], 0)
        self.assertGreater(audit["guidance_feet_loss_mean"], 0.0)
        self.assertLessEqual(audit["guidance_update_max_abs"], 0.05 + 1e-6)

    def test_same_configuration_twice_is_bitwise_identical(self):
        model = _DeterministicModel().eval()
        conditions = _conditions()
        outputs = []
        for _ in range(2):
            inputs = _guidance_inputs()
            diffusion = GaussianDiffusion(500)
            guidance = HOIContactGuidance(
                GuidanceSettings(enabled=True, arm=ARM_B, last_steps=8, clamp=0.05),
                posterior_variance=diffusion.posterior_variance,
                codec=inputs["codec"],
                frame=inputs["frame"],
                rest_human_offsets=inputs["rest_human_offsets"],
                parents_24=inputs["parents_24"],
                rest_vertices=inputs["rest_vertices"],
                audit=GuidanceAudit(),
            )
            generator = torch.Generator().manual_seed(77)
            with torch.no_grad():
                outputs.append((
                    diffusion.sample(
                        model, generator=generator, guidance=guidance, **conditions,
                    ),
                    generator.get_state(),
                ))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))


class _StubDataset:
    """The minimal evaluator-dataset contract the guided sampler consumes."""

    load_scene = False

    def __init__(self, names, offsets):
        self.scene_name = list(names)
        self.rest_human_offsets = offsets
        self.parents_24 = get_smpl_parents(use_joints24=True).copy()
        self.min_torch = torch.tensor([-2.0, -2.0, -2.0])
        self.max_torch = torch.tensor([2.0, 2.0, 2.0])
        # Deliberately different from the joint range, as in the real dataset,
        # so a zero state does not place the object exactly on the pelvis.
        self.obj_min_torch = torch.tensor([-3.0, -1.0, -3.0])
        self.obj_max_torch = torch.tensor([3.0, 3.0, 3.0])

    def normalize_torch(self, data, is_object=False):
        minimum = self.obj_min_torch if is_object else self.min_torch
        maximum = self.obj_max_torch if is_object else self.max_torch
        return -1.0 + 2.0 * (data - minimum) / (maximum - minimum)


def _plausible_rest_offsets(sequences: int) -> torch.Tensor:
    """A crude but non-degenerate SMPL-shaped rest skeleton."""
    offsets = torch.zeros(sequences, 24, 3)
    for joint in range(1, 24):
        offsets[:, joint, 1] = -0.15 if joint in (4, 5, 7, 8, 10, 11) else 0.1
        offsets[:, joint, 0] = 0.05 * ((joint % 3) - 1)
        offsets[:, joint, 2] = 0.02
    return offsets


def _p_sample_loop_arguments(batch: int, frame: WindowFrame):
    transform = torch.eye(4).expand(batch, 4, 4).clone()
    transform[:, :3, :3] = frame.world_to_local.transpose(-1, -2)
    transform[:, :3, 3] = frame.origin
    return (
        torch.zeros(batch, REPRESENTATION.history_frames, REPRESENTATION.dimension),
        transform,
        None,
        torch.zeros(batch, 768),
        torch.zeros(batch, 3),
        None,
        torch.zeros(batch, 3),
        None,
        None,
        torch.zeros(batch, dtype=torch.long),
        torch.full((batch,), 48, dtype=torch.long),
        torch.full((batch,), 96, dtype=torch.long),
        None,
        None,
        torch.ones(batch, dtype=torch.bool),
        torch.zeros(batch, 1, 1, 1024, 3),
        None,
        frame.object_reference.reshape(batch, 1, 3, 3),
    )


class SamplerWiringTests(unittest.TestCase):
    """The end-to-end ``p_sample_loop`` path, which the GPU run actually uses."""

    def _sampler(self, *, enabled: bool, names, offsets):
        sampler = HOIPriorSampler(
            "cpu",
            guidance={
                "enabled": enabled,
                "arm": ARM_B,
                "guidance_scale": 1.0,
                "last_steps": 4,
                "clamp": 0.05,
                "clamp_target": "update",
            },
        )
        sampler.set_dataset_and_model(_StubDataset(names, offsets), _DeterministicModel().eval())
        return sampler

    def test_frame_is_recovered_from_the_evaluator_transform(self):
        batch = 2
        frame = _frame(batch)
        offsets = _plausible_rest_offsets(3)
        sampler = self._sampler(
            enabled=True, names=["a_box_0", "b_box_1", "c_box_2"], offsets=offsets,
        )
        arguments = _p_sample_loop_arguments(batch, frame)
        guidance = sampler._build_guidance(
            arguments[1], arguments[17],
            {"box": torch.randn(4096, 3)},
            {0: "a_box_0", 1: "b_box_1"},
            batch,
        )
        self.assertTrue(torch.allclose(guidance.frame.origin, frame.origin, atol=1e-6))
        self.assertTrue(torch.allclose(
            guidance.frame.world_to_local, frame.world_to_local, atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            guidance.frame.object_reference, frame.object_reference, atol=1e-6,
        ))
        self.assertEqual(tuple(guidance.rest_vertices.shape), (batch, 2048, 3))

    def test_rest_offsets_are_selected_by_sequence_name(self):
        offsets = _plausible_rest_offsets(3)
        offsets[0, 5, 0] = 1.0
        offsets[1, 5, 0] = 2.0
        offsets[2, 5, 0] = 3.0
        sampler = self._sampler(
            enabled=True, names=["a_box_0", "b_box_1", "c_box_2"], offsets=offsets,
        )
        guidance = sampler._build_guidance(
            _p_sample_loop_arguments(2, _frame(2))[1],
            _frame(2).object_reference.reshape(2, 1, 3, 3),
            {"box": torch.randn(4096, 3)},
            {0: "c_box_2", 1: "a_box_0"},
            2,
        )
        self.assertTrue(torch.equal(
            guidance.rest_human_offsets[:, 5, 0], torch.tensor([3.0, 1.0]),
        ))

    def test_duplicate_sequence_names_with_conflicting_offsets_fail_closed(self):
        offsets = _plausible_rest_offsets(2)
        offsets[1, 5, 0] = 1.0
        with self.assertRaises(ValueError):
            self._sampler(enabled=True, names=["a_box_0", "a_box_0"], offsets=offsets)

    def test_guided_and_unguided_p_sample_loop_agree_except_for_guidance(self):
        batch = 1
        frame = _frame(batch)
        offsets = _plausible_rest_offsets(1)
        names = ["a_box_0"]
        arguments = _p_sample_loop_arguments(batch, frame)
        rest = {"box": torch.randn(4096, 3, generator=torch.Generator().manual_seed(3))}
        results = {}
        for enabled in (False, True):
            sampler = self._sampler(enabled=enabled, names=names, offsets=offsets)
            torch.manual_seed(1234)
            sample, _ = sampler.p_sample_loop(
                *arguments, rest, {0: "a_box_0"}, object_only=True,
            )
            results[enabled] = (sample[0], sampler)
        unguided, plain_sampler = results[False]
        guided, guided_sampler = results[True]
        self.assertTrue(bool(torch.isfinite(guided).all()))
        self.assertFalse(torch.equal(unguided, guided))
        self.assertIsNone(plain_sampler.guidance_settings)
        audit = guided_sampler.audit_dict()["inference_guidance"]
        self.assertEqual(audit["guidance_applied_steps"], 3)
        self.assertEqual(audit["guidance_sample_calls"], 1)
        self.assertEqual(audit["guidance_nonfinite_steps"], 0)
        self.assertGreater(audit["guidance_feet_loss_mean"], 0.0)
        self.assertNotIn("inference_guidance", plain_sampler.audit_dict())

    def test_disabled_sampler_p_sample_loop_is_bitwise_unchanged(self):
        batch = 1
        frame = _frame(batch)
        offsets = _plausible_rest_offsets(1)
        arguments = _p_sample_loop_arguments(batch, frame)
        rest = {"box": torch.randn(4096, 3, generator=torch.Generator().manual_seed(3))}
        outputs = []
        for guidance in (None, {"enabled": False}):
            sampler = HOIPriorSampler("cpu", guidance=guidance)
            sampler.set_dataset_and_model(
                _StubDataset(["a_box_0"], offsets), _DeterministicModel().eval(),
            )
            torch.manual_seed(1234)
            sample, _ = sampler.p_sample_loop(
                *arguments, rest, {0: "a_box_0"}, object_only=True,
            )
            outputs.append(sample[0])
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


class ConfigurationTests(unittest.TestCase):
    def test_sampler_config_ships_guidance_off(self):
        config = OmegaConf.load(SAMPLER_CONFIG)
        self.assertEqual(
            config.pelvis._target_, "priors.diffusion.HOIPriorSampler",
        )
        guidance = config.pelvis.guidance
        self.assertEqual(sorted(guidance.keys()), sorted(GUIDANCE_KEYS))
        self.assertFalse(bool(guidance.enabled))
        self.assertEqual(guidance.arm, ARM_A)
        self.assertEqual(float(guidance.guidance_scale), 1.0)
        self.assertEqual(int(guidance.last_steps), DEFAULT_LAST_STEPS)
        self.assertEqual(float(guidance.clamp), DEFAULT_CLAMP)
        self.assertEqual(guidance.clamp_target, "update")

    def test_sampler_leaves_guidance_disabled_for_the_shipped_config(self):
        config = OmegaConf.load(SAMPLER_CONFIG)
        sampler = HOIPriorSampler("cpu", guidance=config.pelvis.guidance)
        self.assertIsNone(sampler.guidance_settings)
        self.assertIsNone(sampler.guidance_audit)
        self.assertNotIn("inference_guidance", sampler.audit_dict())

    def test_enabled_configuration_is_reported_in_the_audit(self):
        sampler = HOIPriorSampler(
            "cpu", guidance={"enabled": True, "arm": ARM_B, "guidance_scale": CHOIS_CLASSIFIER_SCALE},
        )
        self.assertIsNotNone(sampler.guidance_settings)
        audit = sampler.audit_dict()["inference_guidance"]
        self.assertEqual(audit["arm"], ARM_B)
        self.assertEqual(audit["guidance_scale"], CHOIS_CLASSIFIER_SCALE)
        self.assertEqual(audit["loss"], "guidance_loss.apply_hoi_guidance_loss")
        self.assertEqual(audit["guidance_applied_steps"], 0)

    def test_unknown_and_invalid_guidance_keys_fail_closed(self):
        with self.assertRaises(ValueError):
            GuidanceSettings.from_config({"enabled": True, "armm": "a"})
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, arm="c")
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, last_steps=0)
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, clamp=0.0)
        with self.assertRaises(ValueError):
            GuidanceSettings(enabled=True, clamp_target="everything")
        self.assertFalse(GuidanceSettings.from_config(None).enabled)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
