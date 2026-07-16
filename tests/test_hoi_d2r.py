import inspect
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from datasets.utils import get_smpl_parents
from priors.diffusion import GaussianDiffusion
from priors.representation import REPRESENTATION
from priors.routed_guidance import (
    CHECKPOINT_SHA256,
    KINEMATIC_METRICS,
    PHASE_OFFSETS,
    PRIOR_ROLLOUT_OFFSETS,
    RUN_ID,
    SELECTION_SHA256,
    UPPER_ROTATION_JOINTS,
    VARIANTS,
    apply_routed_guidance_update,
    mechanism_gate,
    route_gradient,
    sample_routed_counterfactual,
    sampler_seed_label,
    select_routed_holdout,
    upper_rotation_mask,
)
from priors.data import PriorWindowDataset
from priors.window_codec import WindowFrame, WindowStateCodec
from tools.diagnose_hoi_d2r import (
    _ancestor_rotation_contract,
    state_displacement,
)
from tools.summarize_hoi_d2r import validate_identity


class _ZeroModel(torch.nn.Module):
    def forward(self, noisy, timesteps, text, bps, goals, progress):
        del timesteps, text, bps, goals, progress
        return torch.zeros_like(noisy)


class _TinyDiffusion:
    timesteps = 2

    def posterior_sample(
        self, noisy, clean, timesteps, noise, fixed_history,
    ):
        del clean, timesteps
        result = noisy + noise * 0.01
        result[:, :2] = fixed_history
        return result


def _codec():
    return WindowStateCodec(
        torch.tensor([-2.0, -2.0, -2.0]),
        torch.tensor([2.0, 2.0, 2.0]),
        torch.tensor([-2.0, -2.0, -2.0]),
        torch.tensor([2.0, 2.0, 2.0]),
        verify_bps=False,
    )


def _frame(batch=1):
    return WindowFrame(
        torch.zeros(batch, 3),
        torch.eye(3).repeat(batch, 1, 1),
        torch.eye(3).repeat(batch, 1, 1),
    )


def _rest_offsets(batch=1):
    value = torch.zeros(batch, 24, 3)
    for joint in range(1, 24):
        value[:, joint, 0] = 0.01 * joint
        value[:, joint, 1] = 0.02
    return value


class D2RSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest=(
                "experiments/splits/"
                "omomo_hoi_train_validation_seed42.json"
            ),
        )

    def test_selection_is_locked_deterministic_and_fresh(self):
        first = select_routed_holdout(self.dataset)
        second = select_routed_holdout(self.dataset)
        self.assertEqual(first["global_indices"], second["global_indices"])
        self.assertEqual(first["sha256"], SELECTION_SHA256)
        self.assertEqual(first["phase_offsets"], list(PHASE_OFFSETS))
        self.assertEqual(first["sequences"], 64)
        self.assertEqual(first["windows"], 192)
        self.assertTrue(
            set(PHASE_OFFSETS).isdisjoint(PRIOR_ROLLOUT_OFFSETS)
        )
        for triple in first["triples"]:
            pi = [
                int(
                    self.dataset.language["pi"][
                        int(self.dataset.indices[position])
                    ]
                )
                for position in triple
            ]
            self.assertEqual(pi, list(PHASE_OFFSETS))

    def test_sampler_rng_label_is_variant_independent(self):
        label = sampler_seed_label(3, 2)
        for variant in VARIANTS:
            self.assertNotIn(variant, label)


class D2RRoutingTests(unittest.TestCase):
    def test_upper_mask_and_parent_mapping_are_exact(self):
        mask = upper_rotation_mask()
        self.assertEqual(
            int(mask.sum()), len(UPPER_ROTATION_JOINTS) * 6,
        )
        rotation_start = REPRESENTATION.field(
            "joint_rotations_6d"
        ).start
        for joint in range(22):
            active = bool(mask[
                rotation_start + joint * 6:
                rotation_start + (joint + 1) * 6
            ].all())
            self.assertEqual(active, joint in UPPER_ROTATION_JOINTS)
        self.assertFalse(mask[:rotation_start].any())
        self.assertFalse(mask[216:].any())
        parents = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy(),
        ).long()
        self.assertTrue(_ancestor_rotation_contract(parents))

    def test_author_human_and_upper_projection_support(self):
        gradient = torch.ones(2, 16, 232)
        author, author_audit = route_gradient(gradient, "author_all")
        self.assertTrue(torch.equal(author[:, :2], torch.zeros_like(author[:, :2])))
        self.assertTrue(torch.equal(author[:, 2:], gradient[:, 2:]))
        self.assertEqual(
            author_audit["routing_formula_replay_max_abs"], 0.0,
        )
        human, _ = route_gradient(gradient, "human_only")
        self.assertTrue(torch.equal(
            human[:, 2:, 216:228],
            torch.zeros_like(human[:, 2:, 216:228]),
        ))
        self.assertTrue(torch.equal(
            human[:, 2:, :216],
            gradient[:, 2:, :216],
        ))
        upper, upper_audit = route_gradient(gradient, "upper_raw")
        mask = upper_rotation_mask()
        self.assertTrue(torch.equal(
            upper[:, 2:, mask], gradient[:, 2:, mask],
        ))
        self.assertTrue(torch.equal(
            upper[:, 2:, ~mask], torch.zeros_like(upper[:, 2:, ~mask]),
        ))
        self.assertEqual(upper_audit["masked_off_max_abs"], 0.0)

    def test_upper_norm_preserves_per_sample_mutable_l2(self):
        torch.manual_seed(42)
        gradient = torch.randn(3, 16, 232)
        routed, audit = route_gradient(gradient, "upper_norm")
        full_norm = torch.linalg.vector_norm(
            gradient[:, 2:].reshape(3, -1), dim=1,
        )
        routed_norm = torch.linalg.vector_norm(
            routed[:, 2:].reshape(3, -1), dim=1,
        )
        self.assertTrue(torch.allclose(
            routed_norm, full_norm, rtol=1e-5, atol=1e-6,
        ))
        self.assertLessEqual(
            audit["norm_replay_relative_error_max"], 1e-5,
        )
        self.assertFalse(
            audit["invalid_nonzero_full_zero_projection"]
        )

    def test_upper_norm_zero_and_invalid_contracts(self):
        zero = torch.zeros(1, 16, 232)
        routed, audit = route_gradient(zero, "upper_norm")
        self.assertTrue(torch.equal(routed, zero))
        self.assertEqual(audit["routing_scale"], [1.0])
        self.assertFalse(
            audit["invalid_nonzero_full_zero_projection"]
        )
        root_only = torch.zeros(1, 16, 232)
        root_only[:, 2:, :3] = 1.0
        _, invalid = route_gradient(root_only, "upper_norm")
        self.assertTrue(
            invalid["invalid_nonzero_full_zero_projection"]
        )

    def test_step_zero_omits_guidance_and_active_step_restores_history(self):
        batch = 1
        clean = torch.zeros(batch, 16, 232)
        clean[:, :, 84:216] = 0.1
        clean[:, :, 219] = 1.0
        clean[:, :, 223] = 1.0
        clean[:, :, 227] = 1.0
        clean[:, :, 228:230] = 1.0
        fixed = clean[:, :2].clone()
        posterior = torch.randn_like(clean)
        parents = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy(),
        ).long()
        skipped, audit = apply_routed_guidance_update(
            posterior,
            clean,
            fixed,
            reverse_step=0,
            variant="upper_norm",
            codec=_codec(),
            frame=_frame(batch),
            rest_human_offsets=_rest_offsets(batch),
            parents_24=parents,
            rest_vertices=torch.randn(batch, 8, 3),
        )
        self.assertIsNone(audit)
        self.assertTrue(torch.equal(skipped[:, :2], fixed))
        guided, audit = apply_routed_guidance_update(
            posterior,
            clean,
            fixed,
            reverse_step=1,
            variant="upper_norm",
            codec=_codec(),
            frame=_frame(batch),
            rest_human_offsets=_rest_offsets(batch),
            parents_24=parents,
            rest_vertices=torch.randn(batch, 8, 3),
        )
        self.assertTrue(audit["finite"])
        self.assertTrue(torch.equal(guided[:, :2], fixed))


class D2RSamplerAndMetricTests(unittest.TestCase):
    def test_paired_sampler_consumes_identical_rng_and_restores_history(self):
        batch = 1
        fixed = torch.zeros(batch, 2, 232)
        arguments = {
            "diffusion": _TinyDiffusion(),
            "model": _ZeroModel(),
            "fixed_history": fixed,
            "text_embedding": torch.zeros(batch, 768),
            "object_bps": torch.zeros(batch, 1024, 3),
            "goals": torch.zeros(batch, 9),
            "progress": torch.zeros(batch, 3),
            "codec": _codec(),
            "frame": _frame(batch),
            "rest_human_offsets": _rest_offsets(batch),
            "parents_24": torch.from_numpy(
                get_smpl_parents(use_joints24=True).copy(),
            ).long(),
            "rest_vertices": torch.randn(batch, 8, 3),
        }
        first_generator = torch.Generator().manual_seed(42)
        second_generator = torch.Generator().manual_seed(42)
        unguided, unguided_audit = sample_routed_counterfactual(
            **arguments,
            generator=first_generator,
            variant="unguided",
        )
        guided, guided_audit = sample_routed_counterfactual(
            **arguments,
            generator=second_generator,
            variant="upper_norm",
        )
        self.assertTrue(torch.equal(
            first_generator.get_state(), second_generator.get_state(),
        ))
        self.assertTrue(torch.equal(unguided[:, :2], fixed))
        self.assertTrue(torch.equal(guided[:, :2], fixed))
        self.assertEqual(unguided_audit["applied_steps"], 0)
        self.assertEqual(guided_audit["applied_steps"], 1)
        self.assertFalse(guided_audit["step_zero_guidance_applied"])

    def test_state_displacement_reports_all_five_fields(self):
        control = [
            torch.zeros(2, 16, 232) for _ in range(3)
        ]
        candidate = [
            torch.ones(2, 16, 232) for _ in range(3)
        ]
        result = state_displacement(
            control, candidate, ["a", "b"],
        )
        self.assertEqual(
            set(result["aggregate"]),
            {field.name for field in REPRESENTATION.fields},
        )
        self.assertEqual(len(result["per_sequence"]), 2)

    def test_sampler_source_has_no_future_gt_or_stored_bps(self):
        source = inspect.getsource(sample_routed_counterfactual)
        self.assertNotIn("future_gt", source)
        self.assertNotIn("target[", source)
        self.assertNotIn("stored_per_frame_bps", source)
        self.assertNotIn('batch["object_bps"]', source)
        self.assertIn("diffusion.posterior_sample(", source)


class D2RGateAndLifecycleTests(unittest.TestCase):
    @staticmethod
    def comparison(
        *,
        contact_lower=0.1,
        precision_lower=-0.01,
        run_ratio=1.6,
        kinematic_upper=1.05,
    ):
        contact = {
            metric: {"bootstrap_95_ci": [contact_lower, 0.2]}
            for metric in (
                "recall",
                "f1",
                "prediction_percent",
                "prediction_run_mean_frames",
            )
        }
        contact["precision"] = {
            "bootstrap_95_ci": [precision_lower, 0.1],
        }
        contact["prediction_run_mean_frames"][
            "candidate_over_control_mean_ratio"
        ] = run_ratio
        return {
            "contact": contact,
            "kinematics": {
                metric: {
                    "bootstrap_95_ci": [0.9, kinematic_upper],
                }
                for metric in KINEMATIC_METRICS
            },
        }

    def test_gate_positive_negative_and_contract_failure(self):
        positive = mechanism_gate(
            {"complete": True}, self.comparison(),
        )
        self.assertEqual(
            positive["classification"],
            "state-routed-guidance-positive-stop",
        )
        negative = mechanism_gate(
            {"complete": True},
            self.comparison(run_ratio=1.4),
        )
        self.assertEqual(
            negative["classification"],
            "state-routed-guidance-negative-stop",
        )
        failure = mechanism_gate(
            {"complete": False}, self.comparison(),
        )
        self.assertEqual(
            failure["classification"],
            "state-routed-guidance-contract-failure-stop",
        )
        for value in (positive, negative, failure):
            self.assertFalse(value["production_guidance_authorized"])
            self.assertFalse(value["training_started"])
            self.assertFalse(value["d2h1_started"])

    def test_plan_and_registry_lock_run_and_checkpoint(self):
        plan = (
            ROOT / "docs/EXPERIMENT_PLAN.md"
        ).read_text(encoding="utf-8")
        self.assertIn(RUN_ID, plan)
        self.assertIn(CHECKPOINT_SHA256, plan)
        self.assertIn("state-routed-guidance-positive-stop", plan)
        records = [
            json.loads(line)
            for line in (
                ROOT / "experiments/registry.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prereg = [
            record for record in records
            if record["experiment_id"]
            == "p1-hoi-d2r-state-routed-guidance-preregister-s42-20260717"
        ]
        self.assertEqual(len(prereg), 1)
        self.assertEqual(prereg[0]["config"]["run_id"], RUN_ID)
        self.assertEqual(
            prereg[0]["config"]["checkpoint"]["sha256"],
            CHECKPOINT_SHA256,
        )

    def test_summary_requires_exact_identity_and_completed_status(self):
        metrics = {
            "run_id": RUN_ID,
            "git_commit": "abc",
            "status": "completed",
        }
        manifest = {
            "experiment_id": RUN_ID,
            "git": {"commit": "abc"},
        }
        validate_identity(metrics, manifest)
        with self.assertRaises(ValueError):
            validate_identity(
                {**metrics, "status": "failed"}, manifest,
            )


if __name__ == "__main__":
    unittest.main()
