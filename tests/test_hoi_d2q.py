import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from datasets.utils import get_smpl_parents
from guidance_loss import apply_hand_object_interaction_guidance_loss
from priors.contact_alignment import geometry_report, semantic_report
from priors.contact_guidance import (
    AUTHOR_BLOB_SHA256,
    AUTHOR_HAND_WEIGHT,
    DIRECT_HAND_INDICES,
    FK_PALM_INDICES,
    MODELS,
    PHASE_OFFSETS,
    RUN_ID,
    SELECTION_SHA256,
    VARIANTS,
    apply_guidance_update,
    author_hand_object_components,
    deterministic_vertex_subset,
    hand_distances,
    mechanism_gate,
    sample_contact_counterfactual,
    sampler_seed_label,
    select_guidance_holdout,
)
from priors.data import PriorWindowDataset
from priors.diffusion import GaussianDiffusion
from priors.window_codec import WindowFrame, WindowStateCodec
from tools.diagnose_hoi_d2q import reports_complete
from tools.summarize_hoi_d2q import validate_identity


class _ZeroModel(torch.nn.Module):
    def forward(self, noisy, timesteps, text, bps, goals, progress):
        del timesteps, text, bps, goals, progress
        return torch.zeros_like(noisy)


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


class D2QSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )

    def test_selection_is_locked_deterministic_and_fresh(self):
        first = select_guidance_holdout(self.dataset)
        second = select_guidance_holdout(self.dataset)
        self.assertEqual(first["global_indices"], second["global_indices"])
        self.assertEqual(first["sha256"], SELECTION_SHA256)
        self.assertEqual(first["phase_offsets"], list(PHASE_OFFSETS))
        self.assertEqual(first["sequences"], 64)
        self.assertEqual(first["windows"], 192)
        for triple in first["triples"]:
            pi = [
                int(self.dataset.language["pi"][int(self.dataset.indices[position])])
                for position in triple
            ]
            self.assertEqual(pi, list(PHASE_OFFSETS))
            self.assertTrue(set(pi).isdisjoint({0, 14, 42, 56, 84, 98}))

    def test_sampler_rng_label_is_model_and_variant_independent(self):
        label = sampler_seed_label(3, 2)
        for model in MODELS:
            self.assertNotIn(model, label)
        for variant in VARIANTS:
            self.assertNotIn(variant, label)


class D2QAuthorFormulaTests(unittest.TestCase):
    def test_author_hand_object_formula_replay(self):
        torch.manual_seed(42)
        human = torch.randn(2, 5, 24, 3)
        vertices = torch.randn(2, 5, 7, 3)
        translation = torch.randn(2, 5, 3)
        rotation = torch.eye(3).repeat(2, 5, 1, 1)
        contact = torch.rand(2, 5, 4)
        contact[:, :, :2] = torch.tensor([
            [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        ])
        expected = apply_hand_object_interaction_guidance_loss(
            human, vertices, translation, rotation, contact,
        )
        actual = author_hand_object_components(
            human, vertices, translation, rotation, contact,
        )
        self.assertTrue(torch.allclose(actual["total"], expected, atol=1e-6))

    def test_mask_com_and_rotation_are_detached(self):
        human = (torch.randn(1, 4, 24, 3) + 2.0).requires_grad_(True)
        vertices = torch.randn(1, 4, 6, 3)
        translation = torch.randn(1, 4, 3, requires_grad=True)
        rotation = torch.eye(3).repeat(1, 4, 1, 1).requires_grad_(True)
        contact = torch.ones(1, 4, 4, requires_grad=True)
        total = author_hand_object_components(
            human, vertices, translation, rotation, contact,
        )["total"]
        gradients = torch.autograd.grad(
            total,
            (human, translation, rotation, contact),
            allow_unused=True,
        )
        self.assertIsNotNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        self.assertIsNone(gradients[3])

    def test_vertex_subset_is_deterministic_and_exact_size(self):
        vertices = torch.arange(3000 * 3, dtype=torch.float32).reshape(3000, 3)
        first = deterministic_vertex_subset(vertices)
        second = deterministic_vertex_subset(vertices)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (2048, 3))

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
        rest_vertices = torch.randn(batch, 8, 3)
        skipped, audit = apply_guidance_update(
            posterior,
            clean,
            fixed,
            reverse_step=0,
            codec=_codec(),
            frame=_frame(batch),
            rest_human_offsets=_rest_offsets(batch),
            parents_24=parents,
            rest_vertices=rest_vertices,
        )
        self.assertIsNone(audit)
        self.assertTrue(torch.equal(skipped[:, :2], fixed))
        guided, audit = apply_guidance_update(
            posterior,
            clean,
            fixed,
            reverse_step=1,
            codec=_codec(),
            frame=_frame(batch),
            rest_human_offsets=_rest_offsets(batch),
            parents_24=parents,
            rest_vertices=rest_vertices,
        )
        self.assertTrue(audit["finite"])
        self.assertEqual(audit["author_hand_weight"], AUTHOR_HAND_WEIGHT)
        self.assertAlmostEqual(
            audit["loss"],
            AUTHOR_HAND_WEIGHT * audit["raw_hand_loss"],
            places=5,
        )
        self.assertTrue(torch.equal(guided[:, :2], fixed))


class D2QSamplerAndMetricTests(unittest.TestCase):
    def test_paired_sampler_consumes_identical_rng_and_restores_history(self):
        batch = 1
        fixed = torch.zeros(batch, 2, 232)
        text = torch.zeros(batch, 768)
        bps = torch.zeros(batch, 1024, 3)
        goals = torch.zeros(batch, 9)
        progress = torch.zeros(batch, 3)
        parents = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy(),
        ).long()
        arguments = {
            "diffusion": GaussianDiffusion(500),
            "model": _ZeroModel(),
            "fixed_history": fixed,
            "text_embedding": text,
            "object_bps": bps,
            "goals": goals,
            "progress": progress,
            "codec": _codec(),
            "frame": _frame(batch),
            "rest_human_offsets": _rest_offsets(batch),
            "parents_24": parents,
            "rest_vertices": torch.randn(batch, 8, 3),
        }
        first_generator = torch.Generator().manual_seed(42)
        second_generator = torch.Generator().manual_seed(42)
        unguided, unguided_audit = sample_contact_counterfactual(
            **arguments, generator=first_generator, guided=False,
        )
        guided, guided_audit = sample_contact_counterfactual(
            **arguments, generator=second_generator, guided=True,
        )
        self.assertTrue(torch.equal(
            first_generator.get_state(), second_generator.get_state(),
        ))
        self.assertTrue(torch.equal(unguided[:, :2], fixed))
        self.assertTrue(torch.equal(guided[:, :2], fixed))
        self.assertEqual(unguided_audit["applied_steps"], 0)
        self.assertEqual(guided_audit["applied_steps"], 499)
        self.assertFalse(guided_audit["step_zero_guidance_applied"])

    def test_fk_and_direct_hand_metrics_are_separate(self):
        fk = torch.zeros(2, 24, 3)
        direct = torch.zeros(2, 28, 3)
        vertices = torch.zeros(2, 1, 3)
        fk[:, FK_PALM_INDICES, 0] = 1.0
        direct[:, DIRECT_HAND_INDICES, 0] = 2.0
        self.assertTrue(torch.equal(
            hand_distances(fk, vertices, FK_PALM_INDICES),
            torch.ones(2, 2),
        ))
        self.assertTrue(torch.equal(
            hand_distances(direct, vertices, DIRECT_HAND_INDICES),
            torch.full((2, 2), 2.0),
        ))

    def test_all_contact_fields_and_physical_thresholds_are_reported(self):
        target_contact = np.asarray([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ])
        predicted_contact = np.asarray([
            [0.9, 0.1, 0.8, 0.2],
            [0.2, 0.8, 0.1, 0.9],
        ])
        distance = np.asarray([[0.01, 0.2], [0.2, 0.01]])
        value = {
            "semantic_vs_gt": semantic_report(
                predicted_contact, target_contact,
            ),
            "fk_physical_geometry_vs_gt": geometry_report(
                distance, distance,
            ),
            "direct_physical_geometry_vs_gt": geometry_report(
                distance, distance,
            ),
        }
        self.assertTrue(reports_complete(value))


class D2QGateAndLifecycleTests(unittest.TestCase):
    @staticmethod
    def comparisons(
        *,
        contact_lower=0.1,
        precision_lower=-0.01,
        run_ratio=1.6,
        kinematic_upper=1.05,
    ):
        contact = {
            metric: {
                "bootstrap_95_ci": [contact_lower, 0.2],
            }
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
            "guided_over_unguided_mean_ratio"
        ] = run_ratio
        return {
            "balanced": {
                "contact": {
                    "fk_physical_geometry_vs_gt": contact,
                },
                "kinematics": {
                    metric: {
                        "bootstrap_95_ci": [0.9, kinematic_upper],
                    }
                    for metric in (
                        "mpjpe_cm",
                        "object_goal_error_cm",
                        "pelvis_goal_error_cm",
                        "object_translation_mae_cm",
                        "foot_sliding",
                    )
                },
            },
        }

    def test_gate_positive_negative_and_contract_failure(self):
        positive = mechanism_gate(
            {"complete": True}, self.comparisons(),
        )
        self.assertEqual(
            positive["classification"],
            "author-contact-guidance-positive-stop",
        )
        negative = mechanism_gate(
            {"complete": True},
            self.comparisons(run_ratio=1.49),
        )
        self.assertEqual(
            negative["classification"],
            "author-contact-guidance-negative-stop",
        )
        failure = mechanism_gate(
            {"complete": False}, self.comparisons(),
        )
        self.assertEqual(
            failure["classification"],
            "author-contact-guidance-contract-failure-stop",
        )
        for decision in (positive, negative, failure):
            self.assertFalse(decision["production_guidance_authorized"])
            self.assertFalse(decision["training_authorized"])
            self.assertFalse(decision["training_started"])

    def test_custom_sampler_reuses_posterior_and_production_default_is_unchanged(self):
        source = inspect.getsource(sample_contact_counterfactual)
        self.assertIn("diffusion.posterior_sample(", source)
        production = inspect.getsource(GaussianDiffusion.sample)
        self.assertNotIn("guidance", production)
        self.assertNotIn("sample_contact_counterfactual", production)
        self.assertNotIn("future_gt", source)
        self.assertNotIn('batch["object_bps"]', source)

    def test_registry_and_stop_contract_are_complete(self):
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        record = next(
            value for value in records
            if value["experiment_id"]
            == "p1-hoi-d2q-author-contact-guidance-preregister-s42-20260716"
        )
        self.assertEqual(record["config"]["run_id"], RUN_ID)
        self.assertEqual(
            record["config"]["selection"]["global_window_indices_sha256"],
            SELECTION_SHA256,
        )
        self.assertEqual(
            set(record["config"]["sampling"]["author_blob_sha256"].values()),
            set(AUTHOR_BLOB_SHA256.values()),
        )
        clarification = next(
            value for value in records
            if value["experiment_id"]
            == "p1-hoi-d2q-author-contact-guidance-scale-clarification-s42-20260716"
        )
        self.assertEqual(
            clarification["config"]["sampling"]["author_outer_hand_weight"],
            AUTHOR_HAND_WEIGHT,
        )
        self.assertFalse(record["config"]["production_sampler_default_change"])
        self.assertFalse(record["config"]["training_started"])
        self.assertFalse(record["config"]["d2h1_started"])
        self.assertFalse(record["config"]["d2g_started"])

    def test_diagnostic_and_summary_are_directly_executable(self):
        for relative in (
            "tools/diagnose_hoi_d2q.py",
            "tools/summarize_hoi_d2q.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(ROOT / relative), "--help"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_summary_identity_uses_manifest_experiment_id_and_exact_commit(self):
        metrics = {"run_id": RUN_ID, "git_commit": "commit", "status": "completed"}
        manifest = {"experiment_id": RUN_ID, "git": {"commit": "commit"}}
        validate_identity(metrics, manifest)
        with self.assertRaisesRegex(ValueError, "run-id mismatch"):
            validate_identity(
                metrics,
                {"run_id": RUN_ID, "git": {"commit": "commit"}},
            )


if __name__ == "__main__":
    unittest.main()
