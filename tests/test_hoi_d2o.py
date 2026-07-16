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

from priors.contact_alignment import (
    DECOMPOSITION_PATHS,
    EXPECTED_CHECKPOINT_SHA256,
    PHASE_OFFSETS,
    PHYSICAL_THRESHOLDS_CM,
    RUN_ID,
    SELECTION_SHA256,
    SEMANTIC_THRESHOLDS,
    binary_counts,
    classification_gate,
    distance_decomposition,
    geometry_report,
    metrics_from_counts,
    sampler_seed_label,
    select_contact_holdout,
    semantic_geometry_report,
    semantic_report,
    unit_binary_report,
)
from priors.data import PriorWindowDataset
from priors.diffusion import GaussianDiffusion
from tools.diagnose_hoi_d2o import ground_truth_summary, reports_complete
from tools.summarize_hoi_d2o import validate_identity


class _ZeroModel(torch.nn.Module):
    def forward(self, noisy, timesteps, text, bps, goals, progress):
        del timesteps, text, bps, goals, progress
        return torch.zeros_like(noisy)


class D2OSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )

    def test_selection_is_locked_deterministic_and_window_disjoint(self):
        first = select_contact_holdout(self.dataset)
        second = select_contact_holdout(self.dataset)
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
            self.assertTrue(set(pi).isdisjoint({0, 42, 84}))

    def test_sampler_rng_label_is_checkpoint_independent(self):
        label = sampler_seed_label(2, 1)
        self.assertEqual(label, sampler_seed_label(2, 1))
        for model in EXPECTED_CHECKPOINT_SHA256:
            self.assertNotIn(model, label)


class D2OMetricTests(unittest.TestCase):
    def test_multi_category_ground_truth_summary_is_python38_compatible(self):
        records = []
        for category, semantic in (
            ("box", [[1.0, 0.0, 0.0, 0.0]]),
            ("chair", [[0.0, 1.0, 0.0, 0.0]]),
        ):
            records.append({
                "object_category": category,
                "per_frame": {
                    "gt_semantic_labels": semantic,
                    "gt_hand_object_distance_m": [[0.01, 0.10]],
                },
            })
        summary = ground_truth_summary(records)
        self.assertEqual(set(summary["by_object_category"]), {"box", "chair"})
        self.assertEqual(summary["frames"], 2)
        for value in summary["by_object_category"].values():
            self.assertEqual(value["by_object_category"], {})
            self.assertEqual(value["frames"], 1)

    def test_production_sampler_pairing_restores_history_and_remains_finite(self):
        diffusion = GaussianDiffusion(500)
        model = _ZeroModel()
        fixed = torch.randn(1, 2, 232)
        text = torch.zeros(1, 768)
        bps = torch.zeros(1, 1024, 3)
        goals = torch.zeros(1, 9)
        progress = torch.zeros(1, 3)
        first_generator = torch.Generator().manual_seed(42)
        second_generator = torch.Generator().manual_seed(42)
        first = diffusion.sample(
            model, fixed, text, bps, goals, progress, generator=first_generator,
        )
        second = diffusion.sample(
            model, fixed, text, bps, goals, progress, generator=second_generator,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:, :2], fixed))
        self.assertTrue(torch.isfinite(first).all())

    def test_binary_truth_table_reports_left_right_and_union(self):
        prediction = np.asarray([
            [True, False],
            [False, True],
            [True, True],
            [False, False],
        ])
        target = np.asarray([
            [True, False],
            [True, False],
            [False, True],
            [False, False],
        ])
        report = unit_binary_report(prediction, target)
        self.assertEqual(set(report), {"left_hand", "right_hand", "union"})
        self.assertEqual(report["left_hand"]["counts"], {
            "tp": 1, "fp": 1, "tn": 1, "fn": 1,
        })
        self.assertEqual(report["union"]["counts"], {
            "tp": 3, "fp": 0, "tn": 1, "fn": 0,
        })
        self.assertEqual(
            metrics_from_counts(binary_counts(prediction[:, 0], target[:, 0])),
            report["left_hand"],
        )

    def test_all_four_semantic_fields_and_all_thresholds_are_reported(self):
        target = np.asarray([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ])
        prediction = np.asarray([
            [0.9, 0.1, 0.8, 0.2],
            [0.2, 0.8, 0.1, 0.9],
        ])
        semantic = semantic_report(prediction, target)
        distance = np.asarray([[0.01, 0.2], [0.2, 0.01]])
        geometry = geometry_report(distance, distance)
        alignment = semantic_geometry_report(prediction, distance)
        value = {
            "semantic_vs_gt": semantic,
            "physical_geometry_vs_gt": geometry,
            "predicted_semantic_vs_predicted_geometry": alignment,
            "distance_decomposition_on_gt_5cm_contact": {
                name: {} for name in DECOMPOSITION_PATHS
            },
        }
        self.assertTrue(reports_complete(value))
        self.assertEqual(set(semantic["per_channel"]), {"0", "1", "2", "3"})
        self.assertEqual(
            set(semantic["thresholds"]),
            {f"{value:g}" for value in SEMANTIC_THRESHOLDS},
        )
        self.assertEqual(
            set(geometry["thresholds_cm"]),
            {f"{value:g}" for value in PHYSICAL_THRESHOLDS_CM},
        )

    def test_human_object_swap_decomposition_is_independent(self):
        generated_joints = torch.zeros(2, 28, 3)
        gt_joints = generated_joints.clone()
        generated_joints[:, 24, 0] = 1.0
        generated_joints[:, 26, 0] = 1.0
        gt_joints[:, 24, 0] = 2.0
        gt_joints[:, 26, 0] = 2.0
        generated_vertices = torch.zeros(2, 1, 3)
        gt_vertices = torch.zeros(2, 1, 3)
        generated_vertices[..., 0] = 0.5
        gt_vertices[..., 0] = 1.5
        value = distance_decomposition(
            generated_joints, gt_joints, generated_vertices, gt_vertices,
        )
        self.assertTrue(torch.allclose(
            value["generated_human_generated_object"],
            torch.full((2, 2), 0.5),
        ))
        self.assertTrue(torch.allclose(
            value["gt_human_generated_object"],
            torch.full((2, 2), 1.5),
        ))
        self.assertTrue(torch.allclose(
            value["generated_human_gt_object"],
            torch.full((2, 2), 0.5),
        ))
        self.assertTrue(torch.allclose(
            value["gt_human_gt_object"],
            torch.full((2, 2), 0.5),
        ))


class D2OGateTests(unittest.TestCase):
    @staticmethod
    def comparisons(semantic_lower=0.1, recall_lower=0.1):
        return {
            f"balanced_vs_{comparator}": {
                "comparator_minus_balanced_semantic_first_two_mse": {
                    "bootstrap_95_ci": [semantic_lower, 0.2],
                },
                "comparator_minus_balanced_physical_recall_5cm": {
                    "bootstrap_95_ci": [recall_lower, 0.2],
                },
            }
            for comparator in ("source", "current")
        }

    @staticmethod
    def gt_alignment(f1=0.9, recall=0.9):
        return {
            "5": {
                "union": {"f1": f1, "recall": recall},
            },
        }

    def test_gate_classifies_contract_label_decoupling_and_mixed(self):
        positive = classification_gate(
            {"finite": True}, self.gt_alignment(), self.comparisons(),
        )
        self.assertEqual(
            positive["classification"],
            "semantic-geometry-decoupling-positive-stop",
        )
        self.assertFalse(positive["training_authorized"])
        mismatch = classification_gate(
            {"finite": True}, self.gt_alignment(f1=0.7), self.comparisons(),
        )
        self.assertEqual(
            mismatch["classification"],
            "label-evaluator-contract-mismatch-stop",
        )
        mixed = classification_gate(
            {"finite": True},
            self.gt_alignment(),
            self.comparisons(semantic_lower=-0.1),
        )
        self.assertEqual(mixed["classification"], "mixed-contact-deficit-stop")
        failure = classification_gate(
            {"finite": False}, self.gt_alignment(), self.comparisons(),
        )
        self.assertEqual(
            failure["classification"],
            "contact-alignment-contract-failure-stop",
        )


class D2OContractAndLifecycleTests(unittest.TestCase):
    def test_production_rollout_recomputes_bps_and_has_no_future_gt_channel(self):
        source = inspect.getsource(
            __import__(
                "tools.diagnose_hoi_d2o",
                fromlist=["rollout_chunk"],
            ).rollout_chunk
        )
        self.assertIn("current_bps(", source)
        self.assertNotIn('batch["object_bps"]', source)
        sampler = (ROOT / "code/priors/diffusion.py").read_text(encoding="utf-8")
        self.assertNotIn("future_gt", sampler)
        self.assertNotIn("stored_per_frame_bps", sampler)

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
            == "p1-hoi-d2o-contact-alignment-r1-preregister-s42-20260716"
        )
        self.assertEqual(record["config"]["run_id"], RUN_ID)
        self.assertEqual(
            record["config"]["selection"]["global_window_indices_sha256"],
            SELECTION_SHA256,
        )
        self.assertFalse(record["config"]["training_started"])
        self.assertFalse(record["config"]["d2h1_started"])
        self.assertFalse(record["config"]["d2g_started"])

    def test_diagnostic_and_summary_are_directly_executable(self):
        for relative in (
            "tools/diagnose_hoi_d2o.py",
            "tools/summarize_hoi_d2o.py",
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
            validate_identity(metrics, {"run_id": RUN_ID, "git": {"commit": "commit"}})


if __name__ == "__main__":
    unittest.main()
