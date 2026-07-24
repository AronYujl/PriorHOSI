import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import run_chois_evaluator
from tools import run_hoi_d2aa_table5 as d2aa


class D2AATable5Test(unittest.TestCase):
    def test_candidate_scope_and_checkpoint_hashes_are_fixed(self):
        self.assertEqual(list(d2aa.CANDIDATES), ["d2v", "d2x", "d2y", "d2z"])
        self.assertEqual(
            [d2aa.CANDIDATES[name]["checkpoint_sha256"] for name in d2aa.CANDIDATES],
            [
                "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4",
                "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51",
                "8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7",
                "44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6",
            ],
        )

    def test_native_and_timing_overrides_change_only_registered_reporting_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output=Path(directory),
                device="cuda:0",
                python=Path("/verified/infbagel/python"),
            )
            native = set(d2aa.native_overrides(args, "d2x", False))
            timing = set(d2aa.native_overrides(args, "d2x", True))
        for overrides in (native, timing):
            self.assertIn("checkpoint_weight_variant=online", overrides)
            self.assertIn("load_scene=false", overrides)
            self.assertIn("sample_type=diffusion", overrides)
            self.assertIn("hoi_expected_sequences=438", overrides)
            self.assertIn("save_motion_params=false", overrides)
        self.assertIn("save_chois_eval_npz=true", native)
        self.assertIn("hoi_sequence_limit=null", native)
        self.assertIn("save_chois_eval_npz=false", timing)
        self.assertIn("hoi_sequence_limit=1", timing)

    def test_nested_comparison_uses_fixed_float_tolerance(self):
        passed = d2aa.compare_nested(
            {"value": 1.0, "nested": {"other": None}},
            {"value": 1.0 + 5e-10, "nested": {"other": None}},
        )
        failed = d2aa.compare_nested(
            {"value": 1.0},
            {"value": 1.0 + 5e-7},
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["mismatch_count"], 1)

    def test_additive_bootstrap_is_deterministic(self):
        matching = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        rows = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        first = run_chois_evaluator._bootstrap_mean_intervals(
            matching, rows, replicates=1000, seed=42,
        )
        second = run_chois_evaluator._bootstrap_mean_intervals(
            matching, rows, replicates=1000, seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 1000)
        self.assertEqual(first["seed"], 42)
        for metric in (
            "MatchingScore", "R-Precision@1", "R-Precision@2", "R-Precision@3",
        ):
            self.assertEqual(len(first[metric]["bootstrap_95_ci"]), 2)

    def test_paired_fid_bootstrap_is_deterministic_on_small_embeddings(self):
        truth = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        prediction = truth + np.asarray([0.1, -0.2])

        def statistics(values):
            return values.mean(axis=0), np.cov(values, rowvar=False)

        def simple_frechet(mean_a, covariance_a, mean_b, covariance_b):
            return float(
                np.square(mean_a - mean_b).sum()
                + np.square(covariance_a - covariance_b).sum()
            )

        first = run_chois_evaluator._bootstrap_fid_interval(
            truth,
            prediction,
            frechet=simple_frechet,
            activation_statistics=statistics,
            replicates=50,
            seed=42,
        )
        second = run_chois_evaluator._bootstrap_fid_interval(
            truth,
            prediction,
            frechet=simple_frechet,
            activation_statistics=statistics,
            replicates=50,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "paired_embedded_sequence")
        self.assertEqual(len(first["bootstrap_95_ci"]), 2)

    def test_table5_mapping_excludes_nonpaper_hand_penetration(self):
        native = {
            "metrics": {
                "end_obj_trans_err": 1.0,
                "xy_points_err": 2.0,
                "foot_sliding": 3.0,
                "contact_precision": 0.1,
                "contact_recall": 0.2,
                "contact_f1": 0.3,
                "contact_percent": 0.4,
                "human_pen_loss_infbagel": 4.0,
                "hand_pen_loss_omomo": 999.0,
                "mpjpe": 5.0,
                "trans_dist": 6.0,
                "obj_trans_dist": 7.0,
                "obj_rot_dist": 8.0,
            },
            "generation_metrics": {"fps": 9.0},
        }
        chois = {
            "metrics": {
                "FID": 10.0,
                "MatchingScore": 11.0,
                "R-Precision@1": 0.11,
                "R-Precision@2": 0.22,
                "R-Precision@3": 0.33,
                "Diversity": 12.0,
            },
        }
        timing = {"generation_metrics": {"fps": 13.0}}
        row = d2aa.table5_row(native, chois, timing)
        self.assertNotIn("hand_pen_loss_omomo", row)
        self.assertEqual(row["Pbody"], 4.0)
        self.assertIsNone(row["Rprec_paper_scalar"])
        self.assertEqual(row["R-Precision@3"], 0.33)

    def test_registry_contains_single_d2aa_preregistration(self):
        records = [
            json.loads(line)
            for line in (d2aa.REPO / "experiments" / "registry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        matches = [
            record for record in records
            if record["experiment_id"]
            == "p1-hoi-d2aa-table5-completion-preregister-s42-20260724"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["status"], "preregistered")


if __name__ == "__main__":
    unittest.main()
